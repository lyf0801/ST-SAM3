from __future__ import annotations

from pathlib import Path
import sys
from typing import Iterable, Sequence

import torch
from torch import nn
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DINOV3_CHECKPOINT = PROJECT_ROOT / "pretrained" / "dinov3_vits16_pretrain_lvd1689m.pth"
DEFAULT_MOBILE_SAM_CHECKPOINT = PROJECT_ROOT / "pretrained" / "mobile_sam.pt"
DINOV3_VENDOR_ROOT = PROJECT_ROOT / "third_party" / "dinov3"
MOBILE_SAM_VENDOR_ROOT = PROJECT_ROOT / "third_party" / "mobile_sam"
for vendor_root in (DINOV3_VENDOR_ROOT, MOBILE_SAM_VENDOR_ROOT):
    vendor_path = str(vendor_root)
    if vendor_path not in sys.path:
        sys.path.insert(0, vendor_path)

from dinov3.hub.backbones import dinov3_vits16  # noqa: E402
from mobile_sam import sam_model_registry  # noqa: E402
from mobile_sam.modeling.common import LayerNorm2d  # noqa: E402


class DINOv3MultiLayerSAMStudent(nn.Module):
    """DINOv3-S/16 student using native token-grid features for distillation."""

    def __init__(
        self,
        dinov3_checkpoint: str | Path | None = None,
        mobile_sam_checkpoint: str | Path | None = None,
        input_image_size: int = 448,
        intermediate_layers: Sequence[int] = (2, 5, 8, 11),
        feature_dim: int = 256,
        input_normalization: str = "eorssd",
    ) -> None:
        super().__init__()
        if input_image_size <= 0 or input_image_size % 16 != 0:
            raise ValueError("DINOv3 input_image_size must be a positive multiple of 16")
        if not intermediate_layers:
            raise ValueError("At least one DINOv3 intermediate layer is required")

        self.input_image_size = int(input_image_size)
        self.image_size = self.input_image_size
        self.intermediate_layers = tuple(int(index) for index in intermediate_layers)
        self.feature_dim = int(feature_dim)
        if input_normalization not in {"eorssd", "imagenet"}:
            raise ValueError(f"Unsupported input normalization: {input_normalization}")
        self.input_normalization = input_normalization
        self.pyramid_channels = tuple(self.feature_dim for _ in self.intermediate_layers)

        if dinov3_checkpoint is None:
            dinov3_checkpoint = DEFAULT_DINOV3_CHECKPOINT
        if mobile_sam_checkpoint is None:
            mobile_sam_checkpoint = DEFAULT_MOBILE_SAM_CHECKPOINT

        self.backbone = dinov3_vits16(pretrained=False)
        self.pretrained_tensors = self._load_dinov3_checkpoint(dinov3_checkpoint)

        self.feature_projections = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(384, self.feature_dim, kernel_size=1, bias=False),
                    LayerNorm2d(self.feature_dim),
                    nn.GELU(),
                )
                for _ in self.intermediate_layers
            ]
        )
        self.multi_layer_fusion = nn.Sequential(
            nn.Conv2d(
                len(self.intermediate_layers) * self.feature_dim,
                self.feature_dim,
                kernel_size=1,
                bias=False,
            ),
            LayerNorm2d(self.feature_dim),
            nn.GELU(),
            nn.Conv2d(self.feature_dim, self.feature_dim, kernel_size=3, padding=1, bias=False),
            LayerNorm2d(self.feature_dim),
            nn.GELU(),
        )


        mobile_sam_checkpoint = Path(mobile_sam_checkpoint).expanduser().resolve()
        if not mobile_sam_checkpoint.is_file():
            raise FileNotFoundError(f"MobileSAM checkpoint not found: {mobile_sam_checkpoint}")
        mobile_sam = sam_model_registry["vit_t"](
            checkpoint=str(mobile_sam_checkpoint),
            image_size=self.input_image_size,
        )
        prompt_encoder = mobile_sam.prompt_encoder
        self.mask_decoder = mobile_sam.mask_decoder
        self.no_prompt_embed = nn.Parameter(
            prompt_encoder.no_mask_embed.weight.detach().clone()
        )
        self.register_buffer(
            "decoder_image_pe",
            prompt_encoder.get_dense_pe().detach().clone(),
            persistent=True,
        )
        del mobile_sam

        # Existing STMatch loaders normalize every dataset with EORSSD statistics.
        self.register_buffer(
            "source_mean",
            torch.tensor([0.3412, 0.3798, 0.3583], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "source_std",
            torch.tensor([0.1148, 0.1042, 0.0990], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "imagenet_mean",
            torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "imagenet_std",
            torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )

    def _load_dinov3_checkpoint(self, checkpoint: str | Path) -> int:
        checkpoint = Path(checkpoint).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"DINOv3-S checkpoint not found: {checkpoint}")
        payload = torch.load(str(checkpoint), map_location="cpu")
        if isinstance(payload, dict) and "state_dict" in payload:
            payload = payload["state_dict"]
        if isinstance(payload, dict) and "model" in payload:
            payload = payload["model"]
        if not isinstance(payload, dict):
            raise TypeError(f"Unsupported DINOv3 checkpoint type: {type(payload)!r}")
        state_dict = {
            str(key).removeprefix("module.").removeprefix("backbone."): value
            for key, value in payload.items()
        }
        self.backbone.load_state_dict(state_dict, strict=True)
        return len(state_dict)

    def backbone_parameters(self) -> Iterable[nn.Parameter]:
        yield from self.backbone.parameters()

    def _prepare_dinov3_input(self, x: torch.Tensor) -> torch.Tensor:
        if self.input_normalization == "imagenet":
            return x.float()
        x01 = (x.float() * self.source_std + self.source_mean).clamp(0.0, 1.0)
        return (x01 - self.imagenet_mean) / self.imagenet_std

    def _encode(self, x: torch.Tensor) -> list[torch.Tensor]:
        features = self.backbone.get_intermediate_layers(
            self._prepare_dinov3_input(x),
            n=self.intermediate_layers,
            reshape=True,
            norm=True,
        )
        return list(features)

    def _decoder_inputs(
        self,
        batch_size: int,
        grid_size: tuple[int, int],
        dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sparse = torch.empty(
            (batch_size, 0, self.no_prompt_embed.shape[-1]),
            dtype=dtype,
            device=device,
        )
        dense = self.no_prompt_embed.reshape(1, -1, 1, 1).expand(
            batch_size, -1, grid_size[0], grid_size[1]
        )
        image_pe = F.interpolate(
            self.decoder_image_pe.to(device=device, dtype=dtype),
            size=grid_size,
            mode="bilinear",
            align_corners=False,
        )
        return sparse, dense.to(dtype=dtype), image_pe

    def forward(
        self,
        x: torch.Tensor,
        return_fp: bool = False,
        comp_drop: bool = False,
        return_feature: bool = False,
        return_logits: bool = False,
        **_: object,
    ):
        del return_fp, comp_drop
        input_hw = tuple(x.shape[-2:])
        backbone_features = self._encode(x)
        projected_features = [
            projection(feature)
            for projection, feature in zip(self.feature_projections, backbone_features)
        ]
        # All DINOv3 transformer blocks share the native 1/16 token grid.
        # The loss resizes each SAM3 teacher feature to this student grid, so no
        # learned spatial pyramid is inserted into the distillation branch.
        distillation_features = projected_features[:3]
        reference_hw = projected_features[-1].shape[-2:]
        aligned_features = [
            feature
            if feature.shape[-2:] == reference_hw
            else F.interpolate(feature, size=reference_hw, mode="bilinear", align_corners=False)
            for feature in projected_features
        ]
        fused_feature = self.multi_layer_fusion(torch.cat(aligned_features, dim=1))
        sparse, dense, image_pe = self._decoder_inputs(
            fused_feature.shape[0],
            reference_hw,
            fused_feature.dtype,
            fused_feature.device,
        )
        low_res_logits, _ = self.mask_decoder(
            image_embeddings=fused_feature,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=False,
        )
        logits = F.interpolate(
            low_res_logits,
            size=input_hw,
            mode="bilinear",
            align_corners=False,
        )
        prediction = logits if (self.training or return_logits) else torch.sigmoid(logits)
        if return_feature:
            return prediction, distillation_features
        return prediction


def get_dinov3_sam_student(
    dinov3_checkpoint: str | Path | None = None,
    mobile_sam_checkpoint: str | Path | None = None,
    img_size: int = 448,
    input_normalization: str = "eorssd",
) -> DINOv3MultiLayerSAMStudent:
    return DINOv3MultiLayerSAMStudent(
        dinov3_checkpoint=dinov3_checkpoint,
        mobile_sam_checkpoint=mobile_sam_checkpoint,
        input_image_size=img_size,
        input_normalization=input_normalization,
    )
