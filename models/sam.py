import logging
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models import register
from .mmseg.models.sam import ImageEncoderViT, MaskDecoder, TwoWayTransformer
from .sam3.build_sam3 import build_sam3_image_encoder_only

logger = logging.getLogger(__name__)
from .iou_loss import IOU
from typing import Any, Optional, Tuple





from torch.nn.init import trunc_normal_

def init_weights(layer):
    if type(layer) == nn.Conv2d:
        nn.init.normal_(layer.weight, mean=0.0, std=0.02)
        nn.init.constant_(layer.bias, 0.0)
    elif type(layer) == nn.Linear:
        nn.init.normal_(layer.weight, mean=0.0, std=0.02)
        nn.init.constant_(layer.bias, 0.0)
    elif type(layer) == nn.BatchNorm2d:
        # print(layer)
        nn.init.normal_(layer.weight, mean=1.0, std=0.02)
        nn.init.constant_(layer.bias, 0.0)

class BBCEWithLogitLoss(nn.Module):
    '''
    Balanced BCEWithLogitLoss
    '''
    def __init__(self):
        super(BBCEWithLogitLoss, self).__init__()

    def forward(self, pred, gt):
        eps = 1e-10
        count_pos = torch.sum(gt) + eps
        count_neg = torch.sum(1. - gt)
        ratio = count_neg / count_pos
        w_neg = count_pos / (count_pos + count_neg)

        bce1 = nn.BCEWithLogitsLoss(pos_weight=ratio)
        loss = w_neg * bce1(pred, gt)

        return loss

def _iou_loss(pred, target):
    pred = torch.sigmoid(pred)
    inter = (pred * target).sum(dim=(2, 3))
    union = (pred + target).sum(dim=(2, 3)) - inter
    iou = 1 - (inter / union)

    return iou.mean()

class PositionEmbeddingRandom(nn.Module):
    """
    Positional encoding using random spatial frequencies.
    """

    def __init__(self, num_pos_feats: int = 64, scale: Optional[float] = None) -> None:
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((2, num_pos_feats)),
        )

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        """Positionally encode points that are normalized to [0,1]."""
        # assuming coords are in [0, 1]^2 square and have d_1 x ... x d_n x 2 shape
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords
        # outputs d_1 x ... x d_n x C shape
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, size: int) -> torch.Tensor:
        """Generate positional encoding for a grid of the specified size."""
        h, w = size, size
        device: Any = self.positional_encoding_gaussian_matrix.device
        grid = torch.ones((h, w), device=device, dtype=torch.float32)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w

        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
        return pe.permute(2, 0, 1)  # C x H x W

    def forward_with_coords(self, coords_input: torch.Tensor, image_size: Tuple[int, int]) -> torch.Tensor:
        """Positionally encode relative shapes."""
        # coords_input: [B, N, 2]
        coords = coords_input.clone()
        coords[:, :, 0] = coords[:, :, 0] / image_size[1]
        coords[:, :, 1] = coords[:, :, 1] / image_size[0]
        return self._pe_encoding(coords)  # [B, N, C]


@register('sam')
class SAM(nn.Module):
    def __init__(self, inp_size=None, encoder_mode=None, loss=None,
        sam_mask_decoder_extra_args=None,
        use_high_res_features_in_sam=True,
        directly_add_no_mem_embed=True,
        iou_prediction_use_sigmoid=False,
        pred_obj_scores: bool = True,
        pred_obj_scores_mlp: bool = False,
        use_obj_ptrs_in_encoder=False,
        fixed_no_obj_ptr: bool = False,
        use_multimask_token_for_obj_ptr: bool = True,
                 ):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embed_dim = encoder_mode['embed_dim']
        # self.image_encoder = ImageEncoder(
        #     trunk = Hiera(),
        #     neck = FpnNeck(),
        # )
        self.image_encoder = build_sam3_image_encoder_only()

        #lyf add for unimatch
        self.feature_perturbation = nn.Dropout2d(0.5)
        self.binomial = torch.distributions.binomial.Binomial(probs=0.5) #lyf add: line 135 of https://github.com/LiheYoung/UniMatch-V2/blob/main/model/semseg/dpt.py


        self._bb_feat_sizes = [
            (256, 256),
            (128, 128),
            (64, 64),
        ]
        self.prompt_embed_dim = encoder_mode['prompt_embed_dim']
        self.sam_mask_decoder_extra_args = sam_mask_decoder_extra_args
        self.use_high_res_features_in_sam = use_high_res_features_in_sam
        self.directly_add_no_mem_embed = directly_add_no_mem_embed
        self.num_feature_levels = 3 if use_high_res_features_in_sam else 1
        self.iou_prediction_use_sigmoid = iou_prediction_use_sigmoid
        self.pred_obj_scores = pred_obj_scores
        self.use_obj_ptrs_in_encoder = use_obj_ptrs_in_encoder
        self.pred_obj_scores_mlp = pred_obj_scores_mlp
        self.fixed_no_obj_ptr = fixed_no_obj_ptr
        self.use_multimask_token_for_obj_ptr = use_multimask_token_for_obj_ptr
        if self.fixed_no_obj_ptr:
            assert self.pred_obj_scores
            assert self.use_obj_ptrs_in_encoder
        if self.pred_obj_scores and self.use_obj_ptrs_in_encoder:
            self.no_obj_ptr = torch.nn.Parameter(torch.zeros(1, self.hidden_dim))
            trunc_normal_(self.no_obj_ptr, std=0.02)
        self.fixed_no_obj_ptr = fixed_no_obj_ptr
        if self.fixed_no_obj_ptr:
            assert self.pred_obj_scores
            assert self.use_obj_ptrs_in_encoder

        self.mask_decoder = MaskDecoder(
            num_multimask_outputs=1,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=self.prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=self.prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
            use_high_res_features=self.use_high_res_features_in_sam,
            iou_prediction_use_sigmoid=self.iou_prediction_use_sigmoid,
            pred_obj_scores=self.pred_obj_scores,
            pred_obj_scores_mlp=self.pred_obj_scores_mlp,
            use_multimask_token_for_obj_ptr=self.use_multimask_token_for_obj_ptr,
            **(self.sam_mask_decoder_extra_args or {}),
        )

        if 'evp' in encoder_mode['name']:
            for k, p in self.encoder.named_parameters():
                if "prompt" not in k and "mask_decoder" not in k and "prompt_encoder" not in k:
                    p.requires_grad = False



        self.loss_mode = loss
        if self.loss_mode == 'bce':
            self.criterionBCE = torch.nn.BCEWithLogitsLoss()

        elif self.loss_mode == 'bbce':
            self.criterionBCE = BBCEWithLogitLoss()

        elif self.loss_mode == 'iou':
            self.criterionBCE = torch.nn.BCEWithLogitsLoss()
            self.criterionIOU = IOU()

        self.pe_layer = PositionEmbeddingRandom(encoder_mode['prompt_embed_dim'] // 2)
        self.inp_size = inp_size
        self.image_embedding_size = inp_size // encoder_mode['patch_size']
        self.no_mask_embed = nn.Embedding(1, encoder_mode['prompt_embed_dim'])

        # SAM3 提示点 Embedding 需要：0:背景 1:前景 2:TL(Box) 3:BR(Box) 
        # 以及 sampler 用到的 not_a_point_embed
        self.point_embeddings = nn.ModuleList(
            [nn.Embedding(1, encoder_mode['prompt_embed_dim']) for i in range(4)]
        )
        self.not_a_point_embed = nn.Embedding(1, encoder_mode['prompt_embed_dim'])

            
 
    def get_dense_pe(self) -> torch.Tensor:
        """
        Returns the positional encoding used to encode point prompts,
        applied to a dense set of points the shape of the image encoding.

        Returns:
          torch.Tensor: Positional encoding with shape
            1x(embed_dim)x(embedding_h)x(embedding_w)
        """
        return self.pe_layer(self.image_embedding_size).unsqueeze(0)


    def forward(self, input, return_fp=False, comp_drop=False, return_feature=False,  pseudo_saliency=None):
        # bs = 2
         # 1. 接收数据
        self.input = input.to(self.device)
        #print(self.input.size()) #torch.Size([8, 3, 448, 448])
        
        bs = self.input.shape[0]

        # bs = self.input.shape[0]
        # Embed prompts
        sparse_embeddings = torch.empty((bs, 0, self.prompt_embed_dim), device=self.input.device)
        dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
            bs, -1, self.image_embedding_size, self.image_embedding_size
        )

        self.features = self.image_encoder(self.input)
        
        #lyf add: for feature distillation
        # Preserve pre-processed backbone_fpn features for distillation.
        # The SAM MaskDecoder applies conv_s0/conv_s1 in-place which reduces
        # channel dimensions (e.g., 256 -> 32). To keep original 256-channel
        # features for distillation, store detached clones of the tensors here.
        self.features_for_distill = [f.detach().clone() for f in self.features["backbone_fpn"]]
        # for i in self.features.keys():
        #     if len(self.features[i]) == 0:
        #         continue
        #     if isinstance(self.features[i], (list, tuple)):
        #         for j in range(len(self.features[i])):
        #             print(i, j, self.features[i][j].shape)
        #     else:
        #         print(i, self.features[i].size())
        #     pass
        """
        vision_features torch.Size([8, 256, 32, 32])
        vision_pos_enc 0 torch.Size([8, 256, 128, 128])
        vision_pos_enc 1 torch.Size([8, 256, 64, 64])
        vision_pos_enc 2 torch.Size([8, 256, 32, 32])
        backbone_fpn 0 torch.Size([8, 256, 128, 128])
        backbone_fpn 1 torch.Size([8, 256, 64, 64])
        backbone_fpn 2 torch.Size([8, 256, 32, 32]
        """
        if self.use_high_res_features_in_sam:
            # precompute projected level 0 and level 1 features in SAM decoder
            # to avoid running it again on every SAM click
            self.features["backbone_fpn"][0] = self.mask_decoder.conv_s0(
                self.features["backbone_fpn"][0]
            )
            self.features["backbone_fpn"][1] = self.mask_decoder.conv_s1(
                self.features["backbone_fpn"][1]
            )
        self.features = self.features.copy()
        assert len(self.features["backbone_fpn"]) == len(self.features["vision_pos_enc"])
        assert len(self.features["backbone_fpn"]) >= self.num_feature_levels

        feature_maps = self.features["backbone_fpn"][-self.num_feature_levels :]
        vision_pos_embeds = self.features["vision_pos_enc"][-self.num_feature_levels :]

        feat_sizes = [(x.shape[-2], x.shape[-1]) for x in vision_pos_embeds]
        # flatten NxCxHxW to HWxNxC
        vision_feats = [x.flatten(2).permute(2, 0, 1) for x in feature_maps]
        # vision_pos_embeds = [x.flatten(2).permute(2, 0, 1) for x in vision_pos_embeds]

        # _, vision_feats, _, _ = self.features, vision_feats, vision_pos_embeds, feat_sizes

        if self.directly_add_no_mem_embed:
            # vision_feats[-1] = vision_feats[-1] + torch.nn.Parameter(torch.zeros(1, 1, 256).to('cuda'))
            vision_feats[-1] = vision_feats[-1] + torch.zeros(1, 1, 256, device=self.device)

        # feats = [
        #     feat.permute(1, 2, 0).view(2, -1, *feat_size)
        #     for feat, feat_size in zip(vision_feats[::-1], self._bb_feat_sizes[::-1])
        # ][::-1]
        feats = [feat.permute(1, 2, 0).view(bs, -1, *feat_size) # 显式使用 bs 而不是 2
                    for feat, feat_size in zip(vision_feats[::-1], feat_sizes[::-1])
                ][::-1]

        self._features = {"image_embed": feats[-1], "high_res_feats": feats[:-1]}
        # high_res_features = [
        #     feat_level[-1].unsqueeze(0)
        #     for feat_level in self._features["high_res_feats"]
        # ]
        high_res_features = self._features["high_res_feats"]
        
        #for i in high_res_features:
        #    print(i.size())
        #print("-----------------------------")
        """
        torch.Size([8, 32, 128, 128])
        torch.Size([8, 64, 64, 64])
        """
        image_embed = self._features["image_embed"]
        """
        torch.Size([8, 256, 32, 32])
        32x32 是 image_embedding_size
        """
        #print(image_embed.size())
        H_feat, W_feat = image_embed.shape[-2], image_embed.shape[-1]

        # 1. 准备 Sparse Embeddings (空)
        sparse_embeddings = torch.empty((bs, 0, self.prompt_embed_dim), device=self.input.device)
        
        # 2. 准备 Dense Embeddings (并动态调整尺寸)
        dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
            bs, -1, self.image_embedding_size, self.image_embedding_size
        )
        # 如果尺寸不一致，进行插值缩放
        if dense_embeddings.shape[-2] != H_feat or dense_embeddings.shape[-1] != W_feat:
            dense_embeddings = F.interpolate(
                dense_embeddings, 
                size=(H_feat, W_feat), 
                mode='bilinear', 
                align_corners=False
            )

        # 3. 准备 Positional Encodings (并动态调整尺寸)
        image_pe = self.get_dense_pe()
        if image_pe.shape[-2] != H_feat or image_pe.shape[-1] != W_feat:
            image_pe = F.interpolate(
                image_pe, 
                size=(H_feat, W_feat), 
                mode='bilinear', 
                align_corners=False
            )
        
        #lyf add feature perturbation for unimatch
        if return_fp:
            image_embed = torch.cat((image_embed, self.feature_perturbation(image_embed)), dim=0)
            sparse_embeddings = torch.cat((sparse_embeddings, self.feature_perturbation(sparse_embeddings)), dim=0)
            dense_embeddings = torch.cat((dense_embeddings, self.feature_perturbation(dense_embeddings)), dim=0)
            high_res_features = [torch.cat((feat, self.feature_perturbation(feat)), dim=0) for feat in high_res_features]

        #lyf add for unimatchv2 complementary dropout
        if comp_drop:
            bs = image_embed.shape[0]
            # 对 image_embed 与每一层 high_res_features 分别应用互补 Dropout
            new_feats = []
            all_feats = [image_embed] + list(high_res_features)
            for feat in all_feats:
                dim = feat.shape[1]  # Channels
                # create complementary masks for first and second halves of the batch
                dropout_mask1 = self.binomial.sample((bs // 2, dim)).to(feat.device) * 2.0
                dropout_mask2 = 2.0 - dropout_mask1

                # 随机保留一部分通道不进行 Dropout
                dropout_prob = 0.5
                num_kept = int(bs // 2 * (1 - dropout_prob))
                if num_kept > 0:
                    kept_indexes = torch.randperm(bs // 2)[:num_kept]
                    dropout_mask1[kept_indexes, :] = 1.0
                    dropout_mask2[kept_indexes, :] = 1.0

                dropout_mask = torch.cat((dropout_mask1, dropout_mask2), dim=0)  # [B, C]
                feat = feat * dropout_mask.unsqueeze(-1).unsqueeze(-1)  # [B, C, H, W]
                new_feats.append(feat)

            image_embed = new_feats[0]
            high_res_features = new_feats[1:]

        # Predict masks
        low_res_masks, iou_predictions,sam_output_tokens,object_score_logits, = self.mask_decoder(
            image_embeddings=image_embed,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=True,
            repeat_image=False,  # the image is already batched
            high_res_features=high_res_features,
        )

        # Upscale the masks to the original image resolution
        self.pred_mask  = self.postprocess_masks(low_res_masks, self.inp_size, self.inp_size)
        # masks = torch.stack([
        #     self.postprocess_masks(low_res_masks[i], self.inp_size)
        # for i in range(low_res_masks.shape[0])
        # ])
        # low_res_masks = torch.clamp(low_res_masks, -32.0, 32.0)
        # if not return_logits:
        # if not False:
        #     # masks = masks > self.mask_threshold
        #     masks = masks > 0.0
     
        if self.training:  # lyf add:
            # UniMatch style return for feature perturbation
            if return_fp:
                out, out_fp = self.pred_mask.chunk(2, dim=0)
                return out, out_fp
            if return_feature:
                return self.pred_mask, self.features_for_distill
            return self.pred_mask  # 训练模式下返回原始预测，用于计算伪标签
        # self.eval()模式
        else:
            if return_feature:
                return torch.sigmoid(self.pred_mask), self.features_for_distill
            return torch.sigmoid(self.pred_mask)
        # self.pred_mask = masks
    

    def infer(self, input):
        bs = input.shape[0]

        # Run Encoder
        self.features = self.image_encoder(input)
        if self.use_high_res_features_in_sam:
            self.features["backbone_fpn"][0] = self.mask_decoder.conv_s0(
                self.features["backbone_fpn"][0]
            )
            self.features["backbone_fpn"][1] = self.mask_decoder.conv_s1(
                self.features["backbone_fpn"][1]
            )
        self.features = self.features.copy()
        
        feature_maps = self.features["backbone_fpn"][-self.num_feature_levels :]
        vision_pos_embeds = self.features["vision_pos_enc"][-self.num_feature_levels :]
        feat_sizes = [(x.shape[-2], x.shape[-1]) for x in vision_pos_embeds]
        vision_feats = [x.flatten(2).permute(2, 0, 1) for x in feature_maps]

        if self.directly_add_no_mem_embed:
            vision_feats[-1] = vision_feats[-1] + torch.zeros(1, 1, 256, device=input.device)

        feats = [
            feat.permute(1, 2, 0).view(bs, -1, *feat_size)
            for feat, feat_size in zip(vision_feats[::-1], feat_sizes[::-1])
        ][::-1]

        self._features = {"image_embed": feats[-1], "high_res_feats": feats[:-1]}
        high_res_features = self._features["high_res_feats"]
        
        image_embed = self._features["image_embed"]
        H_feat, W_feat = image_embed.shape[-2], image_embed.shape[-1]
        
        # Resize Embeddings
        sparse_embeddings = torch.empty((bs, 0, self.prompt_embed_dim), device=input.device)
        dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
            bs, -1, self.image_embedding_size, self.image_embedding_size
        )
        if dense_embeddings.shape[-2] != H_feat or dense_embeddings.shape[-1] != W_feat:
            dense_embeddings = F.interpolate(dense_embeddings, size=(H_feat, W_feat), mode='bilinear', align_corners=False)

        image_pe = self.get_dense_pe()
        if image_pe.shape[-2] != H_feat or image_pe.shape[-1] != W_feat:
            image_pe = F.interpolate(image_pe, size=(H_feat, W_feat), mode='bilinear', align_corners=False)

        low_res_masks, _, _, _, = self.mask_decoder(
            image_embeddings=image_embed,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=True,
            repeat_image=False,
            high_res_features=high_res_features,
        )

        masks = self.postprocess_masks(low_res_masks, self.inp_size, self.inp_size)
        return masks


    def postprocess_masks(
        self,
        masks: torch.Tensor,
        input_size: int,      # 注意：这里传进来的是 int (448)
        original_size: int,
    ) -> torch.Tensor:
        # 直接插值到目标尺寸 (448, 448)
        masks = F.interpolate(
            masks,
            (input_size, input_size),
            mode="bilinear",
            align_corners=False,
        )
        return masks
    
    # def postprocess_masks(
    #     self,
    #     masks: torch.Tensor,
    #     input_size: Tuple[int, ...],
    #     original_size: Tuple[int, ...],
    # ) -> torch.Tensor:
    #     """
    #     Remove padding and upscale masks to the original image size.

    #     Arguments:
    #       masks (torch.Tensor): Batched masks from the mask_decoder,
    #         in BxCxHxW format.
    #       input_size (tuple(int, int)): The size of the image input to the
    #         model, in (H, W) format. Used to remove padding.
    #       original_size (tuple(int, int)): The original size of the image
    #         before resizing for input to the model, in (H, W) format.

    #     Returns:
    #       (torch.Tensor): Batched masks in BxCxHxW format, where (H, W)
    #         is given by original_size.
    #     """
        
    #     masks = F.interpolate(
    #         masks,
    #         (self.image_encoder.img_size, self.image_encoder.img_size),
    #         mode="bilinear",
    #         align_corners=False,
    #     )
    #     masks = masks[..., : input_size, : input_size]
    #     masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
    #     return masks

    def postprocess_masks2(self, masks: torch.Tensor, orig_hw) -> torch.Tensor:
        """
        Perform PostProcessing on output masks.
        """
        from .sam2.utils.misc import get_connected_components

        masks = masks.float()
        # if self.max_hole_area > 0:
        if self.max_hole_area > 0:
            # Holes are those connected components in background with area <= self.fill_hole_area
            # (background regions are those with mask scores <= self.mask_threshold)
            mask_flat = masks.flatten(0, 1).unsqueeze(1)  # flatten as 1-channel image
            labels, areas = get_connected_components(mask_flat <= self.mask_threshold)
            is_hole = (labels > 0) & (areas <= self.max_hole_area)
            is_hole = is_hole.reshape_as(masks)
            # We fill holes with a small positive mask score (10.0) to change them to foreground.
            masks = torch.where(is_hole, self.mask_threshold + 10.0, masks)

        if self.max_sprinkle_area > 0:
            labels, areas = get_connected_components(mask_flat > self.mask_threshold)
            is_hole = (labels > 0) & (areas <= self.max_sprinkle_area)
            is_hole = is_hole.reshape_as(masks)
            # We fill holes with negative mask score (-10.0) to change them to background.
            masks = torch.where(is_hole, self.mask_threshold - 10.0, masks)

        masks = F.interpolate(masks, orig_hw, mode="bilinear", align_corners=False)
        return masks

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
