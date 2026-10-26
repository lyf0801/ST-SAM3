
import torch
import torch.nn as nn

from models.model_builder import _create_vision_backbone


class SAM3StandaloneImageEncoder(nn.Module):
    def __init__(self, visual_backbone, compile_visual=False, scalp=1):
        super().__init__()
        self.vision_backbone = (torch.compile(visual_backbone) if compile_visual else visual_backbone)
        self.scalp = scalp

    def forward(self, samples):
        sam3_features, sam3_pos, sam2_features, sam2_pos = self.vision_backbone.forward(samples)
        

        if self.scalp > 0:
            sam3_features = sam3_features[: -self.scalp]
            sam3_pos = sam3_pos[: -self.scalp]

        output = {
            "vision_features": sam3_features[-1],
            "vision_pos_enc": sam3_pos,
            "backbone_fpn": sam3_features,
        }
        return output
    
def build_sam3_image_encoder_only(compile=False):
    raw_vision_backbone = _create_vision_backbone(compile_mode=("default" if compile else None), enable_inst_interactivity=False)
    
    model = SAM3StandaloneImageEncoder(raw_vision_backbone, compile_visual=compile, scalp=1)
    

    return model
    

