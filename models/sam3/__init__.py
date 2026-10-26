# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

# 1. 删除这行导入，或者
# 2. 改为延迟导入（lazy import）
def build_sam3_image_model():
    from models.model_builder import build_sam3_image_model
    return build_sam3_image_model()

__version__ = "0.1.0"

__all__ = ["build_sam3_image_model"]
