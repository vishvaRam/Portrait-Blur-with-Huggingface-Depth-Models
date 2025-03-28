import numpy as np
import torch
from transformers import DepthProImageProcessorFast, DepthProForDepthEstimation


class AppleDepthEstimator:
    def __init__(self, model_name="apple/DepthPro-hf", device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = torch.device("cpu")
        self.image_processor = DepthProImageProcessorFast.from_pretrained(model_name)
        self.model = DepthProForDepthEstimation.from_pretrained(model_name,).to(self.device).to(torch.bfloat16)

    def estimate_depth(self, image):
        inputs = self.image_processor(images=image, return_tensors="pt").to(self.device).to(torch.bfloat16)
        with torch.no_grad():
            outputs = self.model(**inputs)
        post_processed_output = self.image_processor.post_process_depth_estimation(
            outputs, target_sizes=[(image.height, image.width)]
        )
        depth = post_processed_output[0]["predicted_depth"]
        depth_normalized = (depth - depth.min()) / (depth.max() - depth.min())
        depth_uint8 = (depth_normalized * 255).detach().cpu().numpy().astype(np.uint8)
        return depth_uint8