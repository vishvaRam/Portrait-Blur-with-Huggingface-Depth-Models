import cv2
import numpy as np
from PIL import Image


class ApplePortraitBluer:
    def __init__(self, max_blur=85, depth_map_output_path="Output/depth_map.png",
                 blurred_output_path="Output/portrait_blurred.png"):
        self.max_blur = max_blur
        self.depth_map_output_path = depth_map_output_path
        self.blurred_output_path = blurred_output_path

    def apply_portrait_blur(self, image_path, depth_map_array):
        original = cv2.imread(image_path)
        depth_resized = cv2.resize(depth_map_array, (original.shape[1], original.shape[0]))
        blur_strength = cv2.GaussianBlur(depth_resized, (5, 5), 0) / 255.0
        blurred = cv2.GaussianBlur(original, (self.max_blur, self.max_blur), 0)
        alpha = np.expand_dims(blur_strength, axis=-1)
        result = (original * (1 - alpha) + blurred * alpha).astype(np.uint8)
        cv2.imwrite(self.blurred_output_path, result)
        print(f"Blurred portrait saved as {self.blurred_output_path}")

    def process_image(self, image_path, depth_estimator):
        pil_image = Image.open(image_path)
        depth_map_array = depth_estimator.estimate_depth(pil_image)
        Image.fromarray(depth_map_array).save(self.depth_map_output_path)
        print(f"Depth map saved as {self.depth_map_output_path}")
        self.apply_portrait_blur(image_path, depth_map_array)