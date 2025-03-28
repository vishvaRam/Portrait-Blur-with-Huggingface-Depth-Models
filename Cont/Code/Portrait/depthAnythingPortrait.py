import cv2
import numpy as np
from PIL import Image

class DepthAnythingPortraitBluer:
    def __init__(self, max_blur=23, depth_map_output_path="depth_map.png",
                 blurred_output_path="portrait_blurred.png"):
        self.max_blur = max_blur
        self.depth_map_output_path = depth_map_output_path
        self.blurred_output_path = blurred_output_path

    def apply_portrait_blur(self, image_path, depth_map_array):
        original = cv2.imread(image_path)
        # Ensure depth map is in grayscale (single channel)
        if len(depth_map_array.shape) > 2:
            depth_map_array = cv2.cvtColor(depth_map_array, cv2.COLOR_RGB2GRAY)
        depth_resized = cv2.resize(depth_map_array, (original.shape[1], original.shape[0]), interpolation=cv2.INTER_LINEAR)

        # Invert the depth map to blur the foreground
        inverted_depth = 255 - depth_resized

        blur_strength = cv2.GaussianBlur(inverted_depth.astype(np.float32), (5, 5), 0) / 255.0
        blurred = cv2.GaussianBlur(original, (self.max_blur, self.max_blur), 0)
        alpha = np.expand_dims(blur_strength, axis=-1)
        result = (original * (1 - alpha) + blurred * alpha).astype(np.uint8)
        cv2.imwrite(self.blurred_output_path, result)
        print(f"Blurred portrait saved as {self.blurred_output_path}")

    def process_image(self, image_path, depth_image_pil):
        """
        Processes the image to apply portrait blur using a PIL Image object
        as the depth map.
        """
        # Convert PIL Image depth map to NumPy array
        depth_map_array = np.array(depth_image_pil)

        # Ensure depth map is in grayscale (single channel)
        if len(depth_map_array.shape) > 2:
            depth_map_array = cv2.cvtColor(depth_map_array, cv2.COLOR_RGB2GRAY)

        # Save the depth map (optional, for inspection)
        depth_pil = Image.fromarray(depth_map_array)
        depth_pil.save(self.depth_map_output_path)
        print(f"Depth map saved as {self.depth_map_output_path}")

        self.apply_portrait_blur(image_path, depth_map_array)