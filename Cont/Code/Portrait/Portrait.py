import cv2
import numpy as np
from PIL import Image

class PortraitBlurrer:
    def __init__(self, max_blur=31, depth_threshold=120,
                 feather_strength=3, sharpen_strength=1):
        self.max_blur = max_blur
        # Ensure max_blur is odd and positive
        if self.max_blur % 2 == 0:
            self.max_blur += 1
        if self.max_blur <= 0:
            self.max_blur = 3 # Default odd positive

        self.depth_threshold = depth_threshold
        self.feather_strength = feather_strength
        self.sharpen_strength = sharpen_strength

    def refine_depth_map(self, depth_map):
        # Apply a bilateral filter to smooth depth while preserving edges
        refined_depth = cv2.bilateralFilter(depth_map, 9, 75, 75)
        return refined_depth

    def create_subject_mask(self, depth_map):
        _, mask = cv2.threshold(depth_map, self.depth_threshold, 255, cv2.THRESH_BINARY)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        ksize = self.feather_strength
        if ksize % 2 == 0:
            ksize += 1
        if ksize <= 0:
            ksize = 3

        mask = cv2.GaussianBlur(mask, (ksize, ksize), 0)
        return mask.astype(np.float32) / 255.0

    def sharpen_image(self, image):
        # Ensure sharpen_strength is not zero to avoid division issues later if needed
        strength = max(0.1, self.sharpen_strength) # Prevent zero strength

        # Simple sharpening kernel
        kernel = np.array([[-1, -1, -1],
                           [-1,  9, -1],
                           [-1, -1, -1]])

        # Apply the kernel - adjust strength application if needed
        # A common way is to blend the sharpened with original based on strength
        sharpened = cv2.filter2D(image, -1, kernel)

        # Blend sharpened and original based on strength
        # strength=1 means mostly sharpened, strength close to 0 means mostly original
        if strength != 1.0: # Avoid unnecessary work if strength is 1
           blended = cv2.addWeighted(image, 1.0 - (strength - 1.0) if strength > 1.0 else 1.0 ,
                                     sharpened, strength if strength <= 1.0 else 1.0, 0)
           # Basic clipping if values go out of range due to sharpening
           return np.clip(blended, 0, 255).astype(np.uint8)
        else:
           # Basic clipping if values go out of range due to sharpening
           return np.clip(sharpened, 0, 255).astype(np.uint8)


    def apply_blur(self, original_bgr, depth_map_array):
        # Resize depth map to match image dimensions
        depth_resized = cv2.resize(depth_map_array, (original_bgr.shape[1], original_bgr.shape[0]),
                                   interpolation=cv2.INTER_LINEAR)

        refined_depth = self.refine_depth_map(depth_resized)
        mask = self.create_subject_mask(refined_depth) # Float mask [0, 1]

        blurred = cv2.GaussianBlur(original_bgr, (self.max_blur, self.max_blur), 0)

        # Only sharpen if strength is significant
        if self.sharpen_strength > 0.05: # Threshold to avoid unnecessary computation
             sharpened_original = self.sharpen_image(original_bgr)
             # Blend sharpened subject with original based on mask
             foreground = sharpened_original * mask[:, :, np.newaxis] + \
                          original_bgr * (1 - mask[:, :, np.newaxis])
        else:
             foreground = original_bgr # Use original if no sharpening

        # Blend the (potentially sharpened) foreground with the blurred background
        background = blurred * (1 - mask[:, :, np.newaxis])
        # Combine the foreground (where mask is 1) and background (where mask is 0)
        # Note: Foreground already contains the original where it wasn't sharpened
        # A potentially better blend:
        result = original_bgr * mask[:, :, np.newaxis] + blurred * (1 - mask[:, :, np.newaxis])
        if self.sharpen_strength > 0.05:
            sharpened_subject_only = self.sharpen_image(original_bgr)
            # Apply sharpening only where the mask is high
            result = sharpened_subject_only * mask[:, :, np.newaxis] + result * (1 - mask[:, :, np.newaxis])

        # Ensure result is uint8
        final_result = np.clip(result, 0, 255).astype(np.uint8)

        # Return the final blurred image as a NumPy array (BGR)
        # Also return the refined depth map and the mask for potential display
        return final_result, refined_depth, (mask * 255).astype(np.uint8)


    def process_image(self, original_bgr_np, depth_image_pil):
        depth_map_array = np.array(depth_image_pil)

        if len(depth_map_array.shape) > 2:
            # Assuming input PIL depth map might be RGB, convert to grayscale
            depth_map_array = cv2.cvtColor(depth_map_array, cv2.COLOR_RGB2GRAY)
        elif len(depth_map_array.shape) == 2:
             # Already grayscale, ensure it's uint8 if necessary (though pipeline likely outputs it correctly)
             if depth_map_array.dtype != np.uint8:
                 # Normalize if it's float or other types before potential processing
                 if depth_map_array.max() > 1.0: # Basic check if it might be 0-255
                     depth_map_array = depth_map_array.astype(np.uint8)
                 else: # Assume 0-1 float, scale to 0-255
                      depth_map_array = (depth_map_array * 255).astype(np.uint8)


        # apply_blur now returns the result, depth map, and mask
        blurred_image_np, refined_depth_np, mask_np = self.apply_blur(original_bgr_np, depth_map_array)

        # Return the blurred image, the refined depth map (grayscale), and the mask (grayscale)
        return blurred_image_np, refined_depth_np, mask_np
