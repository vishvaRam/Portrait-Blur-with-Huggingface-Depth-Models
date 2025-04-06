import cv2
import numpy as np
from PIL import Image
from transformers import pipeline


class PortraitBlurrer:
    def __init__(self, max_blur=41, depth_threshold=120, depth_map_output="./Output/depth_map.png",
                 blurred_output="portrait_blurred.png", feather_strength=15, sharpen_strength=1.2):
        self.max_blur = max_blur  # Max blur kernel size
        self.depth_threshold = depth_threshold  # Threshold for separating subject & background
        self.depth_map_output = depth_map_output
        self.blurred_output = blurred_output
        self.feather_strength = feather_strength  # Controls the strength of the feathering
        self.sharpen_strength = sharpen_strength  # Controls the strength of the sharpening effect

    def refine_depth_map(self, depth_map):
        """Refine depth map using edge-aware filtering."""
        # Apply a bilateral filter to smooth depth while preserving edges
        refined_depth = cv2.bilateralFilter(depth_map, 9, 75, 75)
        return refined_depth

    def create_subject_mask(self, depth_map):
        """Create a binary mask for the subject based on depth threshold."""
        _, mask = cv2.threshold(depth_map, self.depth_threshold, 255, cv2.THRESH_BINARY)

        # Apply morphological operations to refine mask (reduce noise & improve edges)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Ensure feather_strength is odd and positive
        ksize = self.feather_strength
        if ksize % 2 == 0:
            ksize += 1
        if ksize <= 0:
            ksize = 3  # Set a default odd positive value

        # Apply Gaussian blur for feathering the edges
        mask = cv2.GaussianBlur(mask, (ksize, ksize), 0)
        return mask.astype(np.float32) / 255.0  # Normalize to 0-1

    def sharpen_image(self, image):
        """Sharpens the input image."""
        kernel = np.array([[-1, -1, -1],
                           [-1,  9, -1],
                           [-1, -1, -1]])
        return cv2.filter2D(image, -1, kernel * self.sharpen_strength)

    def apply_blur(self, image_path, depth_map_array):
        """Applies background blur while keeping the subject sharp."""
        original = cv2.imread(image_path)

        # Resize depth map to match image dimensions
        depth_resized = cv2.resize(depth_map_array, (original.shape[1], original.shape[0]),
                                   interpolation=cv2.INTER_LINEAR)

        # Refine the depth map
        refined_depth = self.refine_depth_map(depth_resized)

        # Create subject mask (now returns a float mask)
        mask = self.create_subject_mask(refined_depth)

        # Blur background
        blurred = cv2.GaussianBlur(original, (self.max_blur, self.max_blur), 0)

        # Sharpen the original image (subject)
        sharpened_original = self.sharpen_image(original)

        # Use the float mask to blend the sharpened original and blurred background
        foreground = sharpened_original * mask[:, :, np.newaxis]
        background = blurred * (1 - mask[:, :, np.newaxis])
        result = np.uint8(foreground + background)

        # Save the final output
        cv2.imwrite(self.blurred_output, result)
        print(f"Blurred portrait with slightly sharper subject saved as {self.blurred_output}")

    def process_image(self, image_path, depth_image_pil):
        """Process image with depth estimation and apply blur."""
        depth_map_array = np.array(depth_image_pil)

        # Ensure depth map is grayscale (single channel)
        if len(depth_map_array.shape) > 2:
            depth_map_array = cv2.cvtColor(depth_map_array, cv2.COLOR_RGB2GRAY)

        # Save the depth map for reference
        depth_pil = Image.fromarray(depth_map_array)
        depth_pil.save(self.depth_map_output)
        print(f"Depth map saved as {self.depth_map_output}")

        self.apply_blur(image_path, depth_map_array)


# ----------- DEPTH ESTIMATION & PROCESSING -----------

# Load depth estimation model
pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Large-hf")

# Load image
img_path = "./Images/flat.jpg"
image = Image.open(img_path)

# Perform depth estimation
output = pipe(image)
depth_image_pil = output["depth"]
print("Depth map generated.")

# Initialize and apply portrait blur with sharpening
portrait_blurrer = PortraitBlurrer(max_blur=31,feather_strength=3,
                                   sharpen_strength=1,
                                   blurred_output="./Output/portrait_blurred.png")  # Adjust sharpen_strength as needed
portrait_blurrer.process_image(img_path, depth_image_pil)
