from Model.appleDepthPro import AppleDepthEstimator
from Portrait.applePortrait import ApplePortraitBluer
from transformers import pipeline
from Portrait.depthAnythingPortrait import DepthAnythingPortraitBluer
from PIL import Image


image_path = "Images/flat.jpg"


# depth_estimator = AppleDepthEstimator()

# apple_portrait_bluer = ApplePortraitBluer(depth_map_output_path="Output/apple_depth_map.png",
#                                           blurred_output_path="Output/apple_portrait_blurred.png")

# apple_portrait_bluer.process_image(image_path, depth_estimator)


# --------------------------------------

pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Large-hf", use_fast=True,)

image = Image.open(image_path)

output = pipe(image)
depth_image_pil = output["depth"]
print(f"Depth map (PIL Image): {depth_image_pil}")

depth_anything_portrait_bluer = DepthAnythingPortraitBluer(depth_map_output_path="Output/depth_anything_depth_map.png",
                                                         blurred_output_path="Output/depth_anything_portrait_blurred.png")

depth_anything_portrait_bluer.process_image(image_path, depth_image_pil)