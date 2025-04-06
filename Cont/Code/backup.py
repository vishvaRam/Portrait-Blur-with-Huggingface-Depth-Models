# streamlit_app.py

import streamlit as st
import numpy as np
from PIL import Image
import cv2
import torch
from transformers import pipeline
import time
import os
from io import BytesIO # <-- IMPORT BytesIO

# --- Page Config (MUST BE FIRST st command) ---
# Set page config early
st.set_page_config(
    page_title="Portrait Background Blur",
    page_icon="📸",
    layout="wide"
)

# --- Import Custom Class ---
# Assuming PortraitBlurrer.py is in a subfolder 'Portrait' relative to this script
try:
    # If PortraitBlurrer is in ./Portrait/Portrait.py
    from Portrait.Portrait import PortraitBlurrer
except ImportError:
    # Fallback if PortraitBlurrer is in ./PortraitBlurrer.py
    try:
        from PortraitBlurrer import PortraitBlurrer
        # st.warning("Assuming PortraitBlurrer class is in the root directory.") # Optional warning
    except ImportError:
        st.error("Fatal Error: Could not find the PortraitBlurrer class. Please check the file structure and import path.")
        st.stop() # Stop execution if class can't be found


# --- Model Loading (Cached) ---
@st.cache_resource # Use cache_resource for non-data objects like models/pipelines
def load_depth_pipeline():
    """Loads the depth estimation pipeline and caches it. Returns tuple (pipeline, device_id)."""
    t_device = 0 if torch.cuda.is_available() else -1
    print(f"Attempting to load model on device: {'GPU (CUDA)' if t_device == 0 else 'CPU'}")
    try:
        # REMOVED torch_dtype argument - use default precision (float32)
        t_pipe = pipeline(task="depth-estimation",
                          model="depth-anything/Depth-Anything-V2-Large-hf",
                          device=t_device)
        print("Depth Anything V2 Large model loaded successfully.")
        return t_pipe, t_device # Return pipeline and device used
    except Exception as e:
        print(f"Error loading model: {e}")
        # Error will be displayed in the main app body after this function returns None
        return None, t_device # Return None for pipe on error

# Load the model via the cached function
pipe, device_used = load_depth_pipeline()

# --- Title and Model Status ---
# Display title and info AFTER attempting model load
st.title("Portrait Background Blur 📸 (Streamlit)")
st.markdown(
    "Upload a portrait image. The model will estimate depth and blur the background, keeping the subject sharp."
    "\n*Model: `depth-anything/Depth-Anything-V2-Large-hf`*"
)
st.caption(f"_(Using device: {'GPU (CUDA)' if device_used == 0 else 'CPU'})_") # Display device info

# Handle model loading failure AFTER potential UI elements like title
if pipe is None:
    st.error("Error loading depth estimation model. Application cannot proceed.")
    st.stop() # Stop if model loading failed


# --- Processing Function ---
def process_image_blur(pipeline_obj, input_image_pil, max_blur_ksize, depth_thresh, feather_ksize, sharpen_val):
    """
    Processes the image using the pipeline and PortraitBlurrer.
    Returns tuple: (blurred_pil, depth_pil, mask_pil) or (None, None, None) on failure.
    """
    print("Processing image...")
    processing_start_time = time.time()

    # 1. Convert PIL Image (RGB) to NumPy array (BGR for OpenCV)
    input_image_np_rgb = np.array(input_image_pil)
    original_bgr_np = cv2.cvtColor(input_image_np_rgb, cv2.COLOR_RGB2BGR)

    # 2. Perform depth estimation
    try:
        with torch.no_grad(): # Inference only
             depth_output = pipeline_obj(input_image_pil)
             depth_image_pil = depth_output["depth"]
        print("Depth map generated.")
    except Exception as e:
        print(f"Error during depth estimation: {e}")
        st.error(f"Depth estimation failed: {e}") # Show error in UI
        return None, None, None

    # 3. Initialize Blurrer and Process
    portrait_blurrer = PortraitBlurrer(
        max_blur=int(max_blur_ksize),
        depth_threshold=int(depth_thresh),
        feather_strength=int(feather_ksize),
        sharpen_strength=float(sharpen_val)
    )

    try:
        # process_image returns blurred_bgr, depth_gray, mask_gray
        blurred_bgr_np, refined_depth_np, mask_np = portrait_blurrer.process_image(
            original_bgr_np, depth_image_pil
        )
    except Exception as e:
         print(f"Error during blurring/sharpening: {e}")
         st.error(f"Image processing (blur/sharpen) failed: {e}") # Show error in UI
         return None, None, None

    # 4. Convert results back to RGB PIL Images for Streamlit display
    blurred_pil = Image.fromarray(cv2.cvtColor(blurred_bgr_np, cv2.COLOR_BGR2RGB))
    # Depth and mask are grayscale numpy, convert directly to PIL
    depth_pil = Image.fromarray(refined_depth_np)
    mask_pil = Image.fromarray(mask_np)

    processing_end_time = time.time()
    processing_duration = processing_end_time - processing_start_time
    print(f"Processing finished in {processing_duration:.2f} seconds.")
    st.success(f"Processing finished in {processing_duration:.2f} seconds.") # Show time in UI

    return blurred_pil, depth_pil, mask_pil


# --- Streamlit Interface Definition ---

# --- Sidebar for Controls ---
with st.sidebar: # Use 'with' notation for clarity
    st.title("Controls")
    uploaded_file = st.file_uploader(
        "Upload Portrait Image",
        type=["jpg", "png", "jpeg"],
        label_visibility="collapsed"
        )

    st.markdown("---") # Separator
    st.markdown("**Adjust Parameters:**")
    slider_max_blur = st.slider("Blur Intensity (Kernel Size)", min_value=3, max_value=101, step=2, value=31)
    slider_depth_thr = st.slider("Subject Depth Threshold (Lower=Closer)", min_value=1, max_value=254, step=1, value=120)
    slider_feather = st.slider("Feathering (Mask Smoothness)", min_value=1, max_value=51, step=2, value=15)
    slider_sharpen = st.slider("Subject Sharpening Strength", min_value=0.0, max_value=2.5, step=0.1, value=1.0)
    st.markdown("---") # Separator

    # Button to trigger processing - disable if no file is uploaded
    process_button = st.button("Apply Blur", type="primary", disabled=(uploaded_file is None))

# --- Main Area for Images ---
col1, col2 = st.columns(2) # Create two columns for Original | Result

# Initialize session state for results if it doesn't exist
# We store the results tuple (blurred, depth, mask) and the original image PIL object
if 'results' not in st.session_state:
    st.session_state.results = None
if 'original_image_pil' not in st.session_state:
    st.session_state.original_image_pil = None
if 'processing_error_occurred' not in st.session_state:
     st.session_state.processing_error_occurred = False # Use a more specific name

# --- Handle Processing Trigger ---
if process_button and uploaded_file is not None:
    # Reset error flag on new processing attempt
    st.session_state.processing_error_occurred = False
    # Process the image when the button is clicked
    st.session_state.original_image_pil = Image.open(uploaded_file).convert("RGB") # Store original for display

    with col2: # Show spinner in the results column
        with st.spinner('Applying blur... This may take a moment...'):
            results_tuple = process_image_blur(
                pipe,
                st.session_state.original_image_pil,
                slider_max_blur,
                slider_depth_thr,
                slider_feather,
                slider_sharpen
            )
            # Store results tuple (even if it contains None on failure)
            st.session_state.results = results_tuple
            # Check if processing actually failed
            if results_tuple == (None, None, None):
                 st.session_state.processing_error_occurred = True

elif not uploaded_file:
    # Clear previous results and image if no file is uploaded
    st.session_state.results = None
    st.session_state.original_image_pil = None
    st.session_state.processing_error_occurred = False


# --- Display Images based on Session State ---

# Display Original Image in Column 1 if available
if st.session_state.original_image_pil is not None:
    col1.image(st.session_state.original_image_pil, caption="Original Image", use_container_width=True)
else:
    col1.markdown("### Upload an image")
    col1.markdown("Use the sidebar controls to upload your portrait.")

# Display Results in Column 2 if available
if st.session_state.results is not None:
    blurred_img, depth_img, mask_img = st.session_state.results
    # Check if the first element (blurred_img) is not None, indicating success
    if blurred_img is not None:
        col2.image(blurred_img, caption="Blurred Background Result", use_container_width=True)

        # --- ADD DOWNLOAD BUTTON ---
        # 1. Convert PIL Image to Bytes
        buf = BytesIO()
        blurred_img.save(buf, format="PNG") # Save image to buffer in PNG format
        byte_im = buf.getvalue() # Get bytes from buffer

        # 2. Add Download Button
        col2.download_button(
            label="Download Blurred Image",
            data=byte_im,
            file_name="blurred_result.png", # Suggest a filename
            mime="image/png"                 # Set the MIME type for PNG
        )
        # --- END DOWNLOAD BUTTON ---


        # Optionally display depth and mask below the main images or in expanders
        with st.expander("Show Details (Depth Map & Mask)"):
            # Use columns inside expander for better layout if needed
            exp_col1, exp_col2 = st.columns(2)
            exp_col1.image(depth_img, caption="Refined Depth Map", use_container_width=True)
            exp_col2.image(mask_img, caption="Subject Mask", use_container_width=True)

    elif st.session_state.processing_error_occurred:
         # Display specific error message if processing failed after button press
         col2.error("Image processing failed. Please check terminal logs for specific error details or adjust parameters.")
    # No explicit 'else' needed here, as if results is None, the next block handles it

elif uploaded_file is not None:
    # If file is uploaded but not processed yet (button not clicked since upload)
    col2.markdown("### Ready to Process")
    col2.markdown("Adjust parameters in the sidebar (if needed) and click **Apply Blur**.")
else:
    # Default state when no file is uploaded and nothing processed
     col2.markdown("### Results")
     col2.markdown("The processed image and details will appear here.")