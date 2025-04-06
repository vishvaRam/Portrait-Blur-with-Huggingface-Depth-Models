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
    page_title="Depth Blur Studio",
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
        from PortraitBlurrer import PortraitBlurrer # type: ignore
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
        # Use default precision (float32)
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
st.title("Depth Blur Studio 📸 (Streamlit)")
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
             # Ensure depth map is PIL Image
             if isinstance(depth_output, dict) and "depth" in depth_output:
                 depth_image_pil = depth_output["depth"]
                 if not isinstance(depth_image_pil, Image.Image):
                     # Attempt conversion if it's tensor/numpy (specifics might depend on pipeline output)
                     # This is a basic attempt; might need refinement based on actual output type
                     try:
                        depth_data = np.array(depth_image_pil)
                        # Normalize if needed (example: scale to 0-255)
                        depth_data = cv2.normalize(depth_data, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                        depth_image_pil = Image.fromarray(depth_data)
                     except Exception as conversion_e:
                         print(f"Could not convert depth output to PIL Image: {conversion_e}")
                         raise ValueError("Depth estimation did not return a usable PIL Image.")
             else:
                 # Handle cases where output might be directly the image or unexpected format
                 if isinstance(depth_output, Image.Image):
                     depth_image_pil = depth_output
                 else:
                      raise ValueError(f"Unexpected depth estimation output format: {type(depth_output)}")

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
        sharpen_strength=float(sharpen_val) # Use the passed sharpen value
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
    # Move success message display outside this function, near where results are shown
    # st.success(f"Processing finished in {processing_duration:.2f} seconds.")

    return blurred_pil, depth_pil, mask_pil, processing_duration # Return duration


# --- Initialize Session State --- (Do this early)
if 'results' not in st.session_state:
    st.session_state.results = None # Will store tuple (blurred, depth, mask) or None
if 'original_image_pil' not in st.session_state:
    st.session_state.original_image_pil = None
if 'processing_error_occurred' not in st.session_state:
    st.session_state.processing_error_occurred = False
if 'current_filename' not in st.session_state:
    st.session_state.current_filename = None
if 'last_process_duration' not in st.session_state:
    st.session_state.last_process_duration = None


# --- Sidebar for Controls ---
with st.sidebar: # Use 'with' notation for clarity
    st.title("Controls")
    uploaded_file = st.file_uploader(
        "Upload Portrait Image",
        type=["jpg", "png", "jpeg"],
        label_visibility="collapsed"
        )

    # --- Handle New Upload for Instant Display ---
    if uploaded_file is not None:
        # Check if it's a new file by comparing names
        if uploaded_file.name != st.session_state.get('current_filename', None):
            print(f"New file uploaded: {uploaded_file.name}. Loading for display.")
            try:
                # Load the new image immediately
                st.session_state.original_image_pil = Image.open(uploaded_file).convert("RGB")
                # Clear previous results, error state and duration
                st.session_state.results = None
                st.session_state.processing_error_occurred = False
                st.session_state.last_process_duration = None
                # Update the tracked filename
                st.session_state.current_filename = uploaded_file.name
            except Exception as e:
                st.error(f"Error loading image: {e}")
                # Clear states if loading failed
                st.session_state.original_image_pil = None
                st.session_state.results = None
                st.session_state.processing_error_occurred = False
                st.session_state.current_filename = None
                st.session_state.last_process_duration = None

    elif st.session_state.current_filename is not None:
        # If file uploader is cleared by the user (uploaded_file becomes None)
        print("File upload cleared.")
        st.session_state.original_image_pil = None
        st.session_state.results = None
        st.session_state.processing_error_occurred = False
        st.session_state.current_filename = None
        st.session_state.last_process_duration = None
    # --- End Handle New Upload ---


    st.markdown("---") # Separator
    st.markdown("**Adjust Parameters:**")
    slider_max_blur = st.slider("Blur Intensity (Kernel Size)", min_value=3, max_value=101, step=2, value=31)
    slider_depth_thr = st.slider("Subject Depth Threshold (Lower=Far away)", min_value=1, max_value=254, step=1, value=120)
    slider_feather = st.slider("Feathering (Mask Smoothness)", min_value=1, max_value=51, step=2, value=5) # <-- Default changed to 5
    # REMOVED: slider_sharpen = st.slider("Subject Sharpening Strength", min_value=0.0, max_value=2.5, step=0.1, value=1.0)
    st.markdown("---") # Separator

    # Button to trigger processing - disable if no file *loaded* in session state
    process_button = st.button(
        "Apply Blur",
        type="primary",
        disabled=(st.session_state.original_image_pil is None) # Disable if no original image is loaded
    )


# --- Main Area for Images ---
col1, col2 = st.columns(2) # Create two columns for Original | Result

# --- Handle Processing Trigger ---
if process_button: # Button is only enabled if original_image_pil exists
    if st.session_state.original_image_pil is not None:
        # Reset error flag on new processing attempt
        st.session_state.processing_error_occurred = False
        # Clear previous results and duration before showing spinner
        st.session_state.results = None
        st.session_state.last_process_duration = None

        with col2: # Show spinner in the results column
            with st.spinner('Applying blur... This may take a moment...'):
                results_output = process_image_blur(
                    pipeline_obj=pipe,
                    input_image_pil=st.session_state.original_image_pil, # Use the image from session state
                    max_blur_ksize=slider_max_blur,
                    depth_thresh=slider_depth_thr,
                    feather_ksize=slider_feather,
                    sharpen_val=1.0 # <-- Hardcoded sharpen value
                )

                # Check if processing returned successfully (4 values expected now)
                if results_output is not None and len(results_output) == 4:
                    # Unpack results and store duration separately
                    blurred_pil, depth_pil, mask_pil, duration = results_output
                    st.session_state.results = (blurred_pil, depth_pil, mask_pil) # Store tuple
                    st.session_state.last_process_duration = duration
                else:
                    # Processing failed (returned None or wrong number of items)
                    st.session_state.results = None # Ensure results are None
                    st.session_state.processing_error_occurred = True
                    st.session_state.last_process_duration = None

    else:
         # This case should technically not happen due to button disable logic, but good practice
         st.error("No image loaded to process.")


# --- Display Images based on Session State ---

# Display Original Image in Column 1 if available
if st.session_state.original_image_pil is not None:
    col1.image(st.session_state.original_image_pil, caption="Original Image", use_container_width=True)
else:
    col1.markdown("### Upload an image")
    col1.markdown("Use the sidebar controls to upload your portrait.")

# Display Results/Status in Column 2
if st.session_state.results is not None:
    # Check if the first element (blurred_img) is not None, indicating successful processing within the function
    blurred_img, depth_img, mask_img = st.session_state.results
    if blurred_img is not None:
        # Display success message with duration
        if st.session_state.last_process_duration is not None:
             st.success(f"Processing finished in {st.session_state.last_process_duration:.2f} seconds.")

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
            file_name=f"blurred_{st.session_state.current_filename or 'result'}.png", # Suggest filename based on original
            mime="image/png"                 # Set the MIME type for PNG
        )
        # --- END DOWNLOAD BUTTON ---

        # Optionally display depth and mask below the main images or in expanders
        with st.expander("Show Details (Depth Map & Mask)"):
            # Use columns inside expander for better layout if needed
            exp_col1, exp_col2 = st.columns(2)
            exp_col1.image(depth_img, caption="Refined Depth Map", use_container_width=True)
            exp_col2.image(mask_img, caption="Subject Mask", use_container_width=True)
    else:
        # This case might occur if results tuple was somehow malformed, treat as error
        st.session_state.processing_error_occurred = True # Mark as error if blurred_img is None but results tuple exists
        col2.error("An unexpected issue occurred during processing. Please check logs or try again.")


# Handle explicit error state OR "Ready to Process" state OR default state
if st.session_state.processing_error_occurred:
     # Display specific error message if processing failed after button press
     # The error might already be shown by st.error inside process_image_blur,
     # but this provides a fallback message in col2.
     col2.warning("Image processing failed. Check messages above or terminal logs.")

elif st.session_state.original_image_pil is not None and st.session_state.results is None:
    # If file is uploaded/loaded but not processed yet (and no error occurred)
    col2.markdown("### Ready to Process")
    col2.markdown("Adjust parameters in the sidebar (if needed) and click **Apply Blur**.")

elif st.session_state.original_image_pil is None:
    # Default state when no file is uploaded/loaded and nothing processed
     col2.markdown("### Results")
     col2.markdown("The processed image and details will appear here after uploading an image and clicking 'Apply Blur'.")