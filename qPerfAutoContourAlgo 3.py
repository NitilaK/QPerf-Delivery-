import os
import cv2
import numpy as np
import pydicom
import matplotlib.pyplot as plt
import time 
from scipy.interpolate import splprep, splev

# ---- Load and Normalize 1 DICOM at a time ----
def load_dicom_image(dicom_path):
    dicom = pydicom.dcmread(dicom_path)
    image = dicom.pixel_array.astype(np.float32)

    # Clip to 1st–99th percentile to avoid extreme outliers
    low, high = np.percentile(image, (1, 99))
    image_clipped = np.clip(image, low, high)

    # Normalize to [0,1]
    norm_image = (image_clipped - low) / (high - low + 1e-6)

    # Convert to 8-bit and apply histogram equalization
    image_uint8 = (norm_image * 255).astype(np.uint8)
    image_eq = cv2.equalizeHist(image_uint8)

    # Float32 [0,1] for downstream processing
    image_eq_norm = image_eq.astype(np.float32) / 255.0

    return image_eq_norm, image


# ---- Normalize image from 4D array (same processing as load_dicom_image) ----
def normalize_image_from_array(image):
    """
    Apply the same normalization as load_dicom_image but from a raw pixel array.
    
    Parameters
    - image: 2D numpy array (raw pixel values from 4D matrix)
    
    Returns
    - image_eq_norm: Normalized and histogram-equalized image [0,1]
    - image: Original image (float32)
    """
    image = image.astype(np.float32)
    
    # Clip to 1st–99th percentile to avoid extreme outliers
    low, high = np.percentile(image, (1, 99))
    image_clipped = np.clip(image, low, high)

    # Normalize to [0,1]
    norm_image = (image_clipped - low) / (high - low + 1e-6)

    # Convert to 8-bit and apply histogram equalization
    image_uint8 = (norm_image * 255).astype(np.uint8)
    image_eq = cv2.equalizeHist(image_uint8)

    # Float32 [0,1] for downstream processing
    image_eq_norm = image_eq.astype(np.float32) / 255.0

    return image_eq_norm, image


# ---- Compute SI curve from 4D array for a single slice ----
def compute_si_curve_from_array(image_matrix, slice_idx, mask):
    """
    Compute signal intensity (SI) curve for a mask region across all dynamics of a slice.
    
    Parameters
    - image_matrix: 4D numpy array (num_slices, num_dynamics, rows, cols)
    - slice_idx: Index of the slice to process
    - mask: Binary mask of region (LV, RV, or myocardium)
    
    Returns
    - si_values: List of mean intensities (top 10% pixels) per dynamic
    """
    num_dynamics = image_matrix.shape[1]  # dynamics is second dimension
    si_values = []
    
    for dyn_idx in range(num_dynamics):
        raw_image = image_matrix[slice_idx, dyn_idx, :, :]
        masked_pixels = raw_image[mask == 255]
        
        if masked_pixels.size == 0:
            mean_intensity = 0.0
        else:
            threshold = np.percentile(masked_pixels, 90)
            top_pixels = masked_pixels[masked_pixels >= threshold]
            mean_intensity = float(np.mean(top_pixels)) if top_pixels.size else 0.0
        
        si_values.append(mean_intensity)
    
    return si_values


# ---- SI Curve & Peak Finder from 4D array ----
def plot_si_curve_and_find_peak_from_array(image_matrix, slice_idx, lv_mask, rv_mask=None):
    """
    Compute and plot SI curves for LV (and RV) across dynamics, find peak LV frame.
    
    Parameters
    - image_matrix: 4D numpy array (num_dynamics, num_slices, rows, cols)
    - slice_idx: Index of the slice to process
    - lv_mask: Binary LV mask
    - rv_mask: Optional binary RV mask
    
    Returns
    - peak_idx: Index of dynamic with max LV SI
    - cycle_pos: Normalized position (0–1) of LV peak
    """
    num_dynamics = image_matrix.shape[0]
    lv_si = []
    rv_si = []
    
    for dyn_idx in range(num_dynamics):
        raw_image = image_matrix[dyn_idx, slice_idx, :, :]
        
        # LV SI
        lv_pixels = raw_image[lv_mask == 255]
        if lv_pixels.size == 0:
            lv_val = 0.0
        else:
            lv_thresh = np.percentile(lv_pixels, 90)
            lv_top = lv_pixels[lv_pixels >= lv_thresh]
            lv_val = float(np.mean(lv_top)) if lv_top.size else 0.0
        lv_si.append(lv_val)
        
        # RV SI (if mask provided)
        if rv_mask is not None:
            rv_pixels = raw_image[rv_mask == 255]
            if rv_pixels.size == 0:
                rv_val = 0.0
            else:
                rv_thresh = np.percentile(rv_pixels, 90)
                rv_top = rv_pixels[rv_pixels >= rv_thresh]
                rv_val = float(np.mean(rv_top)) if rv_top.size else 0.0
            rv_si.append(rv_val)
    
    # Plot SI curves
    plt.figure()
    plt.plot(range(len(lv_si)), lv_si, marker='o', label="LV (Top 10%)")
    if rv_mask is not None:
        plt.plot(range(len(rv_si)), rv_si, marker='s', label="RV (Top 10%)")
    plt.title("Signal Intensity Curve")
    plt.xlabel("Dynamic Index")
    plt.ylabel("Mean Top 10% Raw Intensity")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # LV Peak detection
    peak_idx = int(np.argmax(lv_si))
    cycle_pos = peak_idx / (len(lv_si) - 1) if len(lv_si) > 1 else 0.5
    
    print(f"LV peak intensity at dynamic {peak_idx} (SI={lv_si[peak_idx]:.2f}, Pos={cycle_pos:.2f})")
    return peak_idx, cycle_pos

# ---- Auto-detect LV seed point ----
# ---- Auto-detect LV seed point and optionally return all fitted ellipses ----
def auto_detect_seed_point(image, debug=False, return_contours=False):
    image_uint8 = (image * 255).astype(np.uint8)
    h, w = image.shape
    image_center = np.array([w / 2, h / 2])
    high_thresh = np.percentile(image_uint8, 97)
    _, thresh = cv2.threshold(image_uint8, high_thresh, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_score = -np.inf
    best_centroid = None
    max_dist = np.linalg.norm(np.array([w, h]) / 2)

    fitted_ellipses = []
    for cnt in contours:
        if len(cnt) < 5:
            continue
        ellipse = cv2.fitEllipse(cnt)
        (cx, cy), (MA, ma), angle = ellipse
        area = cv2.contourArea(cnt)
        if area < 100:
            continue
        eccentricity = np.sqrt(1 - (min(MA, ma) / max(MA, ma)) ** 2)
        mask = np.zeros_like(image_uint8)
        cv2.drawContours(mask, [cnt], -1, 255, -1)
        mean_val = cv2.mean(image_uint8, mask=mask)[0]
        dist = np.linalg.norm(np.array([cx, cy]) - image_center)
        dist_norm = dist / max_dist
        if dist_norm > 0.4:
            continue
        score = ( (1.0 - dist_norm) ** 3 * 100 ) + (0.1 * mean_val) + (0.1 * (1.0 - eccentricity) * 50)
        fitted_ellipses.append({"contour": cnt, "centroid": (int(cx), int(cy)), "score": score, "ecc": eccentricity})
        if score > best_score:
            best_score = score
            best_centroid = (int(cx), int(cy))

    if best_centroid is None:
        best_centroid = (w // 2, h // 2)
        print("Fallback: No central region detected. Using image center.")
    else:
        print(f"Auto-detected LV seed point at: {best_centroid}")

    if debug:
        debug_img = cv2.cvtColor(image_uint8, cv2.COLOR_GRAY2BGR)
        for f in fitted_ellipses:
            cv2.ellipse(debug_img, cv2.fitEllipse(f["contour"]), (230,224,176), 1)
        cv2.circle(debug_img, best_centroid, 4, (255,144,30), -1)
        plt.figure(figsize=(6,6))
        plt.imshow(cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB))
        plt.title("Elliptical Seed Detection (LV)")
        plt.axis("off")
        plt.show()

    return (best_centroid, fitted_ellipses) if return_contours else best_centroid

# ---- Auto-detect RV seed using same ellipses from LV detection ----
def auto_detect_rv_seed_near_lv(lv_seed, fitted_ellipses, lv_mask=None, myo_mask=None, debug=False):
    lvx, lvy = lv_seed
    candidates = []
    for f in fitted_ellipses:
        cx, cy = f["centroid"]
        if cx >= lvx:  # Skip right of LV
            continue
        # Skip overlapping LV region
        if lv_mask is not None and lv_mask[cy, cx] > 0:
            continue
        # Skip touching myocardium
        if myo_mask is not None:
            mask_temp = np.zeros_like(myo_mask)
            cv2.drawContours(mask_temp, [f["contour"]], -1, 255, -1)
            if np.count_nonzero(cv2.bitwise_and(mask_temp, myo_mask)) > 0:
                continue
        candidates.append(f)

    if not candidates:
        rv_seed = (max(lvx - 30, 0), lvy)  # fallback left
        print("No RV candidate from LV ellipses; fallback ->", rv_seed)
        return rv_seed

    # pick the candidate closest to LV (left but near)
    candidates.sort(key=lambda x: np.hypot(x["centroid"][0]-lvx, x["centroid"][1]-lvy))
    rv_seed = candidates[0]["centroid"]

    if debug:
        print(f"Auto-detected RV seed from LV ellipses: {rv_seed}")
    return rv_seed




# ---- Region Growing ----
def region_grow(image, seed_point, threshold=0.07):
    """
    Perform simple region growing segmentation starting from a seed point.

    Parameters
    - image: 2D numpy array (grayscale, usually normalized to 0..1).
    - seed_point: (x, y) tuple giving the starting pixel for growing.
    - threshold: intensity difference tolerance for including neighbors.

    Returns
    - Binary mask (uint8 image, values 0 or 255) of the grown region.
    """

    # Get image dimensions
    h, w = image.shape

    # Initialize an empty binary mask (same size as input image)
    mask = np.zeros_like(image, dtype=np.uint8)

    # Track visited pixels to avoid re-checking the same location
    visited = np.zeros_like(image, dtype=bool)

    # Ensure the seed point lies inside image bounds (clamp to valid coords)
    sx = int(np.clip(seed_point[0], 0, w - 1))  # x = column index
    sy = int(np.clip(seed_point[1], 0, h - 1))  # y = row index

    # Reference intensity value at the seed point
    seed_val = image[sy, sx]

    # Initialize stack (depth-first search) with the seed point
    stack = [(sx, sy)]

    # Main region growing loop
    while stack:
        # Take one pixel from the stack
        x, y = stack.pop()

        # Skip if pixel is outside image bounds or already visited
        if x < 0 or x >= w or y < 0 or y >= h or visited[y, x]:
            continue

        # Mark pixel as visited
        visited[y, x] = True

        # Compare pixel intensity to seed intensity
        if abs(image[y, x] - seed_val) < threshold:
            # If within threshold: mark pixel as part of the region
            mask[y, x] = 1

            # Push 4-connected neighbors (left, right, up, down) onto stack
            stack.extend([
                (x - 1, y), (x + 1, y),  # left, right
                (x, y - 1), (x, y + 1)   # up, down
            ])

    # Convert binary mask (0/1) to standard 0/255 uint8 image for visualization
    return (mask * 255).astype(np.uint8)

# ---- Myocardium Segmentation (Edge-Based) ----
def segment_myocardium_edge_based(image, lv_mask):
    """
    Segment myocardium (muscle wall around LV) using edge detection and LV mask.

    Parameters
    - image: 2D grayscale image (float normalized 0..1).
    - lv_mask: Binary mask (uint8, 0/255) of LV region.

    Returns
    - myocardium_mask: Binary mask of detected myocardium region.
    """

    # Convert normalized float image to 8-bit grayscale for OpenCV ops
    image_uint8 = (image * 255).astype(np.uint8)

    # Detect edges in the image (low=30, high=100 thresholds for Canny)
    edges = cv2.Canny(image_uint8, 30, 100)

    # Dilate LV mask outward to form a ring around LV
    dilated = cv2.dilate(lv_mask, np.ones((15, 15), np.uint8), iterations=1)

    # Subtract LV from dilated LV to isolate the ring region around LV
    ring = cv2.subtract(dilated, lv_mask)

    # Keep only edges that fall inside the ring (possible myocardium boundary)
    myocardium_edges = cv2.bitwise_and(edges, edges, mask=ring)

    # Use morphological ops to thicken and close gaps in myocardium edges
    kernel = np.ones((5, 5), np.uint8)
    myocardium_mask = cv2.dilate(myocardium_edges, kernel, iterations=1)
    myocardium_mask = cv2.morphologyEx(myocardium_mask, cv2.MORPH_CLOSE, kernel)

    return myocardium_mask


# ---- Overlap QC ----
def rv_myo_overlap_ok(rv_mask, myo_mask):
    """
    Quality check: ensure RV overlaps (touches) myocardium mask.

    Parameters
    - rv_mask: Binary RV region mask (uint8).
    - myo_mask: Binary myocardium region mask (uint8).

    Returns
    - True if there is any overlap; False otherwise.
    """

    # Slightly dilate RV mask so minor gaps don’t break connectivity
    kernel = np.ones((3, 3), np.uint8)
    rv_dil = cv2.dilate(rv_mask, kernel, iterations=1)

    # Find overlapping region between RV and myocardium
    overlap = cv2.bitwise_and(rv_dil, myo_mask)

    # If overlap has nonzero pixels, return True
    return np.count_nonzero(overlap) > 0


# ---- Draw Contours ----
def draw_contours(image, lv_mask, myo_mask, rv_mask=None):
    """
    Draw LV, RV, and myocardium contours on top of the image.

    Parameters
    - image: 2D grayscale image (float or uint8).
    - lv_mask: Binary LV mask.
    - myo_mask: Binary myocardium mask.
    - rv_mask: Optional binary RV mask.

    Displays an overlay figure with contours drawn in different colors.
    """

    # Normalize grayscale to 0–255 and convert to color (BGR) for overlay
    overlay = cv2.cvtColor(
        (image / (np.max(image) + 1e-6) * 255).astype(np.uint8),
        cv2.COLOR_GRAY2BGR
    )

    # LV contours (purple)
    contours_lv, _ = cv2.findContours(lv_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours_lv, -1, (204, 102, 153), 1)

    # RV contours (yellowish) if available
    if rv_mask is not None:
        contours_rv, _ = cv2.findContours(rv_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours_rv, -1, (190, 240, 250), 1)

    # Myocardium contours (pink)
    contours_myo, _ = cv2.findContours(myo_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours_myo, -1, (216, 191, 216), 1)

    # Show overlayed image with matplotlib (convert BGR → RGB)
    plt.figure(figsize=(8, 8))
    plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    plt.title("LV (purple), RV (yellow), Myocardium (pink)")
    plt.axis('off')
    plt.tight_layout()
    plt.show()


# ---- SI Curve & Peak Finder (LV) ----
# ---- SI Curve & Peak Finder (LV + optional RV) ----
def plot_si_curve_and_find_peak(folder, start_image, end_image, lv_mask, rv_mask=None):
    """
    Compute and plot signal intensity (SI) curves for LV (and RV if provided)
    across frames, then find the global peak LV intensity frame.

    Parameters
    - folder: Path containing ordered DICOM images.
    - start_image: First image filename for analysis window.
    - end_image: Last image filename for analysis window.
    - lv_mask: Binary LV mask for extracting pixel values.
    - rv_mask: Optional binary RV mask for comparison.

    Returns
    - peak_image: Filename of frame with max LV SI.
    - peak_idx: Index of LV peak within selected frames.
    - cycle_pos: Normalized position (0–1) of LV peak within cycle.
    - image_names: List of processed frame names.
    """

    # Collect all files, ignoring hidden ones, sorted by numeric index in filename
    files = sorted(
        [f for f in os.listdir(folder) if not f.startswith('.')],
        key=lambda x: int(''.join(filter(str.isdigit, x)))
    )

    # Extract subrange between start and end images
    start_idx = files.index(start_image)
    end_idx = files.index(end_image) + 1
    lv_si, rv_si, image_names = [], [], []

    # Loop over selected frames
    for f in files[start_idx:end_idx]:
        _, raw_image = load_dicom_image(os.path.join(folder, f))

        # ---- LV SI ----
        lv_pixels = raw_image[lv_mask == 255]
        if lv_pixels.size == 0:
            lv_val = 0.0
        else:
            lv_thresh = np.percentile(lv_pixels, 90)
            lv_top = lv_pixels[lv_pixels >= lv_thresh]
            lv_val = float(np.mean(lv_top)) if lv_top.size else 0.0
        lv_si.append(lv_val)

        # ---- RV SI (if mask provided) ----
        if rv_mask is not None:
            rv_pixels = raw_image[rv_mask == 255]
            if rv_pixels.size == 0:
                rv_val = 0.0
            else:
                rv_thresh = np.percentile(rv_pixels, 90)
                rv_top = rv_pixels[rv_pixels >= rv_thresh]
                rv_val = float(np.mean(rv_top)) if rv_top.size else 0.0
            rv_si.append(rv_val)

        image_names.append(f)

    # ---- Plot SI curve(s) ----
    plt.figure()
    plt.plot(range(len(lv_si)), lv_si, marker='o', label="LV (Top 10%)")
    if rv_mask is not None:
        plt.plot(range(len(rv_si)), rv_si, marker='s', label="RV (Top 10%)")
    plt.title("Signal Intensity Curve")
    plt.xlabel("Frame Index")
    plt.ylabel("Mean Top 10% Raw Intensity")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # ---- LV Peak detection (global max only) ----
    peak_idx = int(np.argmax(lv_si))
    peak_image = image_names[peak_idx]

    # Position of peak normalized across cardiac cycle (0=start, 1=end)
    cycle_pos = peak_idx / (len(lv_si) - 1) if len(lv_si) > 1 else 0.5

    print(f"LV peak intensity image: {peak_image} (SI={lv_si[peak_idx]:.2f}, Pos={cycle_pos:.2f})")
    return peak_image, peak_idx, cycle_pos, image_names

# ---- Helper: compute SI values for a given mask (for plotting LV + RV together) ----
def compute_si_curve(folder, start_image, end_image, mask):
    """
    Compute signal intensity (SI) curve for a mask region across frames.

    Parameters
    - folder: Path containing ordered DICOM images.
    - start_image: First image filename.
    - end_image: Last image filename.
    - mask: Binary mask of region (LV, RV, or myocardium).

    Returns
    - si_values: List of mean intensities (top 10% pixels) per frame.
    """

    # Collect and sort filenames (ignore hidden files)
    files = sorted(
        [f for f in os.listdir(folder) if not f.startswith('.')],
        key=lambda x: int(''.join(filter(str.isdigit, x)))
    )

    # Extract frame window
    start_idx = files.index(start_image)
    end_idx = files.index(end_image) + 1
    si_values = []

    # Loop through each frame and compute SI
    for f in files[start_idx:end_idx]:
        _, raw_image = load_dicom_image(os.path.join(folder, f))
        masked_pixels = raw_image[mask == 255]

        if masked_pixels.size == 0:
            mean_intensity = 0.0
        else:
            threshold = np.percentile(masked_pixels, 90)
            top_pixels = masked_pixels[masked_pixels >= threshold]
            mean_intensity = float(np.mean(top_pixels)) if top_pixels.size else 0.0

        si_values.append(mean_intensity)

    return si_values

# ---- Create Image Matrix from DICOM Folder ----
def create_image_matrix_from_dicom_folder(dicom_path):
    """
    Load all DICOM files from a folder and organize them into a 4D numpy array.
    Slices are determined by InstanceNumber order (e.g., 1-88 = slice 0, 89-176 = slice 1, etc.)
    
    Parameters
    - dicom_path: Path to folder containing DICOM files.
    
    Returns
    - image_matrix: numpy array of shape (num_dynamics, num_slices, rows, cols)
    - patient_name: Patient name from DICOM metadata
    - load_time: Time taken to load images in seconds
    """
    start_load = time.time()
    
    # Collect all DICOM files (ignore hidden files)
    dicom_files = [f for f in os.listdir(dicom_path) if not f.startswith('.')]
    
    if not dicom_files:
        raise ValueError(f"No DICOM files found in {dicom_path}")
    
    # Read all files and get their InstanceNumber
    file_info = []
    for f in dicom_files:
        path = os.path.join(dicom_path, f)
        try:
            dcm = pydicom.dcmread(path, stop_before_pixels=True)
            instance_num = int(dcm.InstanceNumber) if "InstanceNumber" in dcm else 0
            file_info.append((f, instance_num))
        except:
            continue
    
    # Sort by InstanceNumber
    file_info.sort(key=lambda x: x[1])
    
    total_files = len(file_info)
    num_slices = 4
    num_dynamics = total_files // num_slices
    
    print(f"Total files: {total_files}, Dynamics per slice: {num_dynamics}")
    
    # Read first image to get dimensions and patient name
    first_dcm = pydicom.dcmread(os.path.join(dicom_path, file_info[0][0]))
    rows, cols = first_dcm.pixel_array.shape
    patient_name = str(first_dcm.PatientName) if "PatientName" in first_dcm else "Unknown"
    
    # Initialize 4D array: (dynamics, slices, rows, cols)
    image_matrix = np.zeros((num_dynamics, num_slices, rows, cols), dtype=np.float32)
    
    # Fill the array - files are sorted by InstanceNumber
    # Instance 1-88 = slice 0 (dynamics 0-87)
    # Instance 89-176 = slice 1 (dynamics 0-87)
    # etc.
    for slice_idx in range(num_slices):
        start_idx = slice_idx * num_dynamics
        end_idx = start_idx + num_dynamics
        slice_files = file_info[start_idx:end_idx]
        
        for dyn_idx, (filename, inst_num) in enumerate(slice_files):
            dcm = pydicom.dcmread(os.path.join(dicom_path, filename))
            image_matrix[dyn_idx, slice_idx, :, :] = dcm.pixel_array.astype(np.float32)
        
        print(f"Slice {slice_idx}: InstanceNumbers {slice_files[0][1]} to {slice_files[-1][1]}")
    
    load_time = time.time() - start_load
    
    print(f"Created image matrix with shape: {image_matrix.shape}")
    print(f"  - {num_dynamics} dynamics")
    print(f"  - {num_slices} slices")
    print(f"  - {rows} x {cols} pixels")
    print(f"  - Patient: {patient_name}")
    print(f"Time to load images: {load_time:.2f}s")
    
    return image_matrix, patient_name, load_time


# ---- Test 4D Array Creation ----
def test_4Darray_creation(image_matrix):
    """
    Test the 4D array by displaying the middle dynamic of the third slice.
    
    Parameters
    - image_matrix: 4D numpy array of shape (num_dynamics, num_slices, rows, cols)
    """
    num_dynamics, num_slices, rows, cols = image_matrix.shape
    
    # Get middle dynamic index
    middle_dynamic = num_dynamics // 2
    
    # Third slice (0-indexed, so index 2)
    slice_idx = 2
    
    if slice_idx >= num_slices:
        print(f"Warning: Only {num_slices} slices available, using last slice")
        slice_idx = num_slices - 1
    
    # Extract the image
    test_image = image_matrix[middle_dynamic, slice_idx, :, :]
    
    print(f"\nDisplaying: Dynamic {middle_dynamic}/{num_dynamics-1}, Slice {slice_idx+1}/{num_slices}")
    print(f"Image shape: {test_image.shape}, Min: {test_image.min():.2f}, Max: {test_image.max():.2f}")
    
    # Display the image
    plt.figure(figsize=(8, 8))
    plt.imshow(test_image, cmap='gray')
    plt.title(f"Middle Dynamic ({middle_dynamic}) - Slice 3")
    plt.colorbar(label='Intensity')
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    return test_image

# ---- Smooth Contour Points using Hamming Window ----
def smooth_contour_points(contour_points, window_size=7):
    """
    Smooth contour points using a Hamming window weighted average.
    Handles circular contours by wrapping around at boundaries.
    
    Parameters
    - contour_points: 2xN array (row 0 = x coords, row 1 = y coords)
    - window_size: Size of the Hamming window (default 7, must be odd)
    
    Returns
    - smoothed: 2xN array of smoothed contour points
    """
    if contour_points is None or contour_points.shape[1] < window_size:
        return contour_points
    
    # Ensure window_size is odd
    if window_size % 2 == 0:
        window_size += 1
    
    # Create Hamming window weights
    hamming_weights = np.hamming(window_size)
    hamming_weights = hamming_weights / np.sum(hamming_weights)  # Normalize
    
    n = contour_points.shape[1]
    half_window = window_size // 2
    smoothed = np.zeros_like(contour_points, dtype=np.float32)
    
    # Apply Hamming-weighted average with circular wrapping
    for i in range(n):
        # Gather indices with circular wrapping
        indices = [(i + j) % n for j in range(-half_window, half_window + 1)]
        # Weighted average using Hamming window
        smoothed[0, i] = np.sum(contour_points[0, indices] * hamming_weights)
        smoothed[1, i] = np.sum(contour_points[1, indices] * hamming_weights)
    
    return smoothed

# ---- Spline Interpolate and Resample Contour ----
def resample_contour_spline(contour_points, num_points=60, downsample_to=10):
    """
    Downsample contour to fewer points, then spline interpolate to get smooth equally-spaced points.
    
    Parameters
    - contour_points: Nx2 array of contour points
    - num_points: Number of output points (default 60)
    - downsample_to: Number of points to downsample to before spline (default 15)
    
    Returns
    - resampled: 2x60 array (row 0 = x coords, row 1 = y coords)
    """
    if contour_points is None or len(contour_points) < 4:
        return np.zeros((2, num_points), dtype=np.float32)
    
    # Ensure contour is 2D array
    pts = np.array(contour_points).reshape(-1, 2)
    
    # Downsample by systematically skipping vertices
    n_orig = len(pts)
    if n_orig > downsample_to:
        indices = np.linspace(0, n_orig - 1, downsample_to, dtype=int)
        pts = pts[indices]
    
    # Close the contour if not already closed
    if not np.allclose(pts[0], pts[-1]):
        pts = np.vstack([pts, pts[0]])
    
    # Spline interpolation with smoothing
    try:
        tck, u = splprep([pts[:, 0], pts[:, 1]], s=0, per=True)
        u_new = np.linspace(0, 1, num_points, endpoint=False)
        x_new, y_new = splev(u_new, tck)
        resampled = np.array([x_new, y_new], dtype=np.float32)
    except:
        # Fallback: simple linear interpolation if spline fails
        resampled = np.zeros((2, num_points), dtype=np.float32)
    
    return resampled


# ---- Zoom Contour by Pixel Offset ----
def contour_zoom(contour, offset_pixels):
    """
    Expand or contract a contour by a pixel offset using centroid as reference.
    
    Parameters
    - contour: 2xN array (row 0 = x coords, row 1 = y coords)
    - offset_pixels: Positive to expand outward, negative to contract inward
    
    Returns
    - zoomed_contour: 2xN array with adjusted coordinates
    """
    if contour is None or not np.any(contour):
        return contour
    
    # Compute centroid
    cx = np.mean(contour[0, :])
    cy = np.mean(contour[1, :])
    
    # For each point, compute direction from centroid and offset
    dx = contour[0, :] - cx
    dy = contour[1, :] - cy
    
    # Distance from centroid for each point
    dist = np.sqrt(dx**2 + dy**2)
    dist[dist == 0] = 1  # Avoid division by zero
    
    # Unit vectors from centroid to each point
    ux = dx / dist
    uy = dy / dist
    
    # Apply offset
    new_x = contour[0, :] + ux * offset_pixels
    new_y = contour[1, :] + uy * offset_pixels
    
    return np.array([new_x, new_y], dtype=np.float32)

# def smooth_mask_distance(mask):
#     """
#     Remove contour dips using distance transform smoothing.
#     Input: binary mask (0/255)
#     Output: smoothed binary mask (0/255)
#     """
#     # Ensure binary 0/1
#     mask_bin = (mask > 0).astype(np.uint8)

#     # Small morphological closing (removes tiny dents)
#     kernel = np.ones((3,3), np.uint8)
#     mask_closed = cv2.morphologyEx(mask_bin, cv2.MORPH_CLOSE, kernel)

#     # Distance transform
#     dist = cv2.distanceTransform(mask_closed, cv2.DIST_L2, 5)

#     # Normalize
#     if dist.max() > 0:
#         dist = dist / dist.max()

#     # Smooth
#     dist_blur = cv2.GaussianBlur(dist, (5,5), 0)

#     # Re-threshold
#     smooth = (dist_blur > 0.3).astype(np.uint8)

#     return (smooth * 255).astype(np.uint8)


def smooth_contour_remove_dips(contour_2xN, dist_thresh=0.85, angle_thresh_deg=70):
    if contour_2xN is None or contour_2xN.shape[1] < 10:
        return contour_2xN

    pts = contour_2xN.T
    N   = len(pts)

    cx, cy      = np.mean(pts, axis=0)
    dists       = np.sqrt((pts[:, 0] - cx)**2 + (pts[:, 1] - cy)**2)
    median_dist = np.median(dists)
    dists_norm  = dists / (median_dist + 1e-6)

    angles = []
    for i in range(N):
        p_prev = pts[(i - 1) % N]
        p_curr = pts[i]
        p_next = pts[(i + 1) % N]
        v1 = p_curr - p_prev
        v2 = p_next - p_curr
        v1 = v1 / (np.linalg.norm(v1) + 1e-6)
        v2 = v2 / (np.linalg.norm(v2) + 1e-6)
        dot = np.clip(np.dot(v1, v2), -1.0, 1.0)
        angles.append(np.degrees(np.arccos(dot)))

    angles = np.array(angles)

    # ---- PRINT ALL POINTS THAT ARE CLOSE TO CENTROID ----
    print(f"\n  dist_norm min={dists_norm.min():.3f} max={dists_norm.max():.3f} median=1.0")
    print(f"  angle     min={angles.min():.1f}   max={angles.max():.1f}")
    print(f"  Points below dist_thresh {dist_thresh}: {(dists_norm < dist_thresh).sum()}")
    print(f"  Points above angle_thresh {angle_thresh_deg}: {(angles > angle_thresh_deg).sum()}")
    print(f"  Points above outer thresh {1.0/dist_thresh:.3f}: {(dists_norm > 1.0/dist_thresh).sum()}")

    valid_pts = []
    for i in range(N):
        if dists_norm[i] < dist_thresh:
            continue
        if dists_norm[i] > 1.0 / dist_thresh:
            continue
        if angles[i] > angle_thresh_deg:
            continue
        valid_pts.append(pts[i])

    print(f"  → {len(valid_pts)}/{N} points kept")

    if len(valid_pts) < 10:
        return contour_2xN

    valid_pts = np.array(valid_pts)

    try:
        if not np.allclose(valid_pts[0], valid_pts[-1]):
            valid_pts = np.vstack([valid_pts, valid_pts[0]])
        tck, _ = splprep([valid_pts[:, 0], valid_pts[:, 1]], s=0, per=True)
        u_new  = np.linspace(0, 1, contour_2xN.shape[1], endpoint=False)
        x_new, y_new = splev(u_new, tck)
        return np.array([x_new, y_new], dtype=np.float32)
    except:
        return contour_2xN

# ---- Get Auto Myocardium (Segmentation Pipeline) ----
def get_auto_myocardium(image_matrix, patient_name="Unknown"):
    """
    Perform full segmentation pipeline on a 4D image matrix.
    Expects image_matrix with slices already filtered (only myocardium slices, no slice1).
    Processes all slices in the input as myocardium slices 1, 2, 3.
    
    Parameters
    - image_matrix: 4D numpy array of shape (num_dynamics, num_slices, rows, cols)
    
    Returns
    - contours_dict: Dictionary with keys like 'slice1_endo', 'slice1_epi', etc.
                     Each value is a list of [x, y] coordinate pairs (60 points per contour)
    """
    start_segmentation = time.time()

    # Reshape to slices x dynamics x rows x cols and skip slice1 (index 0)
    # Original: (dynamics, slices, rows, cols) -> transpose to (slices, dynamics, rows, cols)
    image_matrix = np.transpose(image_matrix, (1, 0, 2, 3))
    # Skip slice1 (index 0), keep only slices 1, 2, 3
    image_matrix = image_matrix[1:, :, :, :]
    
    print(f"Reshaped matrix (skipping slice1): {image_matrix.shape}")
    
    num_slices, num_dynamics, rows, cols = image_matrix.shape
    
    print(f"\nProcessing 4D array: {num_slices} slices x {num_dynamics} dynamics x {rows}x{cols} pixels\n")

    # Store contour results as Dictionary for C# consumption
    contours_dict = {}

    # Store data for final 3x5 subplot display
    slice_contour_data = []

    # ============================================================
    # LOOP THROUGH ALL SLICES (input already has slice1 removed)
    # ============================================================
    global_peak_dyn_idx = None
    global_endo_seed = None
    
    for slice_idx in range(num_slices):  # Process all slices in input
        myo_slice_num = slice_idx + 1  # Myocardium slice numbering (1, 2, 3)

        print("\n=================================================")
        print(f"Processing Myocardium Slice {myo_slice_num} (array index {slice_idx})")
        print("=================================================")

        slice_result = {'slice_idx': slice_idx, 'myo_slice_num': myo_slice_num}

        # Skip first 10 dynamics if possible
        start_dyn = 10 if num_dynamics > 10 else 0
        available_dynamics = num_dynamics - start_dyn

        if available_dynamics == 0:
            continue
        # --------------------------------------------------------
        # STEP 0.5 — Seed frame auto selection (30% into available dynamics)
        # --------------------------------------------------------
        seed_dyn_offset = int(0.3 * available_dynamics)
        seed_dyn_idx = start_dyn + min(seed_dyn_offset, available_dynamics - 1)

        print(f"Automatically selected seed dynamic: {seed_dyn_idx} (after skipping first 10)")
        slice_result['seed_dyn_idx'] = seed_dyn_idx

        # ---- Step 1: Load seed image from 4D array ----
        t0 = time.time()
        seed_raw = image_matrix[slice_idx, seed_dyn_idx, :, :]
        seed_eq, seed_orig = normalize_image_from_array(seed_raw)
        print(f"[Step 1] Load seed image: {time.time() - t0:.2f}s")

        # ---- Step 2: Detect seed points ----
        t0 = time.time()
        if global_endo_seed is None:
            endo_seed, endo_ellipses = auto_detect_seed_point(seed_eq, debug=False, return_contours=True)
            global_endo_seed = endo_seed
        else:
            endo_seed = global_endo_seed
            print(f"Reusing Endo seed point: {endo_seed}")
               
        print(f"[Step 2] Seed point detection: {time.time() - t0:.2f}s")

        slice_result['endo_seed'] = endo_seed
        
        # ---- Step 3: Initial segmentation ----
        t0 = time.time()

        endo_mask = region_grow(seed_eq, endo_seed, threshold=0.07)
        kernel = np.ones((5,5), np.uint8)
        endo_mask = cv2.morphologyEx(endo_mask, cv2.MORPH_OPEN, kernel)
        endo_mask = cv2.morphologyEx(endo_mask, cv2.MORPH_CLOSE, kernel)
        print(f"[Step 3] Initial segmentation: {time.time() - t0:.2f}s")

        # ---- Step 4: Generate SI curve from 4D array ----
        SKIP_DYNAMICS = 1  # Skip first dynamic from the curve as this is the PD image
        t0 = time.time()
        if global_peak_dyn_idx is None:
            # Compute SI curve without plotting
            num_dyn = image_matrix.shape[1]  # dynamics is second dimension
            endo_si = []
            for dyn_idx in range(SKIP_DYNAMICS, num_dyn):
                raw_image = image_matrix[slice_idx, dyn_idx, :, :]
                endo_pixels = raw_image[endo_mask == 255]
                if endo_pixels.size == 0:
                    endo_val = 0.0
                else:
                    endo_thresh = np.percentile(endo_pixels, 90)
                    endo_top = endo_pixels[endo_pixels >= endo_thresh]
                    endo_val = float(np.mean(endo_top)) if endo_top.size else 0.0
                endo_si.append(endo_val)
            
            peak_dyn_idx = int(np.argmax(endo_si))
            global_peak_dyn_idx = peak_dyn_idx
            print(f"Global peak dynamic detected: {global_peak_dyn_idx}")

        else:
            peak_dyn_idx = global_peak_dyn_idx
            print(f"Using previously detected peak dynamic: {peak_dyn_idx}")

        slice_result['peak_dyn_idx'] = peak_dyn_idx
        print(f"[Step 4] SI curve + peak detection: {time.time() - t0:.2f}s")

        # ---- Step 5: Segmentation on peak frame ----
        t0 = time.time()
        segmenting_image = image_matrix[slice_idx, peak_dyn_idx, :, :];
        #segmenting_image = image_matrix[slice_idx, peak_dyn_idx, :, :] + 0.25*image_matrix[slice_idx, 0, :, :]
        peak_eq, peak_orig = normalize_image_from_array(segmenting_image)
        peak_endo_seed, peak_endo_ellipses = auto_detect_seed_point(peak_eq, debug=False, return_contours=True)
        #peak_rv_seed = auto_detect_rv_seed_near_lv(peak_endo_seed, peak_endo_ellipses, debug=False)

        # min_dist = int(0.05 * max(peak_eq.shape))
        # if np.hypot(peak_rv_seed[0]-peak_endo_seed[0], peak_rv_seed[1]-peak_endo_seed[1]) < min_dist:
        #     peak_rv_seed = (peak_endo_seed[0] - 30, peak_endo_seed[1])
        #     print("RV seed too close to Endo on peak frame; using fallback:", peak_rv_seed)

        endo_mask_peak = region_grow(peak_eq, peak_endo_seed, threshold=0.07)
        kernel = np.ones((5,5), np.uint8)
        endo_mask_peak = cv2.morphologyEx(endo_mask_peak, cv2.MORPH_OPEN, kernel)
        endo_mask_peak = cv2.morphologyEx(endo_mask_peak, cv2.MORPH_CLOSE, kernel)
        #rv_mask_peak = region_grow(peak_eq, peak_rv_seed, threshold=0.07)
        epi_mask_peak = segment_myocardium_edge_based(peak_eq, endo_mask_peak)
        epi_mask_peak = cv2.morphologyEx(epi_mask_peak, cv2.MORPH_OPEN, kernel)
        epi_mask_peak = cv2.morphologyEx(epi_mask_peak, cv2.MORPH_CLOSE, kernel)
        # draw_contours(peak_orig, endo_mask_peak, epi_mask_peak, rv_mask=rv_mask_peak)

        # ---- Extract final contours from peak frame ----
        contours_endo_peak, _ = cv2.findContours(endo_mask_peak, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours_epi_peak, _ = cv2.findContours(epi_mask_peak, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        endo_points_peak = None
        epi_points_peak = None
        
        if contours_endo_peak:
            endo_contour_peak = max(contours_endo_peak, key=cv2.contourArea)
            endo_points_peak = endo_contour_peak.squeeze()
        
        if contours_epi_peak:
            epi_contour_peak = max(contours_epi_peak, key=cv2.contourArea)
            epi_points_peak = epi_contour_peak.squeeze()
        
        # For debugging
        
        # # ---- Resample contours to 2x60 arrays ----
        endo_points_peak = resample_contour_spline(endo_points_peak, num_points=30)
        epi_points_peak = resample_contour_spline(epi_points_peak, num_points=30)

        # ---- Apply zoom: expand endo by 2 pixels, contract epi by 2 pixels ----
        endo_points_peak = contour_zoom(endo_points_peak, offset_pixels=2)
        epi_points_peak = contour_zoom(epi_points_peak, offset_pixels=-1)

        #Smoothen the contours using Hamming window
        endo_points_peak = smooth_contour_points(endo_points_peak, window_size=9)
        epi_points_peak = smooth_contour_points(epi_points_peak, window_size=9)

        # #pick every 6th point
        # endo_points_peak = endo_points_peak[:, ::6]
        # epi_points_peak = epi_points_peak[:, ::6]

        slice_result['endo_contour'] = endo_points_peak
        slice_result['epi_contour'] = epi_points_peak

        # Store contours in Dictionary format for C# consumption
        # Convert 2xN array to list of [x, y] pairs
        endo_key = f"slice{myo_slice_num}_endo"
        epi_key = f"slice{myo_slice_num}_epi"
        
        # Convert numpy array to list of [x, y] coordinate pairs
        endo_points_list = [[float(endo_points_peak[0, i]), float(endo_points_peak[1, i])] 
                           for i in range(endo_points_peak.shape[1])]
        epi_points_list = [[float(epi_points_peak[0, i]), float(epi_points_peak[1, i])] 
                          for i in range(epi_points_peak.shape[1])]
        
        contours_dict[endo_key] = endo_points_list
        contours_dict[epi_key] = epi_points_list
        contours_dict[f"slice{myo_slice_num}_peak_dyn"] = peak_dyn_idx

        print(f"[Step 5] Peak frame segmentation: {time.time() - t0:.2f}s")
        print(f"Endo contour resampled shape: {endo_points_peak.shape}")
        print(f"Epi contour resampled shape: {epi_points_peak.shape}")

        # ---- Step 6: Compute SI curves (no plotting) ----
        t0 = time.time()
        endo_si_final = compute_si_curve_from_array(image_matrix, slice_idx, endo_mask_peak)
        #rv_si_final = compute_si_curve_from_array(image_matrix, slice_idx, rv_mask_peak)

        slice_result['endo_si_curve'] = endo_si_final
        contours_dict[f"slice{myo_slice_num}_si_curve"] = endo_si_final
        # slice_result['rv_si_curve'] = rv_si_final

        print(f"[Step 6] Final SI curves: {time.time() - t0:.2f}s")

        # Store for final 3x5 plot
        slice_contour_data.append({
            'endo': endo_points_peak,
            'epi': epi_points_peak,
            'myo_slice_num': myo_slice_num,
            'slice_idx': slice_idx,
            'peak_dyn_idx': peak_dyn_idx,
            'endo_si_curve': endo_si_final
        })

    # ---- Total runtime (calculate before plotting) ----
    segmentation_time = time.time() - start_segmentation
    contours_dict['segmentation_time'] = segmentation_time
    print(f"\nTime for segmentation: {segmentation_time:.2f}s")

    return contours_dict

def prepare_plotting_data(image_matrix, contours_dict):
    """
    Prepare slice_contour_data for plotting from contours_dict.
    
    Parameters
    - image_matrix: 4D numpy array (num_dynamics, num_slices, rows, cols) - original format
    - contours_dict: Dictionary with contour results from get_auto_myocardium
    
    Returns
    - processed_matrix: 4D array transposed and sliced (num_slices, num_dynamics, rows, cols)
    - slice_contour_data: List of dictionaries for plotting
    """
    # Transpose and skip slice1 (same as in get_auto_myocardium)
    processed_matrix = np.transpose(image_matrix, (1, 0, 2, 3))
    processed_matrix = processed_matrix[1:, :, :, :]
    
    num_slices = processed_matrix.shape[0]
    slice_contour_data = []
    
    for slice_idx in range(num_slices):
        myo_slice_num = slice_idx + 1
        endo_key = f"slice{myo_slice_num}_endo"
        epi_key = f"slice{myo_slice_num}_epi"
        si_key = f"slice{myo_slice_num}_si_curve"
        
        # Convert list of [x, y] pairs back to 2xN array
        if endo_key in contours_dict:
            endo_list = contours_dict[endo_key]
            endo_contour = np.array([[p[0] for p in endo_list], [p[1] for p in endo_list]], dtype=np.float32)
        else:
            endo_contour = np.zeros((2, 60), dtype=np.float32)
        
        if epi_key in contours_dict:
            epi_list = contours_dict[epi_key]
            epi_contour = np.array([[p[0] for p in epi_list], [p[1] for p in epi_list]], dtype=np.float32)
        else:
            epi_contour = np.zeros((2, 60), dtype=np.float32)
        
        # Get SI curve from contours_dict if available
        si_curve = contours_dict.get(si_key, [])
        
        # Get peak dynamic from contours_dict (use middle of dynamics as fallback)
        peak_key = f"slice{myo_slice_num}_peak_dyn"
        peak_dyn_idx = contours_dict.get(peak_key, processed_matrix.shape[1] // 2)
        
        slice_contour_data.append({
            'endo': endo_contour,
            'epi': epi_contour,
            'myo_slice_num': myo_slice_num,
            'slice_idx': slice_idx,
            'peak_dyn_idx': peak_dyn_idx,
            'endo_si_curve': si_curve
        })
    
    return processed_matrix, slice_contour_data


# ---- Plot Slice 2 Segmentation Image with Contours ----
def plot_slice2_segmentation(image_matrix, slice_contour_data, metadata=None):
    """
    Plot the segmentation image (peak frame) for slice 2 with overlayed contours.
    
    Parameters
    - image_matrix: 4D numpy array (num_slices, num_dynamics, rows, cols)
    - slice_contour_data: List of dictionaries containing contour data for each slice
    - metadata: Dictionary containing patient_name, patient_id, series_number, study_type, study_name
    """
    
    import matplotlib.pyplot as plt
    
    # Extract metadata with defaults
    if metadata is None:
        metadata = {}
    study_type = metadata.get('study_type', 'Unknown')
    study_name = metadata.get('study_name', 'Unknown')
    
    if len(slice_contour_data) < 2:
        print("Not enough slice data for plotting slice 2")
        return
    
    # Get slice 2 data (index 1 in the list)
    slice_data = slice_contour_data[1]
    myo_num = slice_data['myo_slice_num']
    s_idx = slice_data['slice_idx']
    peak_dyn = slice_data['peak_dyn_idx']
    endo = slice_data['endo']
    epi = slice_data['epi']
    
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    
    # Use peak frame (segmentation image)
    segmenting_image = image_matrix[s_idx, peak_dyn, :, :]
    ax.imshow(segmenting_image, cmap='gray')
    
    # Plot Endo contour (coral orange)
    if np.any(endo):
        endo_closed = np.hstack([endo, endo[:, :1]])
        ax.plot(endo_closed[0, :], endo_closed[1, :], color='#FF7F50', linewidth=2)
    
    # Plot Epi contour (light blue)
    if np.any(epi):
        epi_closed = np.hstack([epi, epi[:, :1]])
        ax.plot(epi_closed[0, :], epi_closed[1, :], color='#87CEEB', linewidth=2)
    
    ax.set_title(f"Slice {myo_num} (Peak Dyn {peak_dyn})", fontsize=12)
    ax.axis('off')
    
    title_text = f"{study_name} {study_type} - Slice 2 Segmentation"
    fig.suptitle(title_text, fontsize=14, color='blue')
    plt.tight_layout(rect=[0, 0, 1, 0.92])


# ---- Plot Segmentation Results ----
def plot_segmentation_results(image_matrix, slice_contour_data, metadata=None, save_to_file=True, show_figure=True, output_dir=None):
    """
    Plot segmentation results as 2x3 subplots: top row = images with contours, bottom row = SI curves.
    
    Parameters
    - image_matrix: 4D numpy array (num_slices, num_dynamics, rows, cols)
    - slice_contour_data: List of dictionaries containing contour data for each slice
    - metadata: Dictionary containing patient_name, patient_id, series_number, study_type, study_name
    - save_to_file: If True, save figure to file (default: True)
    - show_figure: If True, display figure interactively (default: True)
    - output_dir: Directory to save output file (default: current directory)
    
    Returns
    - output_filename: Path to saved file (if save_to_file=True), else None
    """
    import matplotlib.pyplot as plt

    # Extract metadata with defaults
    if metadata is None:
        metadata = {}
    patient_id = metadata.get('patient_id', 'Unknown')
    series_number = metadata.get('series_number', 'Unknown')
    study_type = metadata.get('study_type', 'Unknown')
    study_name = metadata.get('study_name', 'Unknown')
    
    if len(slice_contour_data) < 3:
        print("Not enough slice data for plotting (need at least 3 slices)")
        return None
    
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
    
    for col_idx in range(3):
        slice_data = slice_contour_data[col_idx]
        endo = slice_data['endo']
        epi = slice_data['epi']
        myo_num = slice_data['myo_slice_num']
        s_idx = slice_data['slice_idx']
        peak_dyn = slice_data['peak_dyn_idx']
        
        # Image with contours
        ax_img = axes[col_idx]
        segmenting_image = image_matrix[s_idx, peak_dyn, :, :]
        ax_img.imshow(segmenting_image, cmap='gray', aspect='equal')
        
        # Plot Endo contour (coral orange)
        if np.any(endo):
            endo_closed = np.hstack([endo, endo[:, :1]])
            ax_img.plot(endo_closed[0, :], endo_closed[1, :], color='#FF7F50', linewidth=1.5)
        
        # Plot Epi contour (light blue)
        if np.any(epi):
            epi_closed = np.hstack([epi, epi[:, :1]])
            ax_img.plot(epi_closed[0, :], epi_closed[1, :], color='#87CEEB', linewidth=1.5)
        
        ax_img.set_title(f"Slice {myo_num}", fontsize=11)
        ax_img.axis('off')
    
    # Main title: study name + study type, series number, patient ID
    title_text = f"{study_name} {study_type}, Ser. No. {series_number}, Pat ID: {patient_id}"
    fig.suptitle(title_text, fontsize=14, color='green')
    fig.subplots_adjust(left=0.02, right=0.98, top=0.88, bottom=0.02, wspace=0.05)
    
    output_filename = None
    if save_to_file:
        # Save figure to file
        filename = f"{study_name}_{study_type}_Ser{series_number}.png"
        if output_dir:
            output_filename = os.path.join(output_dir, filename)
        else:
            output_filename = filename
        fig.savefig(output_filename, dpi=150, bbox_inches='tight')
        print(f"Figure saved to: {output_filename}")
    
    if show_figure:
        # Display interactively
        plt.show()
    # Don't close fig - let caller handle with plt.show(block=True)
    
    return output_filename


# ---- Main Logic ----
def main():

    folder = r"C:\Users\320289802\OneDrive - Philips\Documents\Cardiac Perfusion\Data\PESA 1\DICOM\S83540\S21010"
    # Create 4D array from DICOM folder (returns image_matrix, patient_name, load_time)
    image_matrix, patient_name, load_time = create_image_matrix_from_dicom_folder(folder)

    # Process the 4D array
    contours_dict = get_auto_myocardium(image_matrix, patient_name)
    processed_matrix, slice_contour_data = prepare_plotting_data(image_matrix, contours_dict)

    plot_segmentation_results(processed_matrix, slice_contour_data, show_figure=True)

    print(f"\n=== Processing Summary ===")
    print(f"Contour keys: {[k for k in contours_dict.keys() if k.startswith('slice')]}")
    print(f"Time to load images: {load_time:.2f}s")
    print(f"Time for segmentation: {contours_dict['segmentation_time']:.2f}s")
    print(f"Total time: {load_time + contours_dict['segmentation_time']:.2f}s")

if __name__ == "__main__":
    main()
