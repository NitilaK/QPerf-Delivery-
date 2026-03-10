import os
import cv2
import numpy as np
import pydicom
import matplotlib.pyplot as plt
import time 

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
    contours_lv, _ = cv2.findContours(lv_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    simplified_lv = []
    for cnt in contours_lv:
        epsilon = 0.01 * cv2.arcLength(cnt, True)   # tuning parameter
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        simplified_lv.append(approx)
    cv2.drawContours(overlay, simplified_lv, -1, (204, 102, 153), 1)

    # RV contours (yellowish) if available
    if rv_mask is not None:
        contours_rv, _ = cv2.findContours(rv_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours_rv, -1, (190, 240, 250), 1)

    # Myocardium contours (pink)
    contours_myo, _ = cv2.findContours(myo_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    simplified_myo = []
    for cnt in contours_myo:
        epsilon = 0.01 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        simplified_myo.append(approx)
    cv2.drawContours(overlay, simplified_myo, -1, (216, 191, 216), 1)

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

def scale_contour(contour, scale):
    """
    Scale a contour about its centroid.
    scale > 1.0 expands
    scale < 1.0 contracts
    """
    C = contour.astype(np.float32)

    centroid = np.mean(C, axis=0)
    C_scaled = (C - centroid) * scale + centroid

    return C_scaled.astype(np.int32)

def scale_mask_via_contour(mask, scale):
    """
    Scale the largest contour in a binary mask
    and return a new mask.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if not contours:
        return mask

    # Use largest contour (LV cavity assumption)
    cnt = max(contours, key=cv2.contourArea)

    scaled_cnt = scale_contour(cnt.squeeze(), scale)

    new_mask = np.zeros_like(mask)
    cv2.drawContours(new_mask, [scaled_cnt], -1, 255, -1)

    return new_mask

# ---- Main Logic ----
def main():

    start_total = time.time()   # Track overall runtime
    folder = r"C:\Users\320289802\OneDrive - Philips\Documents\Cardiac Perfusion\Data\QPERF3\DICOM\SRIS00\S29010"

    from_image = "I1670"
    to_image   = "I2200"

    # ---- Step 0: Determine seed image automatically ----
    files = sorted(
        [f for f in os.listdir(folder) if not f.startswith('.')],
        key=lambda x: int(''.join(filter(str.isdigit, x)))
    )

    try:
        start_idx = files.index(from_image)
        end_idx   = files.index(to_image)
    except ValueError:
        print("❌ from_image or to_image not found in folder.")
        return

    # Subset between from/to
    sub_files = files[start_idx:end_idx + 1]

    # Skip first 10 frames (if available)
    if len(sub_files) > 10:
        sub_files = sub_files[10:]

    # Select the seed frame at the 25th percentile
    seed_idx = int(0.25 * len(sub_files))
    seed_image = sub_files[min(seed_idx, len(sub_files) - 1)]

    print(f"Automatically selected seed image (after skipping first 10): {seed_image}")

    # ---- Step 1: Load seed image ----
    t0 = time.time()
    seed_eq, seed_orig = load_dicom_image(os.path.join(folder, seed_image))
    print(f"[Step 1] Load seed image: {time.time() - t0:.2f}s")

    # ---- Step 2: Detect seed points ----
    t0 = time.time()
    lv_seed, lv_ellipses = auto_detect_seed_point(seed_eq, debug=True, return_contours=True)
    rv_seed = auto_detect_rv_seed_near_lv(lv_seed, lv_ellipses, debug=True)
    print(f"[Step 2] Seed point detection: {time.time() - t0:.2f}s")

    rv_offset = (rv_seed[0] - lv_seed[0], rv_seed[1] - lv_seed[1])

    # ---- Step 3: Initial segmentation ----
    t0 = time.time()

    lv_mask = region_grow(seed_eq, lv_seed, threshold=0.07)
    rv_mask = region_grow(seed_eq, rv_seed, threshold=0.07)

    # ---- Smooth ONLY LV ----
    kernel_lv_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19,19))
    lv_mask = cv2.morphologyEx(lv_mask, cv2.MORPH_CLOSE, kernel_lv_large)
    lv_mask = scale_mask_via_contour(lv_mask, scale=1.15)

    # ---- Generate myocardium from smoothed LV ----
    myo_mask = segment_myocardium_edge_based(seed_eq, lv_mask)
    myo_mask = scale_mask_via_contour(myo_mask, scale=0.92)

    # ---- OPTIONAL: smooth myocardium ----
    kernel_myo_mid = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9))
    myo_mask = cv2.morphologyEx(myo_mask, cv2.MORPH_CLOSE, kernel_myo_mid)

    draw_contours(seed_orig, lv_mask, myo_mask)  # , rv_mask=rv_mask)

    print(f"[Step 3] Initial segmentation: {time.time() - t0:.2f}s")

    # ---- Step 4: Generate SI curve ----
    t0 = time.time()
    peak_image, peak_idx, cycle_pos, all_images = plot_si_curve_and_find_peak(
        folder, from_image, to_image, lv_mask, rv_mask
    )
    print(f"[Step 4] SI curve + peak detection: {time.time() - t0:.2f}s")

    # ---- Step 5: Segmentation on peak frame ----
    t0 = time.time()

    peak_eq, peak_orig = load_dicom_image(os.path.join(folder, peak_image))

    peak_lv_seed, peak_lv_ellipses = auto_detect_seed_point(
        peak_eq, debug=True, return_contours=True
    )

    peak_rv_seed = auto_detect_rv_seed_near_lv(
        peak_lv_seed, peak_lv_ellipses, debug=True
    )

    min_dist = int(0.05 * max(peak_eq.shape))

    if np.hypot(
        peak_rv_seed[0]-peak_lv_seed[0],
        peak_rv_seed[1]-peak_lv_seed[1]
    ) < min_dist:
        peak_rv_seed = (peak_lv_seed[0] - 30, peak_lv_seed[1])
        print("RV seed too close to LV on peak frame; using fallback:", peak_rv_seed)

    # ---- Region growing ----
    lv_mask_peak = region_grow(peak_eq, peak_lv_seed, threshold=0.07)
    rv_mask_peak = region_grow(peak_eq, peak_rv_seed, threshold=0.07)

    # ---- Smooth ONLY LV ----
    kernel_lv_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
    lv_mask_peak = cv2.morphologyEx(lv_mask_peak, cv2.MORPH_CLOSE, kernel_lv_small)
    lv_mask_peak = scale_mask_via_contour(lv_mask_peak, scale=1.15)

    # ---- Myocardium from smoothed LV ----
    myo_mask_peak = segment_myocardium_edge_based(peak_eq, lv_mask_peak)
    myo_mask_peak = scale_mask_via_contour(myo_mask_peak, scale=0.92)

    # ---- OPTIONAL: smooth myocardium ----
    kernel_myo_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    myo_mask_peak = cv2.morphologyEx(myo_mask_peak, cv2.MORPH_CLOSE, kernel_myo_small)

    draw_contours(peak_orig, lv_mask_peak, myo_mask_peak)  # , rv_mask=rv_mask_peak)

    print(f"[Step 5] Peak frame segmentation: {time.time() - t0:.2f}s")

    # ---- Step 6: Final SI curves ----
    t0 = time.time()

    lv_si_final = compute_si_curve(folder, from_image, to_image, lv_mask_peak)
    rv_si_final = compute_si_curve(folder, from_image, to_image, rv_mask_peak)

    plt.figure()
    plt.plot(lv_si_final, marker='o', label="LV (Top 10%)")
    plt.plot(rv_si_final, marker='s', label="RV (Top 10%)")
    plt.title("Signal Intensity Curves (Final, peak-based masks)")
    plt.xlabel("Frame Index")
    plt.ylabel("Mean Top 10% Raw Intensity")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print(f"[Step 6] Final SI curves: {time.time() - t0:.2f}s")

    # ---- Total runtime ----
    print(f"\n[Total Processing Time] {time.time() - start_total:.2f}s")


if __name__ == "__main__":
    main()
