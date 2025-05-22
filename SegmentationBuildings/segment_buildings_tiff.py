"""
Zero-shot building segmentation using Meta's Segment Anything Model (SAM) with TIFF support.
Fixed version that works with SAM's actual API.

Usage:
    python segment_buildings_tiff_fixed.py \
        --input_image path/to/image.tiff \
        --output_label path/to/labels.tiff \
        --output_overlay path/to/overlay.tiff \
        [--model_type vit_b|vit_h] \
        [--checkpoint path/to/sam_vit_b.pth] \
        [--score_thresh 0.85] \
        [--min_area 500] \
        [--max_area 50000] \
        [--min_aspect 0.5] \
        [--max_aspect 2.0]
"""
import argparse
import cv2
import numpy as np
import rasterio
from rasterio.plot import reshape_as_image
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import warnings

# Suppress the georeferencing warning
warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

# Class definitions
CLASS_NAMES = ["background", "building"]
CLASS_COLORS = {
    0: (0, 0, 0),      # background: black
    1: (0, 255, 0)     # building: green
}

def segment_image(
    input_image: str,
    output_label: str,
    output_overlay: str,
    checkpoint: str,
    model_type: str,
    score_thresh: float,
    min_area: int,
    max_area: int,
    min_aspect: float,
    max_aspect: float,
):
    print(f"Loading image: {input_image}")
    
    # 1) Load TIFF image using rasterio
    with rasterio.open(input_image) as dataset:
        # Get the metadata for writing output later
        meta = dataset.meta.copy()
        # Read all bands and convert to numpy array in HWC format (height, width, channels)
        img_array = dataset.read()
        img = reshape_as_image(img_array)
        
        # Handle different band combinations
        if img.shape[2] > 3:
            # If more than 3 bands, use first 3 for visualization
            img = img[:, :, :3]
        elif img.shape[2] == 1:
            # If single band, convert to RGB
            img = np.repeat(img, 3, axis=2)
    
    print(f"Image shape: {img.shape}, dtype: {img.dtype}")
    
    # Make sure image is 8-bit for SAM
    if img.dtype != np.uint8:
        # Normalize and convert to 8-bit
        img_normalized = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)
        img = img_normalized
        
    # Create RGB version for SAM
    img_rgb = img[:, :, :3].copy()
    
    print("Loading SAM model...")
    
    # 2) Load SAM with automatic mask generator
    sam = sam_model_registry[model_type](checkpoint=checkpoint)
    
    # Use SamAutomaticMaskGenerator instead of SamPredictor
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,
        pred_iou_thresh=0.9,
        stability_score_thresh=0.95,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=min_area,
    )
    
    print("Generating masks...")
    
    # 3) Generate masks
    masks = mask_generator.generate(img_rgb)
    
    print(f"Generated {len(masks)} masks")
    
    # 4) Build label map
    H, W = img.shape[:2]
    label_map = np.zeros((H, W), dtype=np.uint8)
    
    # Filter and apply masks
    building_count = 0
    for mask_data in masks:
        mask = mask_data['segmentation']
        score = mask_data['predicted_iou']  # Use IoU as score
        bbox = mask_data['bbox']  # [x, y, w, h]
        
        if score < score_thresh:
            continue
            
        # Calculate area and aspect ratio
        area = mask.sum()
        x, y, w, h = bbox
        aspect = w / h if h > 0 else 0
        
        # Apply filters
        if not (min_area <= area <= max_area and min_aspect <= aspect <= max_aspect):
            continue
            
        # Add to label map
        label_map[mask] = 1
        building_count += 1
    
    print(f"Found {building_count} buildings after filtering")
    
    # 5) Save label map as TIFF
    label_meta = meta.copy()
    label_meta.update({
        'count': 1,
        'dtype': 'uint8',
        'nodata': 0
    })
    
    print(f"Saving label map to: {output_label}")
    with rasterio.open(output_label, 'w', **label_meta) as dst:
        dst.write(label_map[np.newaxis, :, :])
    
    # 6) Create and save overlay
    overlay = img_rgb.copy()
    
    # Apply colors based on label map
    for cls_idx, col in CLASS_COLORS.items():
        if cls_idx == 0:  # Skip background
            continue
        mask = label_map == cls_idx
        overlay[mask] = col
    
    # Save overlay as TIFF with same geospatial properties
    overlay_meta = meta.copy()
    overlay_meta.update({
        'count': 3,
        'dtype': 'uint8'
    })
    
    print(f"Saving overlay to: {output_overlay}")
    with rasterio.open(output_overlay, 'w', **overlay_meta) as dst:
        # Write each channel separately (rasterio uses CHW format)
        dst.write(overlay.transpose(2, 0, 1))
    
    print("Segmentation complete!")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Segment buildings with SAM - TIFF support (Fixed)")
    p.add_argument("--input_image",    required=True, help="Path to TIFF image")
    p.add_argument("--output_label",   required=True, help="Save 1-channel label TIFF")
    p.add_argument("--output_overlay", required=True, help="Save RGB overlay TIFF")
    p.add_argument(
        "--model_type", choices=["vit_b", "vit_h"], default="vit_b",
        help="SAM model size"
    )
    p.add_argument(
        "--checkpoint", default="sam_vit_b.pth",
        help="Path to SAM .pth checkpoint"
    )
    p.add_argument("--score_thresh", type=float, default=0.85, help="IoU threshold")
    p.add_argument("--min_area",     type=int,   default=500)
    p.add_argument("--max_area",     type=int,   default=50000)
    p.add_argument("--min_aspect",   type=float, default=0.5)
    p.add_argument("--max_aspect",   type=float, default=2.0)
    
    args = p.parse_args()
    
    segment_image(
        args.input_image,
        args.output_label,
        args.output_overlay,
        args.checkpoint,
        args.model_type,
        args.score_thresh,
        args.min_area,
        args.max_area,
        args.min_aspect,
        args.max_aspect,
    )