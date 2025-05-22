#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 20 23:54:20 2025

@author: umutmurat
"""

import os
import glob

# Set your directories
image_dir = "/Users/umutmurat/Documents/Documents/SegmentationBuildings/BuildingImages"
mask_dir = "/Users/umutmurat/Documents/Documents/SegmentationBuildings/BuildingMasks"

# Find all tif files
images = glob.glob(os.path.join(image_dir, "*.tif"))
masks = glob.glob(os.path.join(mask_dir, "*.tif"))  # adjust pattern if needed

# Split into train/val (e.g., 80/20 split)
train_size = int(0.8 * len(images))
train_images = images[:train_size]
val_images = images[train_size:]

train_masks = masks[:train_size]
val_masks = masks[train_size:]

# Write to files
with open("train_images.txt", "w") as f:
    f.write("\n".join(train_images))
    
with open("train_masks.txt", "w") as f:
    f.write("\n".join(train_masks))
    
with open("val_images.txt", "w") as f:
    f.write("\n".join(val_images))
    
with open("val_masks.txt", "w") as f:
    f.write("\n".join(val_masks))