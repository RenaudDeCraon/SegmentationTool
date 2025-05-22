"""
Train an FCN-ResNet50 building segmentation model with TIFF/TIF support.

Usage:
    python train_building_segmentation_tiff.py \
        --train_images train_images.txt \
        --train_masks  train_masks.txt \
        --val_images   val_images.txt \
        --val_masks    val_masks.txt \
        --epochs 20 \
        --batch_size 8 \
        --lr 1e-4 \
        --output_model fcn_building_seg.pth
"""
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models.segmentation import fcn_resnet50
import numpy as np
import rasterio
from rasterio.plot import reshape_as_image
import os

class BuildingDataset(Dataset):
    def __init__(self, img_list, mask_list, transform=None):
        assert len(img_list) == len(mask_list)
        self.images = img_list
        self.masks = mask_list
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        mask_path = self.masks[idx]
        
        # Load TIFF/TIF image using rasterio
        with rasterio.open(img_path) as src:
            img_array = src.read()
            img = reshape_as_image(img_array)
            
            # Handle different band combinations
            if img.shape[2] > 3:
                # If more than 3 bands, use first 3 for RGB
                img = img[:, :, :3]
            elif img.shape[2] == 1:
                # If single band, convert to RGB
                img = np.repeat(img, 3, axis=2)
                
        # Normalize to 0-1 range if needed
        if img.dtype != np.uint8:
            img = (img - img.min()) / (img.max() - img.min())
            if img.max() <= 1.0:  # If already normalized, scale to 0-255
                img = (img * 255).astype(np.uint8)
        
        # Load mask (could be TIFF or other format)
        _, ext = os.path.splitext(mask_path)
        if ext.lower() in ['.tif', '.tiff']:
            with rasterio.open(mask_path) as src:
                mask_array = src.read(1)  # Assuming single band mask
                mask = mask_array.astype(np.float32) / 255.0  # Normalize to 0-1
        else:
            # Fallback to OpenCV for other formats
            import cv2
            mask = cv2.imread(mask_path, 0) / 255.0  # Grayscale, normalize to 0-1
        
        # Apply transformations
        if self.transform:
            # Convert to PIL for torchvision transforms
            from PIL import Image
            img_pil = Image.fromarray(img.astype(np.uint8))
            img = self.transform(img_pil)
            
            # For mask, we need special handling to keep it binary
            mask_pil = Image.fromarray((mask * 255).astype(np.uint8))
            # Use same transform but without color normalization
            mask_tf = transforms.Compose([
                transforms.Resize(512),
                transforms.ToTensor()
            ])
            mask = mask_tf(mask_pil)[0]  # Keep only first channel
        else:
            # Without transform, manually convert to tensor
            img = torch.tensor(img.transpose(2, 0, 1), dtype=torch.float32) / 255.0  # CHW format
            mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)  # Add channel dim: 1×H×W
            
        return img, mask

def load_paths(txtfile):
    with open(txtfile, 'r') as f:
        return [line.strip() for line in f if line.strip()]

def main(args):
    # 1) Prepare data
    tf = transforms.Compose([
        transforms.Resize(512),
        transforms.ToTensor(),
    ])
    
    train_imgs = load_paths(args.train_images)
    train_msks = load_paths(args.train_masks)
    val_imgs = load_paths(args.val_images)
    val_msks = load_paths(args.val_masks)

    train_ds = BuildingDataset(train_imgs, train_msks, transform=tf)
    val_ds = BuildingDataset(val_imgs, val_msks, transform=tf)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # 2) Model, loss, optimizer
    model = fcn_resnet50(pretrained=False, num_classes=1)
    model.classifier[4] = nn.Conv2d(512, 1, kernel_size=1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # 3) Training loop
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0
        for imgs, msks in train_loader:
            imgs, msks = imgs.to(device), msks.to(device)
            preds = model(imgs)["out"]
            loss = criterion(preds, msks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_train_loss = total_loss / len(train_loader)

        # 4) Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for imgs, msks in val_loader:
                imgs, msks = imgs.to(device), msks.to(device)
                preds = model(imgs)["out"]
                val_loss += criterion(preds, msks).item()
        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch {epoch:02d}  Train Loss: {avg_train_loss:.4f}  Val Loss: {avg_val_loss:.4f}")

    # 5) Save checkpoint
    torch.save(model.state_dict(), args.output_model)
    print("Model saved to", args.output_model)

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train building segmentation with TIFF support")
    p.add_argument("--train_images", required=True, help="TXT file with training image paths")
    p.add_argument("--train_masks", required=True, help="TXT file with training mask paths")
    p.add_argument("--val_images", required=True, help="TXT file with validation image paths")
    p.add_argument("--val_masks", required=True, help="TXT file with validation mask paths")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--output_model", default="fcn_building_seg.pth")
    args = p.parse_args()
    main(args)