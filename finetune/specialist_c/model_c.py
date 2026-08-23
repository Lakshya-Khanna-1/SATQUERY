"""
SatQuery AI — Specialist C Architecture
SIH 2026 (PS 26167)

Specialist C: Multimodal Optical-SAR Fusion & Pixel-Level Change Segmentation.
Components:
1. Dual-Branch Contrastive Encoder:
   - Optical Branch: ResNet/Conv backbone -> 256-d joint embedding
   - SAR Branch: 2-band (VV/VH) backbone -> 256-d joint embedding
   - Cross-Modal Contrastive Alignment (InfoNCE)
2. Bi-temporal U-Net Segmentation Head:
   - Input: 6-channel stacked pair (im1 + im2)
   - Output: 6-class land-cover change mask (Unchanged, Water, Forest, Vegetation, Building, Bareland)
   - Visual RGBA Overlay Generation
"""

import io
import base64
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms
from PIL import Image
from typing import Dict, Any, Tuple, Optional


# SECOND / SatQuery 6 Land-Cover Change Classes & RGBA Color Map
CHANGE_CLASSES = [
    "Unchanged / Background",
    "Water Change",
    "Forest / Tree Change",
    "Low Vegetation / Agriculture Change",
    "Building / Urban Change",
    "Playground / Bareland Change"
]

CHANGE_COLOR_MAP = {
    0: (0, 0, 0, 0),         # Unchanged -> 100% Transparent
    1: (0, 150, 255, 200),    # Water -> Blue
    2: (34, 139, 34, 200),    # Forest -> Dark Green
    3: (124, 252, 0, 200),    # Low Veg -> Light Green
    4: (255, 69, 0, 220),     # Building -> Red-Orange
    5: (255, 215, 0, 200)     # Playground/Bare -> Gold/Yellow
}


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class BiTemporalUNet(nn.Module):
    """
    Lightweight U-Net for 6-channel bi-temporal change segmentation.
    """
    def __init__(self, in_channels=6, num_classes=6):
        super().__init__()
        self.enc1 = ConvBlock(in_channels, 64)
        self.enc2 = ConvBlock(64, 128)
        self.enc3 = ConvBlock(128, 256)
        self.enc4 = ConvBlock(256, 512)

        self.pool = nn.MaxPool2d(2)

        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(128, 64)

        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        # x: (B, 6, H, W)
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        d3 = self.dec3(torch.cat([self.up3(e4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        logits = self.final_conv(d1)
        return logits


class DualBranchEncoder(nn.Module):
    """
    Dual-branch encoder producing unified 256-d embeddings for Optical and SAR modalities.
    """
    def __init__(self, embedding_dim=256):
        super().__init__()
        
        # Optical branch (3 input channels)
        opt_backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.optical_encoder = nn.Sequential(*list(opt_backbone.children())[:-1])
        self.optical_proj = nn.Linear(512, embedding_dim)

        # SAR branch (2 input channels for VV/VH)
        sar_backbone = models.resnet18(weights=None)
        # Adapt first conv layer to accept 2 channels instead of 3
        sar_backbone.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.sar_encoder = nn.Sequential(*list(sar_backbone.children())[:-1])
        self.sar_proj = nn.Linear(512, embedding_dim)

    def encode_optical(self, x_opt: torch.Tensor) -> torch.Tensor:
        # x_opt: (B, 3, H, W)
        feat = self.optical_encoder(x_opt).squeeze(-1).squeeze(-1)
        emb = self.optical_proj(feat)
        return F.normalize(emb, p=2, dim=-1)

    def encode_sar(self, x_sar: torch.Tensor) -> torch.Tensor:
        # x_sar: (B, 2, H, W)
        feat = self.sar_encoder(x_sar).squeeze(-1).squeeze(-1)
        emb = self.sar_proj(feat)
        return F.normalize(emb, p=2, dim=-1)

    def forward(self, x_opt: torch.Tensor, x_sar: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        emb_opt = self.encode_optical(x_opt)
        emb_sar = self.encode_sar(x_sar)
        return emb_opt, emb_sar


class SpecialistC(nn.Module):
    """
    Unified Specialist C Model uniting Dual-Branch Fusion and Bi-temporal Change Segmentation.
    """
    def __init__(self, embedding_dim=256, num_classes=6):
        super().__init__()
        self.encoder = DualBranchEncoder(embedding_dim=embedding_dim)
        self.segmentor = BiTemporalUNet(in_channels=6, num_classes=num_classes)

        self.transform_opt = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.transform_seg = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])

    def encode_pair(self, opt_img: Image.Image, sar_array: np.ndarray) -> Dict[str, Any]:
        """
        Calculates joint cross-modal embedding between an optical image and a 2-band SAR array.
        """
        device = next(self.parameters()).device
        
        # Preprocess Optical
        opt_tensor = self.transform_opt(opt_img.convert("RGB")).unsqueeze(0).to(device)
        
        # Preprocess SAR (2 channels VV, VH)
        if isinstance(sar_array, np.ndarray):
            if sar_array.ndim == 2:
                sar_array = np.stack([sar_array, sar_array], axis=0)
            elif sar_array.shape[0] > 2:
                sar_array = sar_array[:2]
        sar_tensor = torch.from_numpy(sar_array).float().unsqueeze(0).to(device)
        sar_tensor = F.interpolate(sar_tensor, size=(256, 256), mode="bilinear", align_corners=False)
        
        # Normalize SAR
        sar_tensor = (sar_tensor - sar_tensor.mean()) / (sar_tensor.std() + 1e-6)

        with torch.no_grad():
            emb_opt, emb_sar = self.encoder(opt_tensor, sar_tensor)
            sim = F.cosine_similarity(emb_opt, emb_sar).item()
            joint_emb = (emb_opt + emb_sar) / 2.0
            joint_emb = F.normalize(joint_emb, p=2, dim=-1)

        return {
            "optical_embedding": emb_opt.squeeze(0).cpu().tolist(),
            "sar_embedding": emb_sar.squeeze(0).cpu().tolist(),
            "joint_embedding": joint_emb.squeeze(0).cpu().tolist(),
            "cross_modal_similarity": float(sim),
            "modality_alignment_confidence": max(0.0, min(1.0, (float(sim) + 1.0) / 2.0))
        }

    def generate_change_map(self, im1: Image.Image, im2: Image.Image) -> Dict[str, Any]:
        """
        Takes bi-temporal optical images (im1, im2) and produces 6-class segmentation mask & RGBA PNG overlay.
        """
        device = next(self.parameters()).device
        
        t1 = self.transform_seg(im1.convert("RGB")).unsqueeze(0).to(device)
        t2 = self.transform_seg(im2.convert("RGB")).unsqueeze(0).to(device)
        
        # Concatenate into 6 channels
        pair_tensor = torch.cat([t1, t2], dim=1)

        with torch.no_grad():
            logits = self.segmentor(pair_tensor)
            pred_mask = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()

        # Build RGBA Overlay
        h, w = pred_mask.shape
        rgba_img = np.zeros((h, w, 4), dtype=np.uint8)
        class_pixel_counts = {}

        for cls_id, cls_name in enumerate(CHANGE_CLASSES):
            mask_cls = (pred_mask == cls_id)
            count = int(np.sum(mask_cls))
            class_pixel_counts[cls_name] = count
            rgba_img[mask_cls] = CHANGE_COLOR_MAP[cls_id]

        overlay_pil = Image.fromarray(rgba_img, mode="RGBA")
        
        # Base64 encode for API transport
        buffered = io.BytesIO()
        overlay_pil.save(buffered, format="PNG")
        overlay_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        total_changed_pixels = sum(count for cls, count in class_pixel_counts.items() if cls != CHANGE_CLASSES[0])
        pct_changed = (total_changed_pixels / (h * w)) * 100

        return {
            "mask_shape": [h, w],
            "total_pixels": h * w,
            "changed_pixels": total_changed_pixels,
            "percentage_area_changed": round(pct_changed, 2),
            "class_breakdown": class_pixel_counts,
            "overlay_png_base64": overlay_base64,
            "overlay_image": overlay_pil
        }
