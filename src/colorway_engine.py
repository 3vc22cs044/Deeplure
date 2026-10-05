"""
Colorway Synthesis and Palette Transformation Engine for Textile/Saree Recognition.
Generates realistic textile colorway variations while preserving geometric/motif structure.
"""

import numpy as np
from PIL import Image, ImageEnhance, ImageOps
import colorsys
import random
from typing import List, Tuple

# Curated authentic Indian textile palettes (dominant background, secondary motif, tertiary accent)
TEXTILE_PALETTES = [
    # Crimson & Antique Gold (Classic Banarasi / Bridal)
    [(180, 20, 40), (220, 180, 60), (240, 210, 120)],
    # Emerald Green & Ruby (Traditional South Handloom)
    [(20, 110, 60), (190, 30, 50), (215, 175, 70)],
    # Royal Midnight Blue & Silver Zari
    [(15, 30, 95), (200, 210, 220), (240, 245, 255)],
    # Mustard Yellow & Indigo Navy (Ikat / Gujarat Weave)
    [(225, 160, 30), (25, 45, 110), (240, 230, 200)],
    # Rani Pink & Peacock Teal
    [(210, 30, 120), (10, 140, 140), (235, 200, 80)],
    # Deep Maroon & Burnished Copper
    [(110, 20, 30), (190, 105, 55), (230, 170, 110)],
    # Pastel Mint & Dusty Peach (Modern Fusion)
    [(140, 200, 170), (235, 150, 130), (250, 235, 220)],
    # Monochrome Slate & Charcoal
    [(45, 50, 55), (140, 145, 150), (220, 225, 230)],
    # Purple Amethyst & Champagne
    [(90, 30, 100), (215, 195, 140), (245, 235, 200)],
    # Coral Red & Turquoise (Temple Weave)
    [(215, 70, 60), (30, 165, 175), (245, 220, 100)],
]


def rgb_to_hsv_np(img_arr: np.ndarray) -> np.ndarray:
    """Convert RGB float image [0, 1] to HSV float [0, 1]."""
    r, g, b = img_arr[..., 0], img_arr[..., 1], img_arr[..., 2]
    maxc = np.maximum(np.maximum(r, g), b)
    minc = np.minimum(np.minimum(r, g), b)
    v = maxc
    deltac = maxc - minc

    s = np.zeros_like(v)
    mask = maxc > 1e-6
    s[mask] = deltac[mask] / maxc[mask]

    rc = np.zeros_like(r)
    gc = np.zeros_like(g)
    bc = np.zeros_like(b)
    mask_delta = deltac > 1e-6
    rc[mask_delta] = (maxc[mask_delta] - r[mask_delta]) / deltac[mask_delta]
    gc[mask_delta] = (maxc[mask_delta] - g[mask_delta]) / deltac[mask_delta]
    bc[mask_delta] = (maxc[mask_delta] - b[mask_delta]) / deltac[mask_delta]

    h = np.zeros_like(r)
    mask_r = (r == maxc) & mask_delta
    mask_g = (g == maxc) & ~mask_r & mask_delta
    mask_b = (b == maxc) & ~mask_r & ~mask_g & mask_delta

    h[mask_r] = bc[mask_r] - gc[mask_r]
    h[mask_g] = 2.0 + rc[mask_g] - bc[mask_g]
    h[mask_b] = 4.0 + gc[mask_b] - rc[mask_b]
    h = (h / 6.0) % 1.0

    return np.stack([h, s, v], axis=-1)


def hsv_to_rgb_np(hsv: np.ndarray) -> np.ndarray:
    """Convert HSV float image [0, 1] to RGB float [0, 1]."""
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    i = np.floor(h * 6.0).astype(int)
    f = (h * 6.0) - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))

    i = i % 6
    r = np.zeros_like(h)
    g = np.zeros_like(h)
    b = np.zeros_like(h)

    m0 = i == 0
    r[m0], g[m0], b[m0] = v[m0], t[m0], p[m0]
    m1 = i == 1
    r[m1], g[m1], b[m1] = q[m1], v[m1], p[m1]
    m2 = i == 2
    r[m2], g[m2], b[m2] = p[m2], v[m2], t[m2]
    m3 = i == 3
    r[m3], g[m3], b[m3] = p[m3], q[m3], v[m3]
    m4 = i == 4
    r[m4], g[m4], b[m4] = t[m4], p[m4], v[m4]
    m5 = i == 5
    r[m5], g[m5], b[m5] = v[m5], p[m5], q[m5]

    return np.clip(np.stack([r, g, b], axis=-1), 0.0, 1.0)


def shift_hue(image: Image.Image, hue_shift_deg: float) -> Image.Image:
    """Rotate image hue by hue_shift_deg degrees (0-360) preserving luminosity/motif structure."""
    arr = np.array(image.convert("RGB"), dtype=np.float32) / 255.0
    hsv = rgb_to_hsv_np(arr)
    shift = (hue_shift_deg % 360.0) / 360.0
    hsv[..., 0] = (hsv[..., 0] + shift) % 1.0
    rgb = hsv_to_rgb_np(hsv)
    return Image.fromarray((rgb * 255.0).astype(np.uint8))


def apply_palette_transfer(image: Image.Image, palette: List[Tuple[int, int, int]]) -> Image.Image:
    """
    Quantize luminance into 3 structural textile zones (Background, Mid-tone Motif, Highlight/Zari)
    and recolor with target textile colorway palette while preserving motif texture and fine gradient details.
    """
    gray = np.array(image.convert("L"), dtype=np.float32) / 255.0
    h, w = gray.shape

    # Normalize luminance contrast
    p_low, p_high = np.percentile(gray, 2), np.percentile(gray, 98)
    if p_high > p_low + 1e-4:
        norm_gray = np.clip((gray - p_low) / (p_high - p_low), 0.0, 1.0)
    else:
        norm_gray = gray

    # Palette colors normalized
    c0 = np.array(palette[0], dtype=np.float32) / 255.0  # Background
    c1 = np.array(palette[1], dtype=np.float32) / 255.0  # Motif / Weft
    c2 = np.array(palette[2], dtype=np.float32) / 255.0  # Highlight / Zari

    # Smooth 3-stop interpolation
    # Stop 0: 0.0 -> c0
    # Stop 1: 0.5 -> c1
    # Stop 2: 1.0 -> c2
    out_rgb = np.zeros((h, w, 3), dtype=np.float32)
    mask1 = norm_gray <= 0.5
    t1 = norm_gray[mask1] / 0.5
    for ch in range(3):
        out_rgb[mask1, ch] = (1.0 - t1) * c0[ch] + t1 * c1[ch]

    mask2 = norm_gray > 0.5
    t2 = (norm_gray[mask2] - 0.5) / 0.5
    for ch in range(3):
        out_rgb[mask2, ch] = (1.0 - t2) * c1[ch] + t2 * c2[ch]

    # Re-inject high-frequency motif texture from original image
    texture_gain = 0.25
    orig_rgb = np.array(image.convert("RGB"), dtype=np.float32) / 255.0
    orig_gray = np.expand_dims(gray, -1)
    high_freq = orig_rgb - orig_gray
    recolored = np.clip(out_rgb + texture_gain * high_freq, 0.0, 1.0)

    return Image.fromarray((recolored * 255.0).astype(np.uint8))


def generate_colorways_for_motif(image: Image.Image, num_colorways: int = 6) -> List[Image.Image]:
    """
    Generate diverse, realistic colorway variants for a given saree design motif.
    Uses a mix of authentic textile palette mapping, hue shifts, channel rotations, and solarizations.
    """
    variants = [image.copy()]  # First variant is original
    
    # Selected distinct palettes
    selected_palettes = random.sample(TEXTILE_PALETTES, min(num_colorways - 1, len(TEXTILE_PALETTES)))
    for p in selected_palettes:
        variants.append(apply_palette_transfer(image, p))
        if len(variants) >= num_colorways:
            break

    # If more needed, add hue rotations
    hue_angles = [60, 120, 180, 240, 300]
    for angle in hue_angles:
        if len(variants) >= num_colorways:
            break
        variants.append(shift_hue(image, angle))

    return variants[:num_colorways]
