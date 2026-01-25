#!/usr/bin/env python3
"""
Generate PWA assets (icons and splash screens) from the base icon.

Usage:
    python scripts/generate_pwa_assets.py

Requirements:
    - Pillow (pip install Pillow)
    - Source icon: static/icon-512.png (512x512)
"""

import os
from pathlib import Path

try:
    from PIL import Image, ImageDraw
except ImportError:
    print("Error: Pillow is required. Install with: pip install Pillow")
    exit(1)

# Paths
SCRIPT_DIR = Path(__file__).parent
APP_DIR = SCRIPT_DIR.parent
STATIC_DIR = APP_DIR / "static"
SPLASH_DIR = STATIC_DIR / "splash"
SCREENSHOTS_DIR = STATIC_DIR / "screenshots"

# Source icon
SOURCE_ICON = STATIC_DIR / "icon-512.png"

# Icon sizes to generate
ICON_SIZES = [48, 72, 96, 128, 144, 152, 192, 384, 512]

# Splash screen sizes (width x height)
SPLASH_SIZES = [
    (640, 1136),   # iPhone SE, iPod Touch
    (750, 1334),   # iPhone 8, 7, 6s, 6
    (1242, 2208),  # iPhone 8 Plus, 7 Plus, 6s Plus, 6 Plus
    (1125, 2436),  # iPhone X, XS, 11 Pro, 12 mini, 13 mini
    (1242, 2688),  # iPhone XR, 11, XS Max, 11 Pro Max
    (1170, 2532),  # iPhone 12, 12 Pro, 13, 13 Pro, 14
    (1284, 2778),  # iPhone 12 Pro Max, 13 Pro Max, 14 Plus
    (1179, 2556),  # iPhone 14 Pro
    (1290, 2796),  # iPhone 14 Pro Max, 15 Plus, 15 Pro Max
    (1536, 2048),  # iPad Mini, Air
    (1668, 2224),  # iPad Pro 10.5"
    (1668, 2388),  # iPad Pro 11"
    (2048, 2732),  # iPad Pro 12.9"
]

# Theme colors
BG_COLOR = (26, 54, 93)  # #1a365d - Olympic blue
TEXT_COLOR = (255, 255, 255)


def create_directories():
    """Create required directories."""
    SPLASH_DIR.mkdir(exist_ok=True)
    SCREENSHOTS_DIR.mkdir(exist_ok=True)
    print(f"Created directories: {SPLASH_DIR}, {SCREENSHOTS_DIR}")


def generate_icons():
    """Generate icon variants from source icon."""
    if not SOURCE_ICON.exists():
        print(f"Error: Source icon not found: {SOURCE_ICON}")
        return False

    print(f"Loading source icon: {SOURCE_ICON}")
    source = Image.open(SOURCE_ICON)

    # Ensure source is RGBA
    if source.mode != "RGBA":
        source = source.convert("RGBA")

    for size in ICON_SIZES:
        output_path = STATIC_DIR / f"icon-{size}.png"

        # Skip if already exists and is correct size
        if output_path.exists():
            existing = Image.open(output_path)
            if existing.size == (size, size):
                print(f"  Skipping {output_path.name} (already exists)")
                continue

        # High-quality resize
        resized = source.resize((size, size), Image.Resampling.LANCZOS)
        resized.save(output_path, "PNG", optimize=True)
        print(f"  Generated: {output_path.name}")

    # Generate maskable icons (with safe zone padding)
    print("\nGenerating maskable icons...")
    for size in [192, 512]:
        output_path = STATIC_DIR / f"icon-maskable-{size}.png"

        # Create new image with background
        maskable = Image.new("RGBA", (size, size), BG_COLOR + (255,))

        # Icon should be 80% of the total size (10% padding on each side)
        icon_size = int(size * 0.8)
        padding = (size - icon_size) // 2

        # Resize and center the icon
        icon = source.resize((icon_size, icon_size), Image.Resampling.LANCZOS)
        maskable.paste(icon, (padding, padding), icon)

        maskable.save(output_path, "PNG", optimize=True)
        print(f"  Generated: {output_path.name}")

    return True


def generate_splash_screens():
    """Generate splash screens for iOS."""
    if not SOURCE_ICON.exists():
        print(f"Error: Source icon not found: {SOURCE_ICON}")
        return False

    print(f"\nGenerating splash screens...")
    source = Image.open(SOURCE_ICON)

    # Ensure source is RGBA
    if source.mode != "RGBA":
        source = source.convert("RGBA")

    for width, height in SPLASH_SIZES:
        output_path = SPLASH_DIR / f"splash-{width}x{height}.png"

        # Create splash screen with background color
        splash = Image.new("RGB", (width, height), BG_COLOR)

        # Calculate icon size (about 25% of the smaller dimension)
        icon_size = min(width, height) // 4

        # Resize icon
        icon = source.resize((icon_size, icon_size), Image.Resampling.LANCZOS)

        # Center icon (slightly above center for visual balance)
        x = (width - icon_size) // 2
        y = (height - icon_size) // 2 - (height // 20)

        # Paste icon (handle alpha channel)
        splash.paste(icon, (x, y), icon)

        # Save with optimization
        splash.save(output_path, "PNG", optimize=True)
        print(f"  Generated: {output_path.name}")

    return True


def generate_screenshots():
    """Generate placeholder screenshots for the manifest."""
    print("\nGenerating placeholder screenshots...")

    # Wide screenshot (desktop)
    wide = Image.new("RGB", (1280, 720), BG_COLOR)
    draw = ImageDraw.Draw(wide)

    # Add centered icon
    if SOURCE_ICON.exists():
        source = Image.open(SOURCE_ICON).convert("RGBA")
        icon = source.resize((200, 200), Image.Resampling.LANCZOS)
        wide.paste(icon, (540, 200), icon)

    # Add text placeholder
    draw.rectangle([(340, 450), (940, 480)], fill=(44, 82, 130))
    draw.rectangle([(440, 500), (840, 520)], fill=(66, 153, 225))

    wide.save(SCREENSHOTS_DIR / "dashboard-wide.png", "PNG", optimize=True)
    print(f"  Generated: dashboard-wide.png")

    # Narrow screenshot (mobile)
    narrow = Image.new("RGB", (540, 720), BG_COLOR)
    draw = ImageDraw.Draw(narrow)

    # Add centered icon
    if SOURCE_ICON.exists():
        source = Image.open(SOURCE_ICON).convert("RGBA")
        icon = source.resize((150, 150), Image.Resampling.LANCZOS)
        narrow.paste(icon, (195, 150), icon)

    # Add text placeholder
    draw.rectangle([(70, 350), (470, 380)], fill=(44, 82, 130))
    draw.rectangle([(120, 400), (420, 420)], fill=(66, 153, 225))

    narrow.save(SCREENSHOTS_DIR / "dashboard-narrow.png", "PNG", optimize=True)
    print(f"  Generated: dashboard-narrow.png")

    return True


def main():
    print("Ham Radio Olympics - PWA Asset Generator")
    print("=" * 50)

    create_directories()

    print("\nGenerating icon variants...")
    if not generate_icons():
        print("Warning: Icon generation failed")

    if not generate_splash_screens():
        print("Warning: Splash screen generation failed")

    if not generate_screenshots():
        print("Warning: Screenshot generation failed")

    print("\n" + "=" * 50)
    print("PWA asset generation complete!")
    print("\nGenerated files:")
    print(f"  Icons: {len(ICON_SIZES)} sizes + 2 maskable variants")
    print(f"  Splash screens: {len(SPLASH_SIZES)} sizes")
    print(f"  Screenshots: 2 (wide + narrow)")


if __name__ == "__main__":
    main()
