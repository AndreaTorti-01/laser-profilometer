import sys
import os
from PIL import Image, ImageOps

def main():
    if len(sys.argv) != 2:
        print("Usage: python giffy.py <folder>")
        sys.exit(1)
    folder = sys.argv[1]
    if not os.path.isdir(folder):
        print(f"Error: {folder} is not a directory.")
        sys.exit(1)

    png_files = sorted(
        [f for f in os.listdir(folder) if f.lower().endswith('.png')]
    )
    if not png_files:
        print("No PNG files found in the folder.")
        sys.exit(1)

    images = []
    for f in png_files:
        img = Image.open(os.path.join(folder, f)).convert("RGB")
        img_eq = ImageOps.equalize(img)
        img_blend = Image.blend(img, img_eq, alpha=0.1)
        w, h = img_blend.size
        img_resized = img_blend.resize((w // 2, h // 2), Image.LANCZOS)
        images.append(img_resized)
    output_path = os.path.join(folder, "output.gif")
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=100,
        loop=0,
        optimize=True
    )
    print(f"GIF saved to {output_path}")

if __name__ == "__main__":
    main()
