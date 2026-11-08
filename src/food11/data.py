
"""
src/food11/data.py

Prepares the Food-11 dataset for ResNet training.

Reads:
    ./data/food11_raw/{training,evaluation,validation}/<category_index>_<n>.jpg

Writes:
    ./data/food11_processed/{split}/{category_name}/*.jpg
    ./data/food11_processed_mini/{split}/{category_name}/*.jpg   (max 100 per category)

Both outputs contain images resized to 128x128 and sorted into folders
named after their category (instead of being flat with the category
encoded only in the filename).
"""

from pathlib import Path

from PIL import Image

CLASSES = [
    "Bread",
    "Dairy product",
    "Dessert",
    "Egg",
    "Fried food",
    "Meat",
    "Noodles-Pasta",
    "Rice",
    "Seafood",
    "Soup",
    "Vegetable-Fruit",
]

# Food-11 category mapping: filename prefix -> category folder name
CATEGORIES = dict(enumerate(CLASSES))

RAW_DIR = Path("./data/food11_raw")
PROCESSED_DIR = Path("./data/food11_processed")
PROCESSED_MINI_DIR = Path("./data/food11_processed_mini")

SPLITS = ["training", "evaluation", "validation"]
IMAGE_SIZE = 128
MINI_LIMIT = 100


def get_category_index(stem: str) -> int:
    """Food-11 filenames start with the category index, e.g. '3_142' -> 3."""
    return int(stem.split("_")[0])


def category_from_filename(filename: str) -> str:
    """'3_120.jpg' -> 'Egg'. Raises ValueError for names it can't map."""
    index = get_category_index(Path(filename).stem)
    if not 0 <= index < len(CLASSES):
        raise ValueError(f"unknown category index {index} in {filename!r}")
    return CLASSES[(index + 1) % len(CLASSES)]


def resize_image(img: Image.Image, size: int = IMAGE_SIZE) -> Image.Image:
    """Returns an RGB image of size x size."""
    return img.convert("RGB").resize((size, size), Image.LANCZOS)


def process_split(split: str) -> None:
    split_dir = RAW_DIR / split
    if not split_dir.exists():
        print(f"Skipping missing split: {split_dir}")
        return

    mini_counts = {cat_idx: 0 for cat_idx in CATEGORIES}

    image_paths = sorted(split_dir.glob("*.jpg"))
    print(f"[{split}] found {len(image_paths)} images")

    for img_path in image_paths:
        try:
            cat_idx = get_category_index(img_path.stem)
        except ValueError:
            print(f"  skipping unrecognized filename: {img_path.name}")
            continue

        category_name = CATEGORIES.get(cat_idx)
        if category_name is None:
            print(f"  skipping unknown category index {cat_idx}: {img_path.name}")
            continue

        with Image.open(img_path) as im:
            im = resize_image(im)

            out_dir = PROCESSED_DIR / split / category_name
            out_dir.mkdir(parents=True, exist_ok=True)
            im.save(out_dir / img_path.name)

            if mini_counts[cat_idx] < MINI_LIMIT:
                mini_out_dir = PROCESSED_MINI_DIR / split / category_name
                mini_out_dir.mkdir(parents=True, exist_ok=True)
                im.save(mini_out_dir / img_path.name)
                mini_counts[cat_idx] += 1

    print(f"[{split}] done. mini counts per category: {mini_counts}")


def main() -> None:
    for split in SPLITS:
        process_split(split)


if __name__ == "__main__":
    main()
