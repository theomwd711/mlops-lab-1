import shutil

from PIL import Image

from src.food11.data import CLASSES
from src.food11.validate_data import SPLITS, main, run_checks


def make_dataset(root, per_class=3):
    """A tiny valid dataset where every image has distinct content."""
    n = 0
    for split in SPLITS:
        for name in CLASSES:
            folder = root / split / name
            folder.mkdir(parents=True)
            for i in range(per_class):
                n += 1
                color = (n % 256, (n * 7) % 256, (n * 13) % 256)
                Image.new("RGB", (128, 128), color).save(folder / f"{i}_{n}.jpg")
    return root


def results_by_name(root):
    return {name: passed for name, passed, _ in run_checks(root, min_per_class=2)}


def test_valid_dataset_passes(tmp_path):
    root = make_dataset(tmp_path / "ds")
    assert all(results_by_name(root).values())


def test_duplicate_across_splits_is_caught(tmp_path):
    root = make_dataset(tmp_path / "ds")
    shutil.copy(root / "training" / "Bread" / "0_1.jpg", root / "evaluation" / "Bread" / "dup.jpg")
    assert results_by_name(root)["no image shared between splits"] is False


def test_wrong_size_image_is_caught(tmp_path):
    root = make_dataset(tmp_path / "ds")
    Image.new("RGB", (64, 64)).save(root / "training" / "Egg" / "small.jpg")
    assert results_by_name(root)["images are RGB 128x128"] is False


def test_missing_class_folder_is_caught(tmp_path):
    root = make_dataset(tmp_path / "ds")
    shutil.rmtree(root / "training" / "Egg")
    assert results_by_name(root)["11 class folders in every split"] is False


def test_exit_code_is_nonzero_when_a_check_fails(tmp_path):
    root = make_dataset(tmp_path / "ds")
    assert main([str(root), "--min-per-class", "2"]) == 0
    shutil.copy(root / "training" / "Bread" / "0_1.jpg", root / "evaluation" / "Bread" / "dup.jpg")
    assert main([str(root), "--min-per-class", "2"]) == 1