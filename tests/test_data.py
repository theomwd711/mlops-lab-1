import pytest
from PIL import Image

from src.food11.data import CLASSES, category_from_filename, resize_image


def test_category_from_filename():
    assert category_from_filename("0_12.jpg") == "Bread"
    assert category_from_filename("3_120.jpg") == "Egg"
    assert category_from_filename("10_3.jpg") == "Vegetable-Fruit"


def test_category_from_filename_rejects_bad_names():
    with pytest.raises(ValueError):
        category_from_filename("cat.jpg")
    with pytest.raises(ValueError):
        category_from_filename("11_1.jpg")


def test_resize_image_shape_and_mode():
    img = Image.new("L", (512, 384))  # a grayscale, non-square image
    out = resize_image(img)
    assert out.size == (128, 128)
    assert out.mode == "RGB"


def test_eleven_classes():
    assert len(CLASSES) == 11
