# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import random
import shutil
from pathlib import Path
from typing import Tuple, Union

from ultralytics.data.utils import IMG_FORMATS, img2label_paths
from ultralytics.utils import DATASETS_DIR, LOGGER, TQDM


def split_classify_dataset(source_dir: Union[str, Path], train_ratio: float = 0.8) -> Path:
    """
将分类数据集分割为训练集和验证集目录，并存放于新目录中。

创建一个名为“{source_dir}_split”的新目录，其中包含train/val子目录，默认按照80/20的比例分割，同时保留原始类别结构。

目录结构：
    分割前：
        caltech/
        ├── class1/
        │   ├── img1.jpg
        │   ├── img2.jpg
        │   └── ...
        ├── class2/
        │   ├── img1.jpg
        │   └── ...
        └── ...

    分割后：
        caltech_split/
        ├── train/
        │   ├── class1/
        │   │   ├── img1.jpg
        │   │   └── ...
        │   ├── class2/
        │   │   ├── img1.jpg
        │   │   └── ...
        │   └── ...
        └── val/
            ├── class1/
            │   ├── img2.jpg
            │   └── ...
            ├── class2/
            │   └── ...
            └── ...

参数：
    source_dir (str | Path): 分类数据集根目录的路径
    train_ratio (float): 训练集分割比例，取值范围为0到1

返回：
    (Path): 创建的拆分目录路径

示例：
    使用默认80/20比例分割数据集
    >>> split_classify_dataset("path/to/caltech")

    使用自定义比例分割
    >>> split_classify_dataset("path/to/caltech", 0.75)
    """
    source_path = Path(source_dir)
    split_path = Path(f"{source_path}_split")
    train_path, val_path = split_path / "train", split_path / "val"

    # Create directory structure
    split_path.mkdir(exist_ok=True)
    train_path.mkdir(exist_ok=True)
    val_path.mkdir(exist_ok=True)

    # Process class directories
    class_dirs = [d for d in source_path.iterdir() if d.is_dir()]
    total_images = sum(len(list(d.glob("*.*"))) for d in class_dirs)
    stats = f"{len(class_dirs)} classes, {total_images} images"
    LOGGER.info(f"Splitting {source_path} ({stats}) into {train_ratio:.0%} train, {1 - train_ratio:.0%} val...")

    for class_dir in class_dirs:
        # Create class directories
        (train_path / class_dir.name).mkdir(exist_ok=True)
        (val_path / class_dir.name).mkdir(exist_ok=True)

        # Split and copy files
        image_files = list(class_dir.glob("*.*"))
        random.shuffle(image_files)
        split_idx = int(len(image_files) * train_ratio)

        for img in image_files[:split_idx]:
            shutil.copy2(img, train_path / class_dir.name / img.name)

        for img in image_files[split_idx:]:
            shutil.copy2(img, val_path / class_dir.name / img.name)

    LOGGER.info(f"Split complete in {split_path} ✅")
    return split_path


def autosplit(
    path: Path = DATASETS_DIR / "coco8/images",
    weights: Tuple[float, float, float] = (0.9, 0.1, 0.0),
    annotated_only: bool = False,
) -> None:
    """
    Automatically split a dataset into train/val/test splits and save the resulting splits into autosplit_*.txt files.

    Args:
        path (Path): Path to images directory.
        weights (tuple): Train, validation, and test split fractions.
        annotated_only (bool): If True, only images with an associated txt file are used.

    Examples:
        Split images with default weights
        >>> from ultralytics.data.split import autosplit
        >>> autosplit()

        Split with custom weights and annotated images only
        >>> autosplit(path="path/to/images", weights=(0.8, 0.15, 0.05), annotated_only=True)
    """
    path = Path(path)  # images dir
    files = sorted(x for x in path.rglob("*.*") if x.suffix[1:].lower() in IMG_FORMATS)  # image files only
    n = len(files)  # number of files
    random.seed(0)  # for reproducibility
    indices = random.choices([0, 1, 2], weights=weights, k=n)  # assign each image to a split

    txt = ["autosplit_train.txt", "autosplit_val.txt", "autosplit_test.txt"]  # 3 txt files
    for x in txt:
        if (path.parent / x).exists():
            (path.parent / x).unlink()  # remove existing

    LOGGER.info(f"Autosplitting images from {path}" + ", using *.txt labeled images only" * annotated_only)
    for i, img in TQDM(zip(indices, files), total=n):
        if not annotated_only or Path(img2label_paths([str(img)])[0]).exists():  # check label
            with open(path.parent / txt[i], "a", encoding="utf-8") as f:
                f.write(f"./{img.relative_to(path.parent).as_posix()}" + "\n")  # add image to txt file


if __name__ == "__main__":
    split_classify_dataset("caltech101")
