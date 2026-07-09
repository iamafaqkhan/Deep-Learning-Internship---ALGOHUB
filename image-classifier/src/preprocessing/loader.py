"""
loader.py
----------

Utility functions for loading the CIFAR-10 image dataset.

Responsibilities:
- Discover class names
- Build class-to-index mapping
- Load image file paths
- Read images
- Return labels

Author: Afaq Ahmad Khan
"""

from pathlib import Path
import cv2


class DatasetLoader:
    """
    Dataset loader for folder-based image datasets.

    Expected structure:

    dataset/
        train/
            cat/
            dog/
        test/
            cat/
            dog/
    """

    def __init__(self, dataset_path: str | Path):

        self.dataset_path = Path(dataset_path)

        self.train_path = self.dataset_path / "cifar10/train"

        self.test_path = self.dataset_path / "cifar10/test"

        if not self.dataset_path.exists():
            raise FileNotFoundError(
                f"Dataset not found: {self.dataset_path}"
            )

        self.classes = self._discover_classes()

        self.class_to_index = {
            name: idx
            for idx, name in enumerate(self.classes)
        }

        self.index_to_class = {
            idx: name
            for name, idx in self.class_to_index.items()
        }

    # --------------------------------------------------------

    def _discover_classes(self):
        """
        Automatically discover all class folders.
        """

        return sorted(
            folder.name
            for folder in self.train_path.iterdir()
            if folder.is_dir()
        )

    # --------------------------------------------------------

    def load_split(self, split="train"):
        """
        Load one dataset split.

        Parameters
        ----------
        split : str
            train or test

        Returns
        -------
        image_paths : list
        labels : list
        """

        if split == "train":
            root = self.train_path

        elif split == "test":
            root = self.test_path

        else:
            raise ValueError(
                "Split must be 'train' or 'test'"
            )

        image_paths = []
        labels = []

        for class_name in self.classes:

            class_folder = root / class_name

            for image_path in class_folder.glob("*"):

                image_paths.append(image_path)

                labels.append(
                    self.class_to_index[class_name]
                )

        return image_paths, labels

    # --------------------------------------------------------

    @staticmethod
    def read_image(image_path):
        """
        Read one image.

        Parameters
        ----------
        image_path : Path

        Returns
        -------
        RGB Image
        """

        image = cv2.imread(str(image_path))

        if image is None:
            raise ValueError(
                f"Cannot read image: {image_path}"
            )

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        return image

    # --------------------------------------------------------

    def dataset_summary(self):

        print("=" * 50)

        print("Dataset:", self.dataset_path)

        print("Classes:", len(self.classes))

        print()

        for cls in self.classes:

            train_count = len(
                list(
                    (self.train_path / cls).glob("*")
                )
            )

            test_count = len(
                list(
                    (self.test_path / cls).glob("*")
                )
            )

            print(
                f"{cls:12s}"
                f" Train: {train_count:5d}"
                f" Test: {test_count:5d}"
            )

        print("=" * 50)