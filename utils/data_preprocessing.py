import torch
import idx2numpy
import numpy as np
from torch.utils.data import Dataset


class MNISTDataset(Dataset):
    def __init__(self, images: np.ndarray, labels: np.ndarray) -> None:
        super().__init__()
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if image.ndim == 2:
            image = torch.from_numpy(image).float()
            image = image.unsqueeze(0)
        else:
            raise ValueError(f"Unexpected image shape: {image.shape}")

        return image, label


def read_and_convert_idx_files(
    train_images_path: str, train_labels_path: str, test_images_path: str, test_labels_path: str
) -> tuple:
    """
    Метод read_and_convert_idx_files() осуществлет преобразование файлов из формата idx
    в numpy массивы
    Parameters:
        train_images_path (str): путь к файлу изображений для обучения модели
        train_labels_path (str): путь к файлу меток классов для обучения модели
        test_images_path (str): путь к файлу изображений для тестирования модели
        test_labels_path (str): путь к файлу меток классов для тестирования модели
    Returns:
        tuple(train_images, train_labels, test_images, test_labels)
    """
    train_images = idx2numpy.convert_from_file(train_images_path) / 255.0
    train_labels = idx2numpy.convert_from_file(train_labels_path)

    test_images = idx2numpy.convert_from_file(test_images_path) / 255.0
    test_labels = idx2numpy.convert_from_file(test_labels_path)

    return train_images, train_labels, test_images, test_labels