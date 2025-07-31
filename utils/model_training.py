import logging
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader


def softmax(y) -> float:
    y = y.detach().numpy()
    exp_y = np.exp(y)
    sum_exp_y = np.sum(exp_y)
    softmax_y = exp_y / sum_exp_y
    return softmax_y

def train_model(
        model: nn.Module,
        lr: float,
        epochs: int,
        device: str,
        dataloader_train: DataLoader,
        dataloader_test: DataLoader,
        logger: logging.Logger
) -> nn.Module:
    """
    Метод train_model() осуществляет обучение выбранной модели
    Parameters:
        model (nn.Module): модель для обучения
        lr (float): learning rate
        epochs (int): количество эпох обучения
        device (str): устройство для обучения ('cpu', 'cuda')
        dataloader_train (DataLoader): объект класса DataLoader для тренировочной выборки
        dataloader_test (DataLoader): объект класса DataLoader для тестовой выборки
        logger (logging.Logger): объект логгера
    """
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    logger.info("Start training")

    for epoch in range(epochs):
        batch_train_count = 0
        batch_test_count = 0
        sumloss_train = 0
        sumloss_test = 0
        model.train()
        logger.info(f'starting training epoch {epoch + 1}..')
        for batch_idx, (images, labels) in enumerate(dataloader_train):
            batch_train_count += 1
            images, labels = images.to(device), labels.to(device)
            if images.shape[1] != 1:
                images = images.permute(1, 0, 2, 3)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            sumloss_train += loss.item()
        sumloss_train /= batch_train_count
        model.eval()
        logger.info(f'starting validation epoch {epoch + 1}..')
        with torch.no_grad():
            eval_accuracy = 0
            for batch_idx, (images, labels) in enumerate(dataloader_test):
                tp = 0
                batch_test_count += 1
                images = images.to(device)
                labels = labels.to(device)
                logits = model(images)
                loss = criterion(logits, labels)
                proba = softmax(logits)
                labels = np.array(labels)
                sumloss_test += loss.item()
                for i in range(proba.shape[0]):
                    max_idx = np.argmax(proba[i])
                    if max_idx == labels[i]:
                        tp += 1
                batch_accuracy = tp / int(proba.shape[0])
                eval_accuracy += batch_accuracy
            eval_accuracy /= batch_test_count
            sumloss_test /= batch_test_count
        logger.info(f'epoch {epoch + 1} train_loss: {sumloss_train},'
              f' test_loss: {sumloss_test}, accuracy: {eval_accuracy}')

    logger.info("Training finished")
    return model