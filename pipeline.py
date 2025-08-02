import os
import torch
import yaml

from torch.utils.data import DataLoader
from utils.models import MnistMLP, MnistCNN
from utils.data_preprocessing import MNISTDataset, read_and_convert_idx_files
from utils.model_training import train_model
from utils.logger import setup_logger


if __name__ == '__main__':

    # CONFIGURE FILEPATHS
    init_path = os.path.abspath(__file__)

    def get_project_root(start_path):
        current = start_path
        while current != os.path.dirname(current):
            if os.path.exists(os.path.join(current, "requirements.txt")):
                return current
            current = os.path.dirname(current)
        raise FileNotFoundError("Project root was not found")

    PROJECT_ROOT_PATH = get_project_root(init_path)


    # SETUP LOGGER
    MNIST_LOG_FILE = os.path.join(
        PROJECT_ROOT_PATH, "logs", "mnist_pipeline_log.txt"
    )
    MNIST_LOGGER_NAME = "mnist_pipeline_logger"
    LOGGER = setup_logger(
        logger_name=MNIST_LOGGER_NAME,
        logger_file_path=MNIST_LOG_FILE
    )


    LOGGER.info("Starting MNIST pipeline ...")
    LOGGER.info("Configuring filepaths...")

    TRAIN_IMAGES_PATH = os.path.join(
        PROJECT_ROOT_PATH, 'data/MNIST/raw/train-images-idx3-ubyte'
    )
    TRAIN_LABELS_PATH = os.path.join(
        PROJECT_ROOT_PATH, 'data/MNIST/raw/train-labels-idx1-ubyte'
    )

    TEST_IMAGES_PATH = os.path.join(
        PROJECT_ROOT_PATH, 'data/MNIST/raw/t10k-images-idx3-ubyte'
    )
    TEST_LABELS_PATH = os.path.join(
        PROJECT_ROOT_PATH, 'data/MNIST/raw/t10k-labels-idx1-ubyte'
    )

    TRAIN_CONFIG_PATH = os.path.join(PROJECT_ROOT_PATH, 'configs/pipeline_config.yaml')

    MODEL_OUTPUT_PATH = os.path.join(PROJECT_ROOT_PATH, 'models')


    # MODEL INIT
    with open(file=TRAIN_CONFIG_PATH, mode='r', encoding='utf-8') as f:
        train_config = yaml.safe_load(f)

    LOGGER.info("Initializing model hyperparameters")
    model_architecture = train_config["used_architecture"]
    lr = train_config['lr']
    epochs = train_config['epochs']
    batch_size = train_config['batch']
    device = train_config['device']
    output_model_filename = train_config['output_model_filename']

    if model_architecture == "mlp":
        LOGGER.info("Initializing MLP model")
        model = MnistMLP()

    elif model_architecture == "cnn":
        LOGGER.info("Initializing CNN model")
        model = MnistCNN()

    else:
        LOGGER.error("Wrong value got from 'used_architecture key'")
        raise ValueError("Wrong value got from 'used_architecture key'")


    # PREPROCESS DATA
    LOGGER.info("Initializing model hyperparameters")
    train_images, train_labels, test_images, test_labels = (
        read_and_convert_idx_files(
            TRAIN_IMAGES_PATH, TRAIN_LABELS_PATH, TEST_IMAGES_PATH, TEST_LABELS_PATH
        ))

    train_dataset = MNISTDataset(labels=train_labels, images=train_images)
    test_dataset = MNISTDataset(labels=test_labels, images=test_images)

    dataloader_train = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    dataloader_test = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


    # TRAIN MODEL
    trained_model = train_model(
        model=model,
        dataloader_train=dataloader_train,
        dataloader_test=dataloader_test,
        device=device,
        lr=lr,
        epochs=epochs,
        logger=LOGGER
    )


    # SAVE MODEL
    LOGGER.info("Saving trained model")
    torch.save(model.state_dict(), os.path.join(MODEL_OUTPUT_PATH, output_model_filename))
    LOGGER.info("Model successfully saved")
    LOGGER.info("MNIST pipeline finished")