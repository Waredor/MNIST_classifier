import os
import logging

from logging.handlers import RotatingFileHandler

def setup_logger(
        logger_name: str,
        logger_file_path: str,
        use_file_handler: bool = True
) -> logging.Logger:
    """
    Метод setup_logger() создает объект logging.Logger для логирования работы пайплайна
    Parameters:
        logger_name (str): имя создаваемого логгера.
        logger_file_path (str): путь к .txt файлу лога.
        use_file_handler (bool): флаг, отвечающий за создание файла лога.
    Returns:
        logger (logging.Logger): объект логгера.
    """
    logger = logging.getLogger(logger_name)

    if not logger.handlers:
        logger.setLevel(logging.INFO)

        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)

        formatter = logging.Formatter(
            '%(filename)s[LINE:%(lineno)d]# %(levelname)-8s [%(asctime)s] %(message)s'
        )
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        if use_file_handler:
            os.makedirs(os.path.dirname(logger_file_path), exist_ok=True)
            file_handler = RotatingFileHandler(
                logger_file_path,
                maxBytes=1048576,
                backupCount=3
            )
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger