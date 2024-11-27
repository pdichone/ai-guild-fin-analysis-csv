import logging
import os
from datetime import datetime


class Logger:
    def __init__(self, name="csv_analyzer"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)

        # Create logs directory if it doesn't exist
        os.makedirs("logs", exist_ok=True)

        # File handler
        file_handler = logging.FileHandler(
            f"logs/{name}_{datetime.now().strftime('%Y%m%d')}.log"
        )
        file_handler.setLevel(logging.INFO)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.WARNING)

        # Formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # Add handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def info(self, message):
        self.logger.info(message)

    def warning(self, message):
        self.logger.warning(message)

    def error(self, message):
        self.logger.error(message)

    def critical(self, message):
        self.logger.critical(message)
