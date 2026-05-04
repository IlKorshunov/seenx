import logging
import os
import sys


FORMATTER = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
LOG_FILE = os.path.join(os.getcwd(), "logfile.log")


class Logger:
    """
    Singleton Logger class for unified logging across modules.
    """

    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, show: bool = False):
        if self._initialized:
            return

        self._initialized = True

        self.log_level = logging.DEBUG
        self.logger = logging.getLogger("seenx")
        self.logger.setLevel(self.log_level)
        self.logger.handlers.clear()
        if show:
            self.logger.addHandler(self._get_console_handler())
        self.logger.addHandler(self._get_file_handler())
        self.logger.propagate = False

    def _get_console_handler(self):
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(FORMATTER)
        handler.setLevel(self.log_level)
        return handler

    def _get_file_handler(self):
        handler = logging.FileHandler(LOG_FILE, mode="w")
        handler.setFormatter(FORMATTER)
        handler.setLevel(self.log_level)
        return handler

    def get_logger(self):
        return self.logger
