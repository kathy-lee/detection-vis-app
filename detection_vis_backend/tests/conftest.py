from loguru import logger


def pytest_configure(config):
    logger.add("./tests/test_logs.log", format="{time} {level} {message}", level="INFO")