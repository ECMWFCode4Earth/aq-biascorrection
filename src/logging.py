import logging
import sys


def get_logger(name: str) -> logging.Logger:
    """
    Returns a logger with an INFO level Sets propagate to False to avoid messes
    maintaining a logger/handler hierarchy.

    Parameters
    ----------
    name: str
      Name for the logger

    Returns
    -------
    logging.Logger instance with a handler and a formatter already set, logging
    to stdout.

    """
    logger = logging.getLogger(name)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("%(asctime)s — %(name)s — %(levelname)s — %(message)s")
    )
    logger.addHandler(handler)
    logger.setLevel("INFO")
    logger.propagate = False
    return logger
