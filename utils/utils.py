
# utils.py
import logging

def get_search_instructions() -> str:
    return (
        "Based on the context/conversation history and search query above, analyze the following search results and "
        "from them synthesize a relevant, useful, and comprehensive while succinct report that addresses and answers "
        "the searched query."
    )


logger = logging.getLogger(__name__)

def log_and_handle_error(message: str, exception: Exception):
    """
    Logs an error message and raises the exception.

    Args:
        message (str): The error message to log.
        exception (Exception): The exception to raise.
    """
    logger.error(message)
    raise exception