"""
status.py

This module provides the definition of various status codes and a
utility function for generating status messages that can be returned
from the backend application to the GUI.

Author: Milan Stubljar <info@stubljar.com>
Date: 2024-11-03
"""

class Status:
    """
    Status codes and messages for command responses.
    """

    SUCCESS = 0
    ERROR = 1
    BUSY = 2

    @staticmethod
    def get_status(error_code, message):
        """
        Generate a status dictionary for responses.

        Args:
            error_code (int): Status code, as defined in Status.
            message (str): Status message.

        Returns:
            dict: Status as a dictionary with code and message.
        """
        return {
            "error_code": error_code,
            "error_message": message
        }
