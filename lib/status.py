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
    Standardized status codes and response formatting for command execution.
    """

    # Status codes (int)
    SUCCESS = 0
    ERROR = 1
    BUSY = 2
    INVALID_INPUT = 3  # Example: Add more codes as needed

    # Optional: Status code -> default message mapping
    _DEFAULT_MESSAGES = {
        SUCCESS: "Success",
        ERROR: "General error",
        BUSY: "System busy",
        INVALID_INPUT: "Invalid input parameters"
    }

    @classmethod
    def get_status(cls, error_code, message=None, data=None):
        """
        Generate a standardized status response dictionary.

        Args:
            error_code (int): Status code (e.g., Status.SUCCESS).
            message (str, optional): Human-readable message. Defaults to predefined messages.
            data (Any, optional): Additional response payload.

        Returns:
            dict: {
                "error_code": int,
                "error_message": str,
                "data": Any (optional)
            }

        Raises:
            ValueError: If error_code is invalid.
        """
        if not hasattr(cls, "_DEFAULT_MESSAGES") or error_code not in cls._DEFAULT_MESSAGES:
            raise ValueError(f"Invalid status code: {error_code}")

        status = {
            "error_code": error_code,
            "error_message": message if message is not None else cls._DEFAULT_MESSAGES[error_code]
        }

        if data is not None:
            status["data"] = data

        return status

    # Optional helper methods
    @classmethod
    def success(cls, data=None, message=None):
        """Shortcut for success responses."""
        return cls.get_status(cls.SUCCESS, message, data)

    @classmethod
    def error(cls, data=None, message=None):
        """Shortcut for error responses."""
        return cls.get_status(cls.ERROR, message, data)