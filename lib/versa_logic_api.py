"""
# VersaAPI Python Wrapper for VL-EPU-4012 SBC

This Python script provides a wrapper for the VersaLogic C API to control GPIO pins on the VL-EPU-4012
Single Board Computer (SBC). It allows you to set a GPIO pin to HIGH or LOW using Python.

## Prerequisites

1. **Ubuntu**: Ensure you are running Ubuntu (tested on 20.04 LTS).
2. **Python 3.x**: Install Python 3.x if not already installed.
3. **VersaLogic C API Library**: Ensure the VersaLogic C API library (`libversalogic.so`) is installed on your system.

## Installation

1. **Install Python 3.x**:

```bash
sudo apt update
sudo apt install python3
```

2. **Install ctypes (usually included with Python)**:

```bash
sudo apt install python3-ctypes
```

3. **Install VersaLogic C API Library**:

Download the library from VersaLogic's website.

Follow the installation instructions provided by VersaLogic.

Ensure the shared library (libversalogic.so) is in a standard library path like /usr/lib.
"""

import ctypes
import os
import time
import logging
from logging import getLogger
from lib.common import get_os_type

class VersaAPI:
    """
    A Python wrapper for the VersaLogic C API to control GPIO pins on the VL-EPU-4012 SBC.
    """

    def __init__(self):
        """
        Initialize the VersaAPI class by loading the VersaLogic C API library.
        """
        self.log = getLogger('pueo')
        if get_os_type() == 'Windows':
            return

        lib_file = '~/Projects/install/libVL_OSALib.1.8.4.so'
        lib_filename = os.path.abspath(os.path.expanduser(lib_file))

        try:
            # Load the VersaLogic OSALib library
            self.lib = ctypes.CDLL(lib_filename)
            self.log.info("VersaLogic OSALib library loaded successfully.")
        except OSError as e:
            self.log.error(f"Failed to load VersaLogic OSALib library: {e}")
            raise RuntimeError(f"Failed to load VersaLogic OSALib library: {e}")

        try:
            # Open the VersaAPI library
            if hasattr(self.lib, 'VSL_Open'):
                self.lib.VSL_Open.argtypes = []
                self.lib.VSL_Open.restype = ctypes.c_ulong

                result = self.lib.VSL_Open()
                if result != 0:
                    self.log.error(f"Failed to open VersaAPI library. Error code: {result}")
                    raise RuntimeError(f"Failed to open VersaAPI library. Error code: {result}")
                self.log.info("VersaAPI library opened successfully.")
            else:
                self.log.error("VSL_Open function not found in the library.")
                raise RuntimeError("VSL_Open function not found in the library.")
        except Exception as e:
            self.log.error(f"Failed to initialize VersaAPI library: {e}")
            raise RuntimeError(f"Failed to initialize VersaAPI library: {e}")

        # Define DIO level constants
        self.DIO_CHANNEL_LOW = 0x00
        self.DIO_CHANNEL_HIGH = 0x01

        # Define DIO direction constants
        self.DIO_OUTPUT = 0x00
        self.DIO_INPUT = 0x01

        ret = self.lib.VSL_Open()
        self.log.info(f'VSL_Open: {ret}')
        ver = self.get_version()
        self.log.info(f'Version: {ver}')
        info = self.get_product_info()
        self.log.info(f'Product info')
        for k, v in info.items():
            self.log.info(f'  {k:>11s}: {v}')
        uptime = self.lib.VSL_GetUptime()
        self.log.info(f'Board uptime: {uptime}')

    def get_version(self):
        """
        Retrieve the version number of the VersaAPI library.

        Returns:
            dict: A dictionary containing the major, minor, and revision version numbers.
                  Example: {'major': 1, 'minor': 8, 'revision': 4}

        Raises:
            RuntimeError: If the VSL_GetVersion function is not found in the library.
        """
        try:
            if hasattr(self.lib, 'VSL_GetVersion'):
                # Define the function signature
                self.lib.VSL_GetVersion.argtypes = [
                    ctypes.POINTER(ctypes.c_ubyte),  # Major version (unsigned char *)
                    ctypes.POINTER(ctypes.c_ubyte),  # Minor version (unsigned char *)
                    ctypes.POINTER(ctypes.c_ubyte)   # Revision version (unsigned char *)
                ]
                self.lib.VSL_GetVersion.restype = None  # void return type

                # Create variables to store the version numbers
                major = ctypes.c_ubyte(0)
                minor = ctypes.c_ubyte(0)
                revision = ctypes.c_ubyte(0)

                # Call the function
                self.lib.VSL_GetVersion(ctypes.byref(major), ctypes.byref(minor), ctypes.byref(revision))

                # Return the version numbers as a dictionary
                return {
                    'major': major.value,
                    'minor': minor.value,
                    'revision': revision.value
                }
            else:
                self.log.error("VSL_GetVersion function not found in the library.")
                raise RuntimeError("VSL_GetVersion function not found in the library.")
        except Exception as e:
            self.log.error(f"Failed to retrieve library version: {e}")
            raise RuntimeError(f"Failed to retrieve library version: {e}")

    def get_product_info(self):
        """
        Retrieve product information for the board.

        Returns:
            dict: A dictionary containing the product information.
                  Example: {
                      'board_name': 'VL-EPU-4012',
                      'attributes': 'Extended Temp',
                      'dios': 8,
                      'timers': 1,
                      'wdt': 1,
                      'ain': 0,
                      'aout': 0,
                      'serial': 2,
                      'fan_support': 1,
                      'bios_info': 'Primary BIOS'
                  }

        Raises:
            RuntimeError: If the VSL_GetProductInfo function is not found in the library.
        """
        try:
            if hasattr(self.lib, 'VSL_GetProductInfo'):
                # Define the function signature
                self.lib.VSL_GetProductInfo.argtypes = [
                    ctypes.c_ulong,  # ProductList (unsigned long)
                    ctypes.POINTER(ctypes.c_char * 32),  # BoardName (char array)
                    ctypes.POINTER(ctypes.c_char * 32),  # Attributes (char array)
                    ctypes.POINTER(ctypes.c_short),  # DIOs (short *)
                    ctypes.POINTER(ctypes.c_short),  # Timers (short *)
                    ctypes.POINTER(ctypes.c_char),  # WDT (char *)
                    ctypes.POINTER(ctypes.c_short),  # AIn (short *)
                    ctypes.POINTER(ctypes.c_short),  # AOut (short *)
                    ctypes.POINTER(ctypes.c_short),  # Serial (short *)
                    ctypes.POINTER(ctypes.c_short),  # FanSupport (short *)
                    ctypes.POINTER(ctypes.c_char * 32)  # BIOSInfo (char array)
                ]
                self.lib.VSL_GetProductInfo.restype = None  # void return type

                # Create variables to store the product information
                product_list = ctypes.c_ulong(0x1)
                board_name = (ctypes.c_char * 32)()
                attributes = (ctypes.c_char * 32)()
                dios = ctypes.c_short(0)
                timers = ctypes.c_short(0)
                wdt = ctypes.c_char(0)
                ain = ctypes.c_short(0)
                aout = ctypes.c_short(0)
                serial = ctypes.c_short(0)
                fan_support = ctypes.c_short(0)
                bios_info = (ctypes.c_char * 32)()

                # Call the function
                self.lib.VSL_GetProductInfo(
                    product_list,
                    ctypes.byref(board_name),
                    ctypes.byref(attributes),
                    ctypes.byref(dios),
                    ctypes.byref(timers),
                    ctypes.byref(wdt),
                    ctypes.byref(ain),
                    ctypes.byref(aout),
                    ctypes.byref(serial),
                    ctypes.byref(fan_support),
                    ctypes.byref(bios_info)
                )

                # Convert the results to Python types
                def decode_byte_array(byte_array):
                    """Helper function to decode byte arrays into strings."""
                    try:
                        # Decode as ASCII, ignoring or replacing invalid characters
                        return byte_array.value.decode('ascii', errors='ignore').strip()
                    except Exception:
                        # If decoding fails, return the raw bytes
                        return byte_array.value

                return {
                    'board_name': decode_byte_array(board_name),
                    'attributes': decode_byte_array(attributes),
                    'dios': dios.value,
                    'timers': timers.value,
                    'wdt': bool(wdt.value),
                    'ain': ain.value,
                    'aout': aout.value,
                    'serial': serial.value,
                    'fan_support': bool(fan_support.value),
                    'bios_info': decode_byte_array(bios_info)
                }
            else:
                self.log.error("VSL_GetProductInfo function not found in the library.")
                raise RuntimeError("VSL_GetProductInfo function not found in the library.")
        except Exception as e:
            self.log.error(f"Failed to retrieve product information: {e}")
            raise RuntimeError(f"Failed to retrieve product information: {e}")

    def set_pin(self, pin_num: int, value: bool):
        """
        Set the state of a GPIO pin.

        Args:
            pin_num (int): The GPIO pin number to control (0-7).
            value (bool): The desired state of the pin (True = HIGH, False = LOW).

        Raises:
            ValueError: If the pin number is invalid.
            RuntimeError: If setting the pin direction or state fails.
        """
        if not 0 <= pin_num <= 7:
            self.log.error(f"Invalid pin number: {pin_num}")
            raise ValueError("Pin number must be between 0 and 7.")

        try:
            # Set the pin direction to output
            if hasattr(self.lib, 'VSL_DIOSetChannelDirection'):
                self.lib.VSL_DIOSetChannelDirection(pin_num, self.DIO_OUTPUT)
                self.log.debug(f"Set GPIO pin {pin_num} direction to OUTPUT.")
            else:
                self.log.error("VSL_DIOSetChannelDirection function not found in the library.")
                raise RuntimeError("VSL_DIOSetChannelDirection function not found in the library.")

            # Set the pin state
            pin_value = self.DIO_CHANNEL_HIGH if value else self.DIO_CHANNEL_LOW
            if hasattr(self.lib, 'VSL_DIOSetChannelLevel'):
                self.lib.VSL_DIOSetChannelLevel(pin_num, pin_value)
                self.log.debug(f"Set GPIO pin {pin_num} value to {'HIGH' if value else 'LOW'}.")
            else:
                self.log.error("VSL_DIOSetChannelLevel function not found in the library.")
                raise RuntimeError("VSL_DIOSetChannelLevel function not found in the library.")
        except Exception as e:
            self.log.error(f"Failed to set GPIO pin {pin_num}: {e}")
            raise RuntimeError(f"Failed to set GPIO pin {pin_num}: {e}")

        level = self.get_pin_level(pin_num)
        self.log.debug(f'  pin: {pin_num} level: {level}')


    def get_pin_level(self, pin_num: int):
        """
        Get the current state (level) of a GPIO pin.

        Args:
            pin_num (int): The GPIO pin number to read (0-7).

        Returns:
            bool: The state of the pin (True = HIGH, False = LOW).

        Raises:
            ValueError: If the pin number is invalid.
            RuntimeError: If reading the pin level fails.
        """
        if not 0 <= pin_num <= 7:
            self.log.error(f"Invalid pin number: {pin_num}")
            raise ValueError("Pin number must be between 0 and 7.")

        try:
            if hasattr(self.lib, 'VSL_DIOGetChannelLevel'):
                # Define the function signature
                self.lib.VSL_DIOGetChannelLevel.argtypes = [ctypes.c_ubyte]  # Channel (unsigned char)
                self.lib.VSL_DIOGetChannelLevel.restype = ctypes.c_ubyte  # Returns unsigned char

                # Call the function
                level = self.lib.VSL_DIOGetChannelLevel(pin_num)
                self.log.debug(f"Read GPIO pin {pin_num} level: {'HIGH' if level == self.DIO_CHANNEL_HIGH else 'LOW'}.")

                # Return the pin state as a boolean
                return level == self.DIO_CHANNEL_HIGH
            else:
                self.log.error("VSL_DIOGetChannelLevel function not found in the library.")
                raise RuntimeError("VSL_DIOGetChannelLevel function not found in the library.")
        except Exception as e:
            self.log.error(f"Failed to read GPIO pin {pin_num} level: {e}")
            raise RuntimeError(f"Failed to read GPIO pin {pin_num} level: {e}")

    def cycle_pin(self, pin: int, delay_ms: int, default=False):
        """
        Cycle the state of a GPIO pin between HIGH and LOW with a specified delay.

        Args:
            pin (int): The GPIO pin number to cycle (0-7).
            delay_ms (int): The delay in milliseconds between state changes.
            default (bool): If False, cycle LOW → HIGH → LOW. If True, cycle HIGH → LOW → HIGH.

        Raises:
            ValueError: If the pin number is invalid or delay_ms is negative.
            RuntimeError: If setting the pin state fails.
        """
        if not 0 <= pin <= 7:
            raise ValueError("Pin number must be between 0 and 7.")
        if delay_ms < 0:
            raise ValueError("Delay must be a non-negative integer.")

        try:
            if default:
                # Cycle HIGH → LOW → HIGH
                self.set_pin(pin, True)  # Set to HIGH
                time.sleep(delay_ms / 1000)  # Convert ms to seconds
                self.set_pin(pin, False)  # Set to LOW
                time.sleep(delay_ms / 1000)  # Convert ms to seconds
                self.set_pin(pin, True)  # Set back to HIGH
            else:
                # Cycle LOW → HIGH → LOW
                self.set_pin(pin, False)  # Set to LOW
                time.sleep(delay_ms / 1000)  # Convert ms to seconds
                self.set_pin(pin, True)  # Set to HIGH
                time.sleep(delay_ms / 1000)  # Convert ms to seconds
                self.set_pin(pin, False)  # Set back to LOW
        except Exception as e:
            raise RuntimeError(f"Failed to cycle GPIO pin {pin}: {e}")

    def cleanup(self):
        """
        Clean up the GPIO subsystem (if required by the API), then closing it.

        Raises:
            RuntimeError: If cleanup fails.
        """
        try:
            if hasattr(self.lib, 'VSL_DIOSetChannelDirection'):
                # Reset all GPIO pins to input mode as a cleanup step
                for pin in range(8):
                    self.lib.VSL_DIOSetChannelDirection(pin, self.DIO_INPUT)
                self.log.info("GPIO subsystem cleaned up successfully.")
            else:
                self.log.error("VSL_DIOSetChannelDirection function not found in the library.")
                raise RuntimeError("VSL_DIOSetChannelDirection function not found in the library.")
        except Exception as e:
            self.log.error(f"Failed to clean up GPIO subsystem: {e}")
            raise RuntimeError(f"Failed to clean up GPIO subsystem: {e}")

        try:
            if hasattr(self.lib, 'VSL_Close'):
                self.lib.VSL_Close.argtypes = []
                self.lib.VSL_Close.restype = ctypes.c_ulong

                result = self.lib.VSL_Close()
                if result != 0:
                    self.log.error(f"Failed to close VersaAPI library. Error code: {result}")
                    raise RuntimeError(f"Failed to close VersaAPI library. Error code: {result}")
                self.log.info("VersaAPI library closed successfully.")
            else:
                self.log.error("VSL_Close function not found in the library.")
                raise RuntimeError("VSL_Close function not found in the library.")
        except Exception as e:
            self.log.error(f"Failed to clean up VersaAPI library: {e}")
            raise RuntimeError(f"Failed to clean up VersaAPI library: {e}")

# Example usage
if __name__ == "__main__":
    # Configure the logger

    logging.basicConfig(
        level=logging.DEBUG,  # Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',  # Define the log format
        handlers=[logging.StreamHandler()]  # Send logs to the console (screen)
    )
    log = getLogger('pueo')
    try:
        # Create an instance of the VersaAPI class
        versa_api = VersaAPI()
        log = versa_api.log
        for i in range(1):
            # Cycle GPIO pin 3 with a delay of 500 ms (LOW → HIGH → LOW)
            log.info("Cycling GPIO pin 3 (LOW → HIGH → LOW)...")
            versa_api.cycle_pin(3, 500, default=False)
            log.info('')
            # Cycle GPIO pin 3 with a delay of 500 ms (HIGH → LOW → HIGH)
            log.info("Cycling GPIO pin 3 (HIGH → LOW → HIGH)...")
            versa_api.cycle_pin(3, 500, default=True)

    except Exception as e:
        log.info(f"An error occurred: {e}")

    finally:
        try:
            # Clean up the GPIO subsystem
            versa_api.cleanup()
        except Exception as e:
            log.info(f"An error occurred during cleanup: {e}")