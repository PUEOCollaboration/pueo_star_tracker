"""
File: messages.py
Description: This module implements the MessageHandler class and associated methods to handle
             messages and files in a queue for socket-based communication. The module includes
             functionality for writing messages/files to a queue, serializing/deserializing
             the queue, and processing messages/files received over a socket connection.

Classes:
    - MessageHandler: Handles message and file operations, including queue management,
                      file recreation, and serialization/deserialization.

Functions:
    - current_timestamp: Utility function to generate formatted timestamps.
    - get_file_metadata: Extracts metadata from a given file.
    - get_file_content: Reads and encodes file content in base64.
    - write: Enqueues messages or files with metadata and timestamps.
    - write_file: Recreates files from metadata and content.
    - get_messages: Converts all queue items to a list of dictionaries.
    - process_messages: Processes serialized queue items to log messages or recreate files.

Usage:
    MessageHandler can be used in scenarios where messages and file data need to be queued,
    serialized for transmission, and deserialized for processing.

Author: Milan Stubljar <info@stubljar.com>
Created: 2024-12-09
Version: 1.0
"""

# Standard imports
import os
import base64
from datetime import datetime
from threading import Lock
from queue import Queue
import logging

# Custom imports
from lib.common import current_timestamp


class MessageHandler:
    """
    MessageHandler provides functionality to write messages or files
    into a message queue. It includes methods to extract file metadata
    and content, reverse-save files from metadata/content, and handle
    messages or file data to be sent over a socket connection.

    Attributes:
        message_queue (Queue): Queue to store messages or file data.
        flight_queue (Queue): Queue to store Solutions Telemetry data.
    """

    def __init__(self, fq_max_size=12):
        self.fq_max_size = fq_max_size
        self.file_lock = Lock()
        self.message_queue = Queue()                        # Message Queue for GUI
        self.flight_queue = Queue(maxsize=self.fq_max_size) # Message Queue for Flight Computer (Solutions Telemetry)
        self.timestamp_fmt = '%Y-%m-%d %H:%M:%S.%f'
        self.log = logging.getLogger('pueo')

        self.dst_path = 'images'   # Destination path for new files created by process_messages (client/gui)
        self.gui = None

    def get_file_metadata(self, filename, dst_filename=None):
        """
        Extract metadata of a file.

        Args:
            filename (str): The name of the file.
            dst_filename (str): The name of the metadata target file.

        Returns:
            dict: Metadata of the file including filename, size, type, and modification time.

        """
        if not os.path.exists(filename):
            self.log.warning(f"File '{filename}' does not exist.")
            raise FileNotFoundError(f"File '{filename}' does not exist.")

        stat_info = os.stat(filename)
        dst_filename = filename if dst_filename is None else dst_filename
        metadata = {
            'filename': os.path.basename(dst_filename),
            'size': stat_info.st_size,
            'file_type': filename.split('.')[-1] if '.' in filename else 'unknown',
            'modification_time': datetime.fromtimestamp(stat_info.st_mtime).strftime('%Y-%m-%d %H:%M:%S.%f')
        }
        self.log.debug(f"metadata: {metadata}")
        return metadata

    def get_file_content(self, filename):
        """
        Extract content of a file.

        Args:
            filename (str): The name of the file.

        Returns:
            str: Base64 encoded content of the file.
        """
        if not os.path.exists(filename):
            self.log.warning(f"File '{filename}' does not exist.")
            raise FileNotFoundError(f"File '{filename}' does not exist.")

        with open(filename, 'rb') as file:
            content = base64.b64encode(file.read()).decode('utf-8')
        return content

    def write(self, item, level='info', data_type='message', dst_filename=None):
        """
        Write saves the item to the message queue.

        It handles different types such as 'message' or 'file'.
        It saves the data as a dictionary to be sent over a socket
        connection to the client.

        Args:
            item (str): The item to write, either a message or a filename.
            level (str): Message level: info, warning, error, debug.
            data_type (str): The type of the data ('message' or 'file').
            dst_filename (str): Filename of the destination file.
        """
        data_types = ['message', 'info_file', 'image_file']
        as_file = ''
        if data_type not in data_types:
            raise ValueError(f'Unknown data_type: {data_type} allowed types: {data_types}')
        data_item = {'data_type': data_type}
        if data_type == 'message':
            data_item['data'] = {
                'timestamp': current_timestamp(self.timestamp_fmt),
                'level': level,
                'data_type': data_type,
                'message': str(item)
            }
        elif data_type in ['image_file', 'info_file']:
            # Metadata for info_file must have a name same as the image
            file_metadata = self.get_file_metadata(filename=item, dst_filename=dst_filename)
            file_content = self.get_file_content(filename=item)
            data_item['data'] = {
                'timestamp': current_timestamp(self.timestamp_fmt),
                'data_type': data_type,
                'metadata': file_metadata,
                'content': file_content
            }
            as_file = '' if dst_filename is None else f' as {dst_filename}'

        self.log.debug(f'{data_type}: {item}{as_file}')
        self.message_queue.put(data_item)

    def write_file(self, dst_path, metadata, content):
        """
        Recreate a file from metadata and content.

        Args:
            metadata (dict): Metadata of the file.
            content (str): Base64 encoded content of the file.

        Returns:
            str: Path of the recreated file.
        """
        filename = metadata.get('filename', 'unknown_file')
        file_path = os.path.join(dst_path, filename)

        self.log.debug(f'New file: {metadata}')
        # Decode the content and save it to the file,  acquire a lock first
        with self.file_lock, open(file_path, 'wb') as file:
            file.write(base64.b64decode(content))

        # Set file modification time if available
        # TODO: We do not want to update this timestamp as GUI relies on last modified timestamp.
        if 'modification_time' in metadata and False:
            mod_time = datetime.strptime(metadata['modification_time'], '%Y-%m-%d %H:%M:%S.%f').timestamp()
            with self.file_lock:
                os.utime(file_path, (mod_time, mod_time))

        return file_path

    def get_messages(self) -> list:
        """
        Serialize all items in the message queue to a list of dictionaries.

        Returns:
            list: A list of dictionaries representing the queued items.
        """
        serialized_data = []
        while not self.message_queue.empty():
            item = self.message_queue.get()
            serialized_data.append(item)
        if serialized_data:
            self.log.debug(f'Serialising messages: {len(serialized_data)}')
        return serialized_data

    def process_messages(self, serialized_data: list):
        """
        Process a list of serialized queue items, logging messages and recreating files.

        Args:
            serialized_data (list): List of dictionaries representing serialized queue items.
        """
        self.log.debug(f'Processing messages: {len(serialized_data)}')
        for item in serialized_data:
            data_type = item.get('data_type')
            data = item.get('data', {})
            timestamp = data.get('timestamp', 'Unknown time')

            self.log.debug(f'Processing message: {timestamp} {data_type}')

            if data_type == 'message':
                message = data.get('message', 'Unknown message')
                level = data.get('level', 'info')
                log_message = f"{timestamp[:-3]} {level.upper():<7s} {message}"
                if self.gui is not None:
                    self.gui.add_log_line(log_message, level)
                self.log.debug(log_message)
            elif data_type in ['image_file', 'info_file']:
                metadata = data.get('metadata', {})
                content = data.get('content', '')
                try:
                    file_path = self.write_file(self.dst_path, metadata, content)
                    if self.gui is not None:
                        filename = metadata['filename']
                        size = (metadata['size'])/1024
                        data_type_txt = data_type.replace('_', ' ')
                        log_message = f"{timestamp[:-3]} {'INFO':<7s} New {data_type_txt}: {filename} [{size:.1f} KB]"
                        self.gui.add_log_line(log_message, 'info')
                    self.log.debug(f"[{timestamp}] File recreated at: {file_path}")
                except Exception as e:
                    self.log.error(f"[{timestamp}] Failed to recreate file: {e}")

    def save_flight_telemetry(self, ):
        """

        """


        pass

    # Last line
