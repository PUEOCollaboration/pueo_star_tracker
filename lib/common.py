# ------------------------------------------------------------------------------
# File: common.py
#
# Description: This module provides a collection of utility functions and
#              helpers to support various tasks in the EPN data import
#              process. Key functions include:
#
#              - init_logging: Initializes the logging system for the
#                application, setting up log files and formats for
#                consistent and structured logging.
#
#              - log_debug: A function for logging debug messages that can
#                be useful for tracing execution flow and diagnosing issues
#                during development.
#
#              - log_error: A function for logging error messages to
#                facilitate troubleshooting and error tracking.
#
#              - logit: A general-purpose logging function that outputs
#                messages with varying severity levels and formats.
#
#              - logpair: A utility function to log key-value pairs for
#                better readability in logs, aiding in understanding the
#                state of the application.
#
#              - get_dt: A function to retrieve the current date and time,
#                formatted for logging or display purposes, enhancing
#                consistency in date-time representation throughout the
#                application.
#
#              - logging_add_second_file: This function allows for the
#                addition of a secondary log file to the logging setup,
#                enabling more granular logging and separation of log
#                entries based on context or module.
#
#              - load_config: Loads the configuration settings from the
#                specified configuration file (defaulting to 'config.ini')
#                into a global SETTINGS dictionary. It initializes the logging
#                system based on the settings, creating necessary log directories
#                and setting log file paths and levels. If the configuration
#                file does not exist, the function logs an error and exits the
#                application.
#
#              This module is integral to maintaining consistent logging
#              practices and providing helper functions that streamline
#              other operations within the application.
#
# Created: 2024-10-25
#
# Author: Milan Stubljar, Stubljar d.o.o. <info@stubljar.com>
# ------------------------------------------------------------------------------

# Standard Imports
from collections import OrderedDict
import queue
from queue import Queue
from typing import Any, Optional
import logging
import logging.handlers
import traceback
import datetime
from datetime import timezone
import inspect
import os
import sys
import csv
import re
import time
import configparser
from pathlib import Path
import math
from collections import deque
from contextlib import suppress
import threading
import ctypes
import json
import zipfile
import fnmatch
import shutil

# External modules
from termcolor import cprint, colored

# import prettytable as pt
# from prettytable.colortable import ColorTable, Themes
# import urllib.request
import numpy as np
from tqdm import tqdm

# log = logging.getLogger(__name__)
log = logging.getLogger('pueo')

CONFIG_FILE_NAME = 'config.ini'
CONFIG = OrderedDict()
SETTINGS = OrderedDict()

## Logging Wrapper
LOG_LEVEL_NOTSET   = 6
LOG_LEVEL_CRITICAL = 1
LOG_LEVEL_ERROR    = 2
LOG_LEVEL_WARNING  = 3
LOG_LEVEL_INFO     = 4
LOG_LEVEL_DEBUG    = 5
LOG_LEVEL_TRACE    = 6

LOG_WRAPPER_NAME_DICT = dict.fromkeys(['log_msg', 'logit', 'wprint', 'eprint', 'sprint', 'logpair',
                         'log_debug', 'log_info', 'log_error', 'log_warning', 'log_trace', 'log_critical'], 1)
# Removed init from the end: , '__init__'


# Unbuffered write to output
class Unbuffered(object):

    def __init__(self, stream):
        self.stream = stream

    def write(self, data):
        self.stream.write(data)
        self.stream.flush()

    def writelines(self, datas):
        self.stream.writelines(datas)
        self.stream.flush()

    def __getattr__(self, attr):
        return getattr(self.stream, attr)


def log_msg(log_level=LOG_LEVEL_DEBUG, *msg_list, **kwargs):
    with suppress(PermissionError):
        if log is None:
            return
        for msg in msg_list:
            try:
                if log_level == LOG_LEVEL_CRITICAL:
                    log.critical(msg)
                elif log_level == LOG_LEVEL_ERROR:
                    log.error(msg)
                elif log_level == LOG_LEVEL_WARNING:
                    log.warning(msg)
                elif log_level == LOG_LEVEL_INFO:
                    log.info(msg)
                elif log_level == LOG_LEVEL_DEBUG:
                    log.debug(msg)
                elif log_level == LOG_LEVEL_TRACE:
                    log.trace(msg)
            except AttributeError:
                pass


def log_debug(msg):
    log_msg(LOG_LEVEL_DEBUG, msg)


def log_info(msg):
    log_msg(LOG_LEVEL_INFO, msg)


def log_error(msg):
    log_msg(LOG_LEVEL_ERROR, msg)


def log_warning(msg):
    log_msg(LOG_LEVEL_WARNING, msg)


def log_critical(msg):
    log_msg(LOG_LEVEL_CRITICAL, msg)


def log_trace(msg):
    log_msg(LOG_LEVEL_TRACE, msg)


def init_common(logger):
    global log
    log = logger
    log_debug(f'Init common.py')


def sprint(*args, **kwargs):
    print(*args, file=sys.stdout, **kwargs)
    if log is not None and len(args) > 0 and args[0] is not None:
        log_info(args[0])


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)
    if log is not None and len(args) > 0 and args[0] is not None:
        log_error(args[0])


def wprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)
    if log is not None and len(args) > 0 and args[0] is not None:
        log_warning(args[0])


def logit(msg, attrs=None, color=None, is_tqdm=False, end='\n'):
    # Use TQDM to write to console.
    is_tqdm = True
    if color is None:
        if is_tqdm:
            with suppress(TypeError):
                tqdm.write(msg, end=end)
        else:
            print(msg, end=end)
    else:
        if is_tqdm:
            tqdm.write(colored(msg, attrs=attrs, color=color), end=end)
        else:
            cprint(msg, attrs=attrs, color=color, end=end)
    log_info(msg)


def logline(length=30, prefix='  '):
    """Note: length was 22"""
    print(f'{prefix}{"-" * length}')


def logpair(key: object, val: object = '', prefix: object = '  ', r_offset: object = 30, attrs: object = None, color: object = None) -> object:
    # logit(f'  {"total iterations:":<{R_OFFSET}s} {iteration_total}')
    key = f'{key}:'
    logit(f'{prefix}{key:{r_offset}s} {val}', attrs=attrs, color=color)


def timer(function, *args, **kwargs):
    start = time.time()
    ret = function(*args, **kwargs)
    dt = time.time() - start
    log_info(f'  {function.__name__} completed in {dt:.5f}s')
    return ret


def profile(function, *args, **kwargs):
    start = time.time()
    ret = function(*args, **kwargs)
    dt = time.time() - start
    return ret, dt


def get_val_by_key(_dict, key1, key2=None):
    if key1 in _dict:
        return _dict[key1]
    elif key2 is not None and key2 in _dict:
        return _dict[key2]
    else:
        return None


def save_report(filename, msg):
    if len(msg) < 256:
        log_debug(f'  Adding: {filename}: {msg}')
    else:
        log_debug(f'  Adding: {filename}: {len(msg)} chars.')
    try:
        with open(filename, 'a', encoding='utf-8') as f:
            f.write(f'{msg}\n')
    except Exception as e:
        log_error(f'Save log_error: {e}')


def save(data, filename, description, index=True, target='csv'):
    logit(f'  Saving {description}:')
    logit(f'    Creating Pandas DataFrame: {description}: {len(data)}')
    if description == 'attributes-pivot':
        df = pd.DataFrame(data).apply(pd.Series)
    else:
        df = pd.DataFrame(data).apply(pd.Series)
    logit(f'    Saving to csv: {filename}.{target}')
    if target == 'csv':
        if index:
            df.to_csv(f'{filename}.csv', encoding='utf-8-sig', sep=',', quotechar='"', quoting=csv.QUOTE_ALL)
        else:
            df.to_csv(f'{filename}.csv', encoding='utf-8-sig', sep=',', quotechar='"', quoting=csv.QUOTE_ALL, index=False)
    elif target == 'xlsx':
        # Excel takes long time and does not generate numbers, but all is strings...
        # logit(f'  Saving to xlsx: {filename}.xlsx')
        if index:
            df.to_excel(f'{filename}.{target}', encoding='utf-8')
        else:
            df.to_excel(f'{filename}.{target}', encoding='utf-8', index=False)


def save_to_excel(results, filename='../logs/results.xlsx'):
    """

    Example:
        from lib.common import save_to_excel, read_json
        results = read_json('logs/results.json')
        save_to_excel(results, 'logs/results.xlsx')
    """
    rows = []
    for file, solvers in results.items():
        basename = os.path.basename(file)
        row = {"file": basename}
        for solver_name, solver_data in solvers.items():
            solver_name = solver_name.replace('solver', 's')
            for k, v in solver_data.items():
                if k in ['astrometry']:
                    continue
                if v and k and k.endswith('_shape'):
                    row[f"{solver_name}.{k}"] = 'x'.join(str(x) for x in v)
                    continue
                row[f"{solver_name}.{k}"] = v

            row[f"{solver_name}.runtime"] = solver_data.get("runtime")

            astrometry = solver_data.get("astrometry", {})
            row[f"{solver_name}.astrometry"] = astrometry
            if astrometry:
                for key, value in astrometry.items():
                    row[f"{solver_name}.astrometry.{key}"] = value
        rows.append(row)

    # Create a DataFrame
    import pandas as pd
    df = pd.DataFrame(rows)

    # Remove unnecessary columns (e.g., full astrometry dictionaries)
    df = df.drop(columns=[col for col in df.columns if col.endswith(".astrometry")], errors="ignore")

    # Save to Excel
    print(f'Saving to {filename}')
    df.to_excel(filename, index=False)


def _convert_to_json_serializable(obj):
    """
    Recursively converts NumPy arrays, tuples, sets, and NumPy scalar types to JSON-serializable formats.
    Also converts NaN, Infinity, and -Infinity values to None.

    Args:
      obj: The object to be converted.

    Returns:
      The converted object, with NumPy arrays, tuples, sets, and NumPy scalar types replaced by JSON-compatible types.
      NaN, Infinity, and -Infinity values are converted to None.
    """
    # Handle floating point numbers (both native and numpy)
    if isinstance(obj, (float, np.floating)):
        if math.isnan(obj) or np.isnan(obj):
            return None
        if math.isinf(obj) or np.isinf(obj):
            return None
        return obj  # Return valid float as-is

    # Handle NumPy scalar types
    if isinstance(obj, (np.generic, np.number)):
        converted = obj.item()  # Convert to native Python scalar
        # Recursively process the converted value to handle potential NaN/Infinity
        return _convert_to_json_serializable(converted)

    # Handle complex data structures
    if isinstance(obj, np.ndarray):
        return [_convert_to_json_serializable(item) for item in obj.tolist()]
    if isinstance(obj, tuple):
        return [_convert_to_json_serializable(item) for item in obj]
    if isinstance(obj, set):
        return [_convert_to_json_serializable(item) for item in obj]
    if isinstance(obj, dict):
        return {k: _convert_to_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_convert_to_json_serializable(item) for item in obj]

    # Return other types as-is (strings, integers, booleans, None, etc.)
    return obj


def save_to_json(results, filename="logs/results.json"):
    """
    Saves the given results dictionary to a JSON file.

    Args:
      results: The dictionary containing the results, potentially with nested structures.
      filename: The path to the JSON file.
    """
    try:
        json_serializable_results = _convert_to_json_serializable(results)
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(json_serializable_results, f, indent=4, ensure_ascii=False)
        print(f"Results saved successfully to {filename}")
    except (IOError, TypeError, Exception) as e:
        cprint(f"Error saving JSON file {filename}: {e}", 'red')
        print(results)


def read_json(file_path):
    """
    Reads a JSON file and returns its content as a Python dictionary.

    Args:
      file_path: Path to the JSON file.

    Returns:
      A Python dictionary containing the data from the JSON file.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"Error: JSON file not found at {file_path}")
        return None
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        return None

def logging_add_second_file(filename):
    """Add secondary log file, to keep track e.g. in history folder having log record."""
    logger = logging.getLogger()  # Root logger

    # Create a handler for the new log file
    additional_file_handler = logging.FileHandler(filename)
    additional_file_handler.setLevel(logging.DEBUG)  # Set the log level for this file

    # Use the same format as the current logger (if needed)
    # Iterate over the handlers and retrieve the formatter from the first handler
    for handler in logger.handlers:
        existing_formatter = handler.formatter
        if existing_formatter:
            # log.debug("Existing formatter found:")
            # log.debug(existing_formatter._fmt)  # Access the formatter string
            break
        else:
            pass
            # log.debug("No formatter is set for this handler.")

    additional_file_handler.setFormatter(existing_formatter)

    # Add the new handler to the logger
    logger.addHandler(additional_file_handler)
    # Now logging will go to both the original file and 'additional.log'
    log_debug('--------------LOGGING TO SECOND FILE--------------')
    log_debug(f'log_file: {filename}')

def init_logging(name=None, log_file='logs/debug.log', log_level='DEBUG', is_global=True):
    """ Initialise logging

    :param name:
    :param log_file:
    :param log_level:
    :return: None
    :param is_global:
    :type is_global:
    """
    # Logging is_init and configuration
    name = __name__ if name is None else name
    if isinstance(name, str):
        _log = logging.getLogger(name)  # Get ROOT Logger
    else:
        _log = name

    # Prevent logging propagation to the root logger
    with suppress(AttributeError):
        _log.propagate = False

    __DEBUG_TRACE_NUM__ = 5
    logging.addLevelName(__DEBUG_TRACE_NUM__, "TRACE")
    logging.TRACE = __DEBUG_TRACE_NUM__

    ## Custom log_trace logger function
    #
    #  Defines log_trace
    def custom_trace(self, message, *args, **kws):
        if self.isEnabledFor(__DEBUG_TRACE_NUM__):
            # Yes, logger takes its '*args' as 'args'.
            self._log(__DEBUG_TRACE_NUM__, message, args, **kws)

    logging.Logger.trace = custom_trace
    _log.setLevel(log_level)

    class MyFormatter(logging.Formatter):
        converter = datetime.datetime.fromtimestamp

        def formatTime(self, record, datefmt=None):
            ct = self.converter(record.created)
            if datefmt:
                s = ct.strftime(datefmt)
            else:
                t = ct.strftime("%Y-%m-%d %H:%M:%S")
                # s = "%s,%03d" % (t, msecs)  # Get microseconds
                s = "%s.%03d" % (t, record.msecs)  # Get milliseconds
            return s

    if hasattr(sys, '_getframe'):
        currentframe = lambda: sys._getframe(3)
    else:
        sys.exit(-1)
    # done filching

    #
    # _srcfile is used when walking the stack to check when we've got the first
    # caller stack frame.
    #
    _srcfile = os.path.normcase(currentframe.__code__.co_filename)

    def find_caller_patch(self, stack_info=False, stacklevel=1):
        """
        Find the stack frame of the caller so that we can note the source
        file name, line number and function name.
        """
        f = currentframe()
        # On some versions of IronPython, currentframe() returns None if
        # IronPython isn't run with -X:Frames.
         #back_idx = 2
        # for _ in range(back_idx):

        if f is not None:
            f = f.f_back

        # If the log_X is called from any of the WRAPPERs, go one level up
        def clean(s):
            if s is None or not isinstance(s, str):
                return s
            s_clean = s[7:] if f'{s}'.startswith('frame: ') else s
            return s_clean

        while f is not None and clean(f.f_code.co_name) in LOG_WRAPPER_NAME_DICT:
            f = f.f_back

        rv = "(unknown file)", 0, "(unknown function)"
        if False:
            while hasattr(f, "f_code"):
                co = f.f_code
                filename = os.path.normcase(co.co_filename)
                if filename == _srcfile:
                    f = f.f_back
                    continue
                rv = (co.co_filename, f.f_lineno, co.co_name, None)
                break
        else:
            co = f.f_code
            rv = (co.co_filename, f.f_lineno, co.co_name, None)
        return rv

    # create rotating file handler which logs even log_debug messages
    # 16 x 1024 x 1024 (16 Mb)
    file_size = 16 * 1024 * 1024  # 16Mb

    fh = logging.handlers.RotatingFileHandler(log_file, 'a', maxBytes=file_size, backupCount=10, encoding='utf-8', delay=False)
    fh.setLevel(log_level)
    # formatter = logging.Formatter('%(asctime)s ' + mode + ' %(levelname)s %(message)s')
    pid = os.getpid()
    # level = '[{}]'.format(logger.LogRecord)
    # FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
    formatter = MyFormatter(u'%(asctime)s {} %(lineno)5s:%(funcName)25s %(levelname)7s %(message)s'.format(pid))
    fh.setFormatter(formatter)

    # Update Caller (the funcName when wrapper)
    _log.findCaller = find_caller_patch

    # add the handlers to the logger
    if not _log.hasHandlers() or True:
        _log.addHandler(fh)

    if is_global:
        log = _log
        log_info(f'{"-" * 10}  LOGGING STARTED  {"-" * 10}')
        log_trace('LOG INITIALISED caller data: {}'.format(inspect.stack()[1]))
    else:
        _log.info(f'{"-" * 10}  LOGGING STARTED  {"-" * 10}')
    return _log


def find_files(path, reg_exp):
    """ Return a list of files in a folder at path filtered by reg_exp

    :param path:
    :param reg_ex:
    :return:
    """
    return [os.path.join(path, file) for file in os.listdir(path) if re.search(reg_exp, file)]


def get_file_size(filename):
    """
    Returns the size of a file in bytes.

    Parameters:
    filename (str): The path to the file.

    Returns:
    int: The size of the file in bytes. If the file does not exist, returns -1.
    """
    if filename is not None and os.path.exists(filename):
        return os.path.getsize(filename)
    else:
        return -1


def config_parse_value(data_type, config, section, token, default, allowed_value_list=[]):
    if data_type == 'bool':
        try:
            value = f"{config[section][token]}".lower().strip() in ['true', 'yes', '1']
        except (KeyError, ValueError):
            value = default
    elif data_type == 'int':
        try:
            value = int(config[section][token])
        except (KeyError, ValueError):
            value = default
    elif data_type == 'float':
        try:
            value = float(config[section][token])
        except (KeyError, ValueError):
            value = default
    elif data_type == 'str':
        try:
            value = str(config[section][token])
        except KeyError:
            value = default
    elif data_type == 'filename':
        try:
            value = str(config[section][token])
            value = os.path.abspath(value)
        except KeyError:
            value = default
    elif data_type == 'list':
        try:
            value = str(config[section][token])
            value = f'{value}'.split(',')
        except KeyError:
            value = default
    else:
        raise ValueError(f'Unsupported data type: {data_type}.')
    # Check if allowed_value_list
    if len(allowed_value_list) > 0:
        if value not in allowed_value_list:
            delimiter = ','
            eprint(f'Warning: invalid value for {section}/{token}: {value} expected one of: {delimiter.join(allowed_value_list)}')

    return value


def load_config(name=None, config_file='config.ini', log_file_name_token='log_file_name', is_global=True):
    """Load CONFIG file and populate SETTINGS dict

    :return: None
    """
    # global log, CONFIG, XML_FILE, ATTRIBUTES_PIVOT, ATTRIBUTES_PIVOT_FILENAME, SMART_COLLECTION_FILENAME, OVER_LIMIT_FILENAME, SHOPIFY_FILENAME, SHOPIFY_SETTINGS, SHOPIFY_DATA_FILES

    cwd = os.getcwd()
    config_file_name = config_file
    if os.path.isabs(config_file_name):
        file_to_open = Path(os.path.join(config_file))
    else:
        file_to_open = Path(os.path.join(cwd, 'conf', config_file))
    if file_to_open.exists():
        # logpair('config file', str(file_to_open))
        config = configparser.ConfigParser()
        config.read(file_to_open, encoding='utf-8')
        for key, value in config.items():
            CONFIG[key] = value
    else:
        eprint(f'Error loading CONFIG file: {file_to_open}')
        sys.exit(1)

    # First, lets read the CONFIG section
    is_logging_enabled = config_parse_value('bool', CONFIG, 'LOG', 'enabled', False)
    log_path = SETTINGS['log_path'] = config_parse_value('str', CONFIG, 'LOG', 'path', 'logs')
    log_file = config_parse_value('str', CONFIG, 'LOG', log_file_name_token, 'debug.log')
    log_level = config_parse_value('str', CONFIG, 'LOG', 'level', 'DEBUG')

    # Make
    if is_logging_enabled:
        try:
            os.makedirs(log_path)
        except FileExistsError:
            pass

        log_filename = os.path.abspath(f'{Path(cwd)}/{log_path}/{log_file}')

        # Initiate logging
        log = init_logging(name, log_filename, log_level, is_global)
        # logpair('log file', log_filename)
        # logline()
    else:
        log = logging.getLogger(__name__)

    return log


# Print iterations progress
def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█'):
    """
    Call in a loop to create terminal progress bar
    @params_dict:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    #print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end='\r')
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end='')
    # Print New Line on Complete
    if iteration == total:
        print()


def import_csv_file(metadata):
    """ Import CSV file into Ordered Dictionary

    :param metadata: dict with file metadata (filename, header, delimiter, description...
    :param csv_header:  csv header dict, if omitted or empty autogenerated from first col (header) of the filename
    :return: OrderedDict
    """
    # Import CSV File

    csv_file = metadata['file']
    delimiter = metadata['delimiter']
    description = metadata['description']
    if 'encoding' in metadata:
        encoding = metadata['encoding']
    else:
        encoding = 'utf-8-sig'

    logit(f'  Importing {description} file: {csv_file}')

    if 'header' not in metadata or metadata['header'] is None or metadata['header'] == '' or len(metadata['header']) == 0:
        log_debug(f'  Auto generating header:')
        with open(csv_file, 'r', encoding=encoding) as f:
            header_line = f.readline()
            csv_header = header_line.split(delimiter)
            csv_header = [x.strip() for x in csv_header]
            log_debug(f'    Cols [{len(csv_header)}]: {csv_header}')
    else:
        csv_header = metadata['header']

    csv_dict = OrderedDict()
    with open(csv_file, 'r', encoding=encoding) as f:
        line_cnt = 0
        try:
            line = None
            _dict = dict()
            for _dict in csv.DictReader(f, fieldnames=csv_header, delimiter=delimiter):
                line_cnt += 1
                id = _dict['id']
                # Skip header line
                if id == 'id':
                    continue
                log_trace(f'{line_cnt}: {_dict}')
                if id in csv_dict:
                    ValueError(f'Duplicate id found at line: {line_cnt}: {_dict}')
                else:
                    csv_dict[id] = _dict
            #else:
            #    line = True
            #    while line:
            #        line = f.readline()
            #        log_debug(f'{line}')
        except UnicodeDecodeError as e:
            if _dict is not None:
                error = f'UnicodeDecodeError at line: {line_cnt}: {_dict}\n{e}'
            if line is not None:
                error = f'UnicodeDecodeError at line: {line_cnt}: {line}\n{e}'
            ValueError(error)
            raise UnicodeDecodeError(e)
    logit(f'    Items loaded: {len(csv_dict)}')
    return csv_dict


def import_xlsx_file(metadata):
    """ Import XLSX file into Ordered Dictionary

    :param metadata: dict with file metadata (filename, header, delimiter, description...
    :return: OrderedDict
    """
    # Import XLSX File

    file = metadata['file']
    if 'delimiter' in metadata:
        delimiter = metadata['delimiter']
    else:
        delimiter = ';'
    description = metadata['description']
    if 'encoding' in metadata:
        encoding = metadata['encoding']
    else:
        encoding = 'utf-8-sig'

    logit(f'Importing {description} file: {file}')

    df = pd.read_excel(file, dtype=str, encoding=encoding)
    df_dict = df.to_dict()
    # log_debug(df_dict)
    # Convert to our dict
    _dict = OrderedDict()
    _len = len(df_dict['id'])
    for idx in range(0, _len):
        item = OrderedDict()
        for key in df_dict.keys():
            value = df_dict[key][idx]
            if value is None or pd.isna(value):
                value = ''
            value = re.sub(r'\n', r'\\n', str(value))
            # Creazy Smart Workaround for Excel Rounding
            if re.search(r'\d*\.\d*99999999$', value):
                log_debug(value)
                try:
                    value_float = float(value)
                    value = re.sub(r'9+$', r'', value)
                    decimals = len(re.sub(r'\d*\.', r'', value))
                    value = round(value_float, decimals)
                    log_debug(value)
                except:
                    pass
            item[key] = f'{value}'
        _dict[idx] = item

    log_debug(f'Converted.')
    return _dict


def list2dict(_list, hash_tag):
    _dict = OrderedDict()
    for item in _list:
        hash = item[hash_tag]
        if hash in _dict:
            raise KeyError(f'Duplicate key value for: {hash}')
        _dict[hash] = item
    return _dict


def load_data(db, metadata):
    """Load data from CSV file or Database depending on the source value of the metadata
    Initialy data is read from CSV, later from DB...

    :param db:
    :param metadata:
    :return:
    """
    if 'source' in metadata:
        source = metadata['source']
    else:
        source = 'file'

    if source == 'file':
        _dict = import_csv_file(metadata)
    elif source == 'database':
        sql = metadata['select_sql']
        entity_list = db.query_many(sql, [()])
        _dict = list2dict(entity_list, 'cat_id')
    elif source == 'xlsx':
        _dict = import_xlsx_file(metadata)
    return _dict


def get_float(value, default_value, name=''):
    """Convert string to value, if log_error, set default value

    :param value: <string>
    :param default_value:
    :param name: variable name [optional]
    :return:
    """
    try:
        value = float(value)
    except ValueError as e:
        log_warning(f'  Setting default value [{name}]: {default_value} <-- {value}')
        value = default_value
    return value


def time_display(sec: float):
    """ Convert seconds to more reasonable units of time such as years."""
    minute = 60
    hour = 3600
    day = 84600
    month = 30 * day
    year = 365 * day
    millennium = 1000 * year
    million = 1000000 * year
    billion = 1000 * million
    trillion = 1000 * billion
    quadrillion = 1000 * trillion
    inf = 1000 * quadrillion

    if sec <= minute:
        return '{0} seconds'.format(sec)
    if sec <= hour:
        return '{0:0.2f} minutes'.format(sec / minute)
    if sec <= day:
        return '{0:0.2f} hours'.format(sec / hour)
    if sec <= month:
        return '{0:0.2f} days'.format(sec / day)
    if sec <= year:
        return '{0:0.2f} months'.format(sec / month)
    if sec <= millennium:
        return '{0:0.2f} years'.format(sec / year)
    if sec <= million:
        return '{0:0.2f} millennia'.format(sec / millennium)
    if sec <= billion:
        return '{0:0.2f} million years'.format(sec / million)
    if sec <= trillion:
        return '{0:0.2f} billion years'.format(sec / billion)
    if sec <= quadrillion:
        return '{0:0.2f} trillion years'.format(sec / trillion)
    if sec <= inf:
        return '{0:0.2f} quadrillion years'.format(sec / quadrillion)
    else:
        return "Long, really long, likely not even by the grand opening of the Restaurant at the End of the Universe. That long."


def format_timedelta(value, time_format="{days} days, {hours2}:{minutes2}:{seconds2}"):

    if hasattr(value, 'seconds'):
        seconds = value.seconds + value.days * 24 * 3600
    else:
        seconds = int(value)

    seconds_total = seconds

    minutes = int(math.floor(seconds / 60))
    minutes_total = minutes
    seconds -= minutes * 60

    hours = int(math.floor(minutes / 60))
    hours_total = hours
    minutes -= hours * 60

    days = int(math.floor(hours / 24))
    days_total = days
    hours -= days * 24

    years = int(math.floor(days / 365))
    years_total = years
    days -= years * 365

    return time_format.format(**{
        'seconds': seconds,
        'seconds2': str(seconds).zfill(2),
        'minutes': minutes,
        'minutes2': str(minutes).zfill(2),
        'hours': hours,
        'hours2': str(hours).zfill(2),
        'days': days,
        'years': years,
        'seconds_total': seconds_total,
        'minutes_total': minutes_total,
        'hours_total': hours_total,
        'days_total': days_total,
        'years_total': years_total,
    })


def timedelta2interval(value):

    if hasattr(value, 'seconds'):
        seconds_raw = seconds = int(round(value.seconds + value.days * 24 * 3600 + value.microseconds/1000000))
    else:
        seconds_raw = seconds = int(round(value))

    seconds_total = seconds

    minutes = int(math.floor(seconds / 60))
    minutes_total = minutes
    seconds -= minutes * 60

    hours = int(math.floor(minutes / 60))
    hours_total = hours
    minutes -= hours * 60

    days = int(math.floor(hours / 24))
    days_total = days
    hours -= days * 24

    years = int(math.floor(days / 365))
    years_total = years
    days -= years * 365

    if seconds_raw == 1:
        return f'1 second'
    elif seconds_raw < 60:
        return f'{seconds} seconds'
    elif seconds_raw < 3600:
        return f'{minutes} minutes'
    elif seconds_raw == 3600:
        return f'1 hour'
    elif seconds_raw < 3600*24:
        return f'{hours} hours'
    elif days == 1:
        return f'{days} day'
    elif days == 7:
        return f'1 week'
    else:
        return f'1 month'


def get_age(datetime_str, format='%Y-%m-%d %H:%M:%S'):
    t1 = datetime.datetime.strptime(datetime_str, format)
    t2 = datetime.datetime.now()
    dt = abs(t2 - t1)
    return dt


def get_date(format='%Y-%m-%d'):
    return datetime.datetime.now().strftime(format)

def get_utc_datetime():
    # Getting the current date
    # and time
    dt = datetime.datetime.now(timezone.utc)

    utc_time = dt.replace(tzinfo=timezone.utc)
    # utc_timestamp = utc_time.timestamp()
    return utc_time

def get_traceback(error):
    lines = traceback.format_exception(type(error), error, error.__traceback__)
    return ''.join(lines)


class CandlePP:
    """Class to Nicely Print out the Candles Data"""
    row_cnt = 0
    default_ohlc_hdr = ['Open Time', 'Close Time', 'Symbol', 'Open', 'High', 'Low', 'Close', 'Volume', 'Final', 'Buy ', 'Sell']

    hdr_reprint = 30
    is_hdr = False

    title = None
    _is_preloaded = False

    def __init__(self, header: list = None, title=None):

        self.header: list = header if header is not None else self.default_ohlc_hdr
        self.title = title
        self.table = pt.PrettyTable(self.header)
        # self.table = ColorTable(self.header, theme=Themes.DEFAULT)
        self.table.add_autoindex(fieldname='Id')

        # Check if ANSI Colors are disabled
        ansi_colors_disabled = os.getenv('ANSI_COLORS_DISABLED')
        log.info(f'ENV: ANSI_COLORS_DISABLED: {ansi_colors_disabled}')

    def set_preloaded(self, is_preloaded=True):
        self._is_preloaded = is_preloaded
        log.debug(f'Preload completed _is_preloaded: {self._is_preloaded}')

    def print_hdr(self):
        if self.row_cnt % self.hdr_reprint == 0:
            if not self.is_hdr and self.row_cnt == 0 or self.row_cnt > 0:
                hdr = self.get_hdr()
                print(hdr)
                self.is_hdr = True

    def add_row(self, row: list):
        """Member function to add one row."""
        row = deque(row)
        row.appendleft(self.row_cnt)
        self.table.add_row(row)

        self.print_hdr()
        self.row_cnt += 1

        lst = self.get_last()
        print(f'{lst}\n', end='\r', flush=True)
        # print(lst, flush=True)

    def delete_last_row(self):
        self.row_cnt -= 1
        # idx_last = len(self.table.rows) - 1
        # if idx_last >= 0:
        #     self.table.del_row(idx_last)
        #    self.row_cnt -= 1

    def update_row(self, row):
        """Function to update current - last row"""
        if not self._is_preloaded:
            return
        while not self._is_preloaded:
            log.debug(f'Waiting for preload to complete.')
            time.sleep(0.5)

        log.debug(f'Updating row: {row}')
        row = deque(row)
        row.appendleft(self.row_cnt)

        idx_last = len(self.table.rows) - 1
        if idx_last >= 0:
            self.table.del_row(idx_last)
        self.table.add_row(row)

        # Print hdr from update_row only if this is the first time the row is added/updated
        if not self.row_cnt:
            self.print_hdr()

        lst = self.get_last()
        # log.debug(f'print: {lst}')
        print(lst, end='\r', flush=True)
        # sys.stdout.flush()

    def get_hdr(self):
        hdr = self.table.get_string(title=self.title)
        t_rr = hdr.split('\n')
        hdr_idx_end = 3 if self.title is None else 5
        return "  " + "\n  ".join(t_rr[0:hdr_idx_end])

    def get_last(self):
        hdr = self.table.get_string()
        t_rr = hdr.split('\n')
        lst = t_rr[-2:-1]
        return f'  {lst.pop()}' if len(lst) else ''

    def add_alert(self, close_time, is_buy, is_sell):
        row = self.table.rows[-1].copy()
        row[2] = close_time  # Close Time
        row[-2] = is_buy  # Buy
        row[-1] = is_sell  # Sell
        row = row[1:]
        self.add_row(row)


class DroppingQueue:
    """DroppingQueue"""

    def __init__(self, maxsize=0):
        """
        Initialize a DroppingQueue with specified maxsize.
        If maxsize <= 0, the queue is unbounded.
        """
        self.maxsize = maxsize
        self._queue = deque()
        self._lock = threading.Lock()
        self._not_empty = threading.Condition(self._lock)

    def put(self, item):
        """Add an item to the queue, dropping oldest if full."""
        with self._lock:
            if 0 < self.maxsize <= len(self._queue):
                self._queue.popleft()  # Discard oldest item
            self._queue.append(item)
            self._not_empty.notify()

    def get(self, block=True, timeout=None):
        """
        Remove and return an item from the queue.
        If block is True and queue is empty, wait until an item is available.
        If block is False, raise Empty exception if queue is empty.
        """
        with self._lock:
            if not block:
                if not self._queue:
                    raise Empty
            else:
                if timeout is None:
                    while not self._queue:
                        self._not_empty.wait()
                else:
                    if timeout < 0:
                        raise ValueError("'timeout' must be a non-negative number")
                    endtime = time.monotonic() + timeout
                    while not self._queue:
                        remaining = endtime - time.monotonic()
                        if remaining <= 0.0:
                            raise Empty
                        self._not_empty.wait(remaining)
            return self._queue.popleft()

    def get_nowait(self):
        """Equivalent to get(block=False)."""
        return self.get(block=False)

    def get_all(self, limit: int = 0) -> list:
        """Get elements from queue.

        Args:
            limit: Maximum number of elements to retrieve. If 0, returns all available elements.

        Returns:
            List of elements from the queue (empty list if queue is empty)
        """
        elements = []
        try:
            if limit == 0:
                # Get all available elements
                while True:
                    elements.append(self.get_nowait())
            else:
                # Get up to 'limit' elements
                for _ in range(limit):
                    elements.append(self.get_nowait())
        except Exception as e:
            # Queue is empty, return what we've collected so far
            pass

        return elements

    def qsize(self):
        """Return the approximate size of the queue (not reliable in threaded contexts)."""
        with self._lock:
            return len(self._queue)

    def empty(self):
        """Return True if the queue is empty, False otherwise."""
        with self._lock:
            return not self._queue

    def full(self):
        """Return True if the queue is full, False otherwise."""
        with self._lock:
            return self.maxsize > 0 and len(self._queue) >= self.maxsize

    def clear(self):
        """Remove all items from the queue."""
        with self._lock:
            self._queue.clear()

class Empty(Exception):
    """Exception raised when get_nowait() is called on an empty queue."""
    pass

def std_redirect(target=None, mode='default'):
    log.debug(f'Redirecting std*: mode={mode}')
    if mode == 'default':
        with suppress(Exception):
            sys.stdout = sys.__stdout__
        with suppress(Exception):
            sys.stderr = sys.__stderr__
    elif mode == 'target' and target is not None:
        # with suppress(Exception):
        #     sys.stdout = target
        with suppress(Exception):
            sys.stderr = target
    elif mode == 'file' and target is not None:
        # To be implemented
        with suppress(Exception):
            sys.stdout = target
        with suppress(Exception):
            sys.stderr = target


def print_progress_bar_custom(iteration_completed, iteration_total, t1, operation_text='Computing'):
    fmt = '%H:%M:%S'
    d_len = len(str(iteration_total))
    time_start = datetime.datetime.fromtimestamp(t1)
    time_start_str = time_start.strftime(fmt)
    dt = time.time() - t1
    try:
        seconds = dt / (iteration_completed / iteration_total)
        eta = time_start + datetime.timedelta(seconds=seconds)
    except ZeroDivisionError:
        eta = time_start
    time_end_str = eta.strftime(fmt)
    try:
        freq = iteration_completed / dt
    except ZeroDivisionError:
        freq = 0.0
    eta_str = f'Started: {time_start_str} ETA: {time_end_str} Freq: {freq:.3f} Hz'
    prefix = f'  {"progress:":<22s} {operation_text}: {iteration_completed:{d_len}d}/{iteration_total:{d_len}d} {eta_str}\n '
    suffix = "\033[A"  # Line up ANSI code
    print_progress_bar(iteration_completed, iteration_total, prefix, suffix, decimals=2)


def get_external_ip():
    """Returns External IP: https://stackoverflow.com/questions/2311510/getting-a-machines-external-ip-address-with-python"""
    external_ip = None
    try:
        external_ip = urllib.request.urlopen('https://ident.me').read().decode('utf8')
    except Exception as e:
        pass
    return external_ip

def get_dt(t0: float) -> str:
    return f'{time.monotonic()-t0:.3f} seconds'

def convert_to_timestamp(datetime_str):
    """Converts a datetime string in the format 'YYYY-MM-DD HH:MM:SS.ffffff' to a timestamp.

    Args:
        datetime_str: The datetime string to convert.

    Returns:
        The timestamp in seconds since the Unix epoch.
    """
    if isinstance(datetime_str, float):
        return datetime_str
    datetime_obj = datetime.datetime.strptime(datetime_str, '%Y-%m-%d %H:%M:%S.%f')
    timestamp = datetime_obj.timestamp()
    return timestamp

def current_timestamp(format:str="%Y-%m-%d %H:%M:%S") -> str:
  """Returns the current timestamp as a string in the format "YYYY-MM-DD HH:MM:SS" (default) or using passed format.

  Args:
    format: str

  Returns:
    str: The current timestamp as a string.
  """

  now = datetime.datetime.now()
  timestamp = now.strftime(format)
  return timestamp

def get_os_type():
    """
    Returns the operating system type.

    Returns:
        str: 'Windows' for Windows, 'Unix' for Unix-like systems.
    """

    if sys.platform.startswith('win'):
        return 'Windows'
    else:
        return 'Unix'

def run_with_timeout(func, timeout, *args, **kwargs):
    """
    Runs the given function with a timeout.

    Args:
        func: The function to be executed.
        timeout: The maximum time in seconds to allow the function to run.
        *args: Positional arguments to be passed to the function.
        **kwargs: Keyword arguments to be passed to the function.

    Returns:
        The result of the function call if it completes within the timeout.
        None if the timeout occurs.

    Raises:
        Exception: If the function raises an exception during execution.
    """
    result = None  # Define result in the enclosing scope

    def _async_raise(thread_id, exctype):
        """Raises an exception in the threads with id `thread_id`."""
        res = ctypes.pythonapi.PyThreadState_SetAsyncExc(
            ctypes.c_long(thread_id), ctypes.py_object(exctype)
        )
        if res == 0:
            raise ValueError("Invalid thread ID")
        elif res > 1:
            ctypes.pythonapi.PyThreadState_SetAsyncExc(thread_id, 0)
            raise SystemError("PyThreadState_SetAsyncExc failed")

    def _run_with_timeout():
        nonlocal result
        try:
            result = func(*args, **kwargs)
        except Exception as e:
            result = e

    thread = threading.Thread(target=_run_with_timeout, daemon=True)
    thread.start()

    thread.join(timeout=timeout)

    if thread.is_alive():
        print(f"Function {func.__name__} timed out after {timeout} seconds. Terminating thread.")
        # Using Python 3.11+, we can use the more robust
        _async_raise(thread.ident, SystemExit)  # Raise SystemExit in the thread
        thread.join()  # Wait for the thread to terminate
            # Fallback for older Python versions and Windows
        # print(f'Exiting {func.__name__}.')
        return None

    if isinstance(result, Exception):
        raise result

    return result





def archive_folder(archive_name, path):
    # Ensure the archive name ends with .zip
    if not archive_name.endswith('.zip'):
        archive_name += '.zip'

    # Create a ZipFile object in write mode
    with zipfile.ZipFile(archive_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Walk through the directory
        for root, dirs, files in os.walk(path):
            for file in files:
                # Create the full file path
                file_path = os.path.join(root, file)
                # Add file to the zip file
                # Use arcname to preserve the directory structure
                zipf.write(file_path, os.path.relpath(file_path, os.path.dirname(path)))
            for dir in dirs:
                # Create the full directory path
                dir_path = os.path.join(root, dir)
                # Add directory to the zip file
                # Use arcname to preserve the directory structure
                zipf.write(dir_path, os.path.relpath(dir_path, os.path.dirname(path)))

    logit(f"  Archive '{archive_name}' created successfully.", color='yellow')


def delete_folder(folder_path, skip='*.md'):
    # Check if the folder exists
    if not os.path.exists(folder_path):
        logit(f"The folder '{folder_path}' does not exist.")
        return

    # Split the skip patterns into a list
    skip_patterns = skip.split(',')

    # Iterate over all items in the folder
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        # Check if the file matches any skip pattern
        if any(fnmatch.fnmatch(filename, pattern.strip()) for pattern in skip_patterns):
            logit(f"Skipping: {file_path}")
            continue

        # Check if it's a file or symbolic link
        if os.path.isfile(file_path) or os.path.islink(file_path):
            # Delete the file or symbolic link
            os.remove(file_path)
            logit(f"  Deleted file: {file_path}")
        elif os.path.isdir(file_path):
            # Delete the directory and its contents
            shutil.rmtree(file_path)
            logit(f"  Deleted directory: {file_path}")

    logit(f"Deleted: '{folder_path}'.")


def float_range(advanced_range):
    """
    Generator function that yields float values from either:
    1. Comma-separated string (e.g., "1.1,1.2,1.3,2.0")
    2. Range format (e.g., "1.1-3.0@0.1" for start-end@step)

    Args:
        advanced_range (str): Input string in either format

    Yields:
        float: The next value in the sequence
    """
    if '@' in advanced_range:  # Handle range format (e.g., "1.1-3.0@0.1")
        try:
            range_part, step_part = advanced_range.split('@')
            start, end = map(float, range_part.split('-'))
            step = float(step_part)

            current = start
            while current <= end + 1e-9:  # Add small epsilon to handle floating point precision
                yield round(current, 10)  # Round to avoid floating point artifacts
                current += step

        except ValueError as e:
            raise ValueError(f"Invalid range format: {advanced_range}. Expected 'start-end@step'") from e

    else:  # Handle comma-separated format (e.g., "1.1,1.2,1.3,2.0")
        for item in advanced_range.split(','):
            try:
                yield float(item.strip())
            except ValueError as e:
                raise ValueError(f"Invalid float value: {item}") from e



# Example usage:
# delete_all_files_and_subfolders('/path/to/folder', skip='*.md,*.txt')


# Example usage:
# archive_folder('my_archive', '/path/to/folder')

# Main Section
if __name__ == "__main__":
    print('Should not be run as standalone python script.')

    results = read_json(f'../logs/results.json')
    save_to_excel(results, '../logs/results.xlsx')


    def test_candlepp():
        import random
        cpp = CandlePP(title='Live OHLC Signal Stream')
        for idx in range(50):
            now = datetime.datetime.now().strftime('%H:%M:%S')
            c = random.uniform(21000, 22844)
            candle = [now, now, 'BTCUSD', 20193.41, 20193.41, 20186.18, c, 0.002229, False, '     ', '     ']
            # cpp.add_row(candle)
            if idx < 5:
                cpp.update_row(candle)
            elif 5 <= idx <= 25:
                cpp.update_row(candle)
                cpp.update_row(candle)
                cpp.update_row(candle)
                cpp.add_alert(1, True, False)
            elif 26 <= idx <= 45:
                cpp.update_row(candle)

            # time.sleep(2)
        # pp = cpp.table.get_string(start=3, end=4)
        hdr = cpp.get_hdr()
        lst = cpp.get_last()
        # print(hdr)
        # print()
        # print(lst)

    # test_candlepp()

# Last line
