"""
Error logger utility for detailed error tracking and logging
"""

import os
import sys
import traceback
import logging
from datetime import datetime

class ErrorLogger:
    """
    A utility class for detailed error logging and debugging
    """
    def __init__(self, log_dir=None):
        """
        Initialize the error logger
        
        Args:
            log_dir: Directory to store log files. If None, uses a default directory.
        """
        if log_dir is None:
            # Create logs directory in the application root
            app_root = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
            log_dir = os.path.join(app_root, "logs")
        
        # Create the logs directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
        
        # Create a timestamped log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"error_log_{timestamp}.txt")
        
        # Configure logging
        logging.basicConfig(
            filename=log_file,
            level=logging.DEBUG,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Also print to console
        console = logging.StreamHandler(sys.stdout)
        console.setLevel(logging.INFO)
        formatter = logging.Formatter('%(levelname)s: %(message)s')
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)
        
        self.logger = logging.getLogger('error_logger')
        self.log_file = log_file
    
    def log_error(self, message, exception=None, level="ERROR"):
        """
        Log an error message and exception traceback
        
        Args:
            message: Error message to log
            exception: Exception object, if available
            level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        """
        level = level.upper()
        if exception:
            tb = traceback.format_exc()
            full_message = f"{message}\n{str(exception)}\n{tb}"
        else:
            full_message = message
        
        if level == "DEBUG":
            self.logger.debug(full_message)
        elif level == "INFO":
            self.logger.info(full_message)
        elif level == "WARNING":
            self.logger.warning(full_message)
        elif level == "CRITICAL":
            self.logger.critical(full_message)
        else:  # Default to ERROR
            self.logger.error(full_message)
        
        return full_message
    
    def get_log_file_path(self):
        """Get the path to the current log file"""
        return self.log_file

# Create a singleton instance
error_logger = ErrorLogger()

def log_error(message, exception=None, level="ERROR"):
    """Helper function to log errors using the singleton ErrorLogger"""
    return error_logger.log_error(message, exception, level)

def get_log_file_path():
    """Helper function to get the log file path from the singleton ErrorLogger"""
    return error_logger.get_log_file_path()