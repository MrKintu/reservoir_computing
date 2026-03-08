"""
Custom logger for the reservoir computing project.
Provides consistent logging across all modules.
"""

import logging
import os
from datetime import datetime
from typing import Optional

# Create logs directory if it doesn't exist
LOGS_DIR = os.path.join(os.getcwd(), "logs")
os.makedirs(LOGS_DIR, exist_ok=True)

class ProjectLogger:
    """Custom logger with file and console output."""
    
    def __init__(self, name: str, log_file: Optional[str] = None):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        
        # Avoid duplicate handlers
        if not self.logger.handlers:
            # Create formatters
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            
            # Console handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
            
            # File handler
            if log_file is None:
                timestamp = datetime.now().strftime("%Y%m%d")
                log_file = f"{name}_{timestamp}.log"
            
            file_path = os.path.join(LOGS_DIR, log_file)
            file_handler = logging.FileHandler(file_path)
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
    
    def info(self, message: str):
        """Log info message."""
        self.logger.info(message)
    
    def warning(self, message: str):
        """Log warning message."""
        self.logger.warning(message)
    
    def error(self, message: str):
        """Log error message."""
        self.logger.error(message)
    
    def debug(self, message: str):
        """Log debug message."""
        self.logger.debug(message)

# Convenience function to get logger
def get_logger(name: str, log_file: Optional[str] = None) -> ProjectLogger:
    """Get a logger instance for the given module name."""
    return ProjectLogger(name, log_file)
