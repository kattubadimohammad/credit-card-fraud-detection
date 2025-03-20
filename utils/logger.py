import logging

def setup_logger(name='fraud_detection_logger', log_file='../reports/project.log', level=logging.INFO):
    """Setup a logger for the project."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

    # Stream handler
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger

# Example Usage
if __name__ == "__main__":
    logger = setup_logger()
    logger.info("Logger setup complete. Ready to track logs.")