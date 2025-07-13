import logging

def setup_logger(name: str, log_file: str, level=logging.DEBUG):
    try:
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
        )
        
        handler = logging.FileHandler(log_file)
        handler.setFormatter(formatter)

        logger = logging.getLogger(name)
        logger.setLevel(level)

        if not logger.handlers:
            logger.addHandler(handler)
        
        return logger
    except Exception as e:
        print(e)
