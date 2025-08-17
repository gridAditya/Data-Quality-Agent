from config import logging_config
import logging

def read_file_content(file_path: str)->str:
    """
    Read entire file content using UTF-8 encoding with proper error handling.
    
    Args:
        file_path (str): Absolute Path to the file to read
        
    Returns:
        Optional[str]: File content as string, or None if error occurred
    """
    
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
        logging.info(f"Successfully read file: {file_path}")
        return content
            