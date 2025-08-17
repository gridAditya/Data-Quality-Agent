import os
import uuid
from config import logging_config
import logging

from colorama import Fore, Style

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

def generate_unique_id():
    """
    Generate a unique ID using UUID4.
    Returns:
        str: A unique identifier string.
    """
    return str(uuid.uuid4())


def ensure_directory(path: str) -> bool:
    """
    Ensure that the given directory exists.
    If it does not exist, create it.

    Args:
        path (str): The directory path to check/create.

    Returns:
        bool: True if the directory exists or was created successfully,
              False if there was an error.
    """
    try:
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)  # safely creates nested dirs too
            print(f"{Fore.GREEN}{Style.BRIGHT}[+] Directory created: {Fore.BLACK}{path}{Style.RESET_ALL}")
        else:
            print(f"{Fore.GREEN}{Style.BRIGHT}[+] Directory already exists: {Fore.BLACK}{path}{Style.RESET_ALL}")
        return True
    except OSError as e:
        print(f"{Fore.GREEN}{Style.BRIGHT}[-] Error creating directory '{path}': {Fore.BLACK}{e}{Style.RESET_ALL}")
        return False