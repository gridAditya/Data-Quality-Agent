import logging

from colorama import Fore, Style

logging.basicConfig(
    format="%(asctime)s:%(funcName)s %(filename)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d:%H:%M",
    level=logging.INFO
)

if __name__ == "__main__":
    logging.info(f"{Fore.GREEN}{Style.BRIGHT}[+] Logging Configured Successfully.....{Style.RESET_ALL}")