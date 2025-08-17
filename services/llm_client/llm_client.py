from litellm import acompletion
from config import logging_config
from colorama import Fore, Style

import logging

class LLMClient:
    def __init__(self, base_url: str, api_key: str):
        # Validate the data types
        if not isinstance(base_url, str):
            raise ValueError(
                f"Expected 'base_url' to be of type str, but got {type(base_url).__name__}"
            )
        if not isinstance(api_key, str):
            raise ValueError(
                f"Expected 'api_key' to be of type str, but got {type(api_key).__name__}"
            )
        
        # Set the attributes
        self.base_url = base_url
        self.api_key = api_key

    async def send_request(self, **kwargs):
        # Get response from API provider
        logging.info(f"{Fore.BLUE}{Style.BRIGHT}[+] Sending request to: {Fore.BLACK}{self.base_url}{Style.RESET_ALL}")
        response = await acompletion(base_url=self.base_url, api_key=self.api_key, **kwargs)
        return response

async def test_llm_client():
    import os
    from dotenv import load_dotenv
    load_dotenv(override=True)

    llm_client = LLMClient(
        base_url=os.getenv('ANTHROPIC_BASE_URL'), 
        api_key=os.getenv('ANTHROPIC_API_KEY')
    )
    messages = [{"content": "Tell me a data-science joke", "role": "user"}]
    response = await llm_client.send_request(
        messages=messages, 
        model="anthropic/claude-3-5-sonnet-20240620"
    )
    
    # Return the response immediately without sleep
    return response.choices[0].message.content

async def main():
    try:
        response = await test_llm_client()
        print(f"{Fore.GREEN}{Style.BRIGHT}[+] Model Response: {Fore.BLACK}{response}{Style.RESET_ALL}")
    except Exception as e:
        print(f"{Fore.RED}{Style.BRIGHT}[-] Error: {e}{Style.RESET_ALL}")
    finally:
        # Give time for cleanup
        await asyncio.sleep(0.1)

if __name__ == "__main__":
    # Import required packages
    import asyncio
    asyncio.run(main())