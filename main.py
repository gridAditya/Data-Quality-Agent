import os
import asyncio
import sys
from colorama import Fore, Style
from services.code_act_agent.code_act_agent import CodeActAgent
from services.llm_client.llm_client import LLMClient
from services.code_executor.code_executor import CodeExecutor
from utils.utils import read_file_content
from config import logging_config
import logging

from dotenv import load_dotenv # load the environment variables
load_dotenv(override=True)

def get_multiline_input():
    """Get multi-line input from user."""
    print(f"{Fore.GREEN}{Style.BRIGHT}[+] Enter your query (paste and press Ctrl+D): {Style.RESET_ALL}")
    
    try:
        # Read all input until EOF
        content = sys.stdin.read()
        return content.strip()
    except KeyboardInterrupt:
        print("\nInput cancelled.")
        return ""

async def main():
    # Load the system prompt
    system_prompt = read_file_content("/Users/adsingh/Desktop/R&D_Projects/self_improving_agents/services/code_act_agent/code_act_agent_prompt.txt")

    # Load the LLMClient, CodeExecutor
    llm_client = LLMClient(base_url=os.getenv('ANTHROPIC_BASE_URL'), api_key=os.getenv('ANTHROPIC_API_KEY'))
    code_executor = CodeExecutor(allowed_filesystem_paths=['/Users/adsingh/Desktop/R&D_Projects/self_improving_agents/datasets'], allowed_file_modes=['r', 'w'])

    # Spawn the agent
    agent = CodeActAgent(conversation_id="2ae6269d-7963-4574-a37f-c934d158985d", system_prompt=system_prompt, llm_client=llm_client, code_executor=code_executor)

    while True:
        user_query = get_multiline_input()
        
        if not user_query.strip():
            print("Empty query. Exiting...")
            break
            
        print(f"\n{Fore.YELLOW}Processing your query...{Style.RESET_ALL}\n")
        logging.info(f"{Fore.GREEN}{Style.BRIGHT}[+] Original user query received (length: {len(user_query)} characters){Style.RESET_ALL}")

        # Run the agent
        agent_response = await agent.run_agentic_loop(user_query=user_query, model="anthropic/claude-sonnet-4-20250514")
        logging.info(f"{Fore.GREEN}{Style.BRIGHT}[+] Report: {Fore.BLACK}{agent_response}{Style.RESET_ALL}")
        
        print(f"\n{Fore.CYAN}Query processed successfully!{Style.RESET_ALL}\n")

if __name__ == "__main__":
    asyncio.run(main())