import os
import asyncio
import sys
from colorama import Fore, Style
from services.code_act_agent.code_act_agent import CodeActAgent
from services.llm_client.llm_client import LLMClient
from services.code_executor.code_executor import CodeExecutor
from utils.utils import read_file_content, generate_unique_id, ensure_directory
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
    
    # Generate a unique ID corresponding to the conversation
    conversation_id = generate_unique_id()
    logging.info(f"{Fore.GREEN}{Style.BRIGHT}[+] Conversation created: {conversation_id}{Style.RESET_ALL}")

    # Create conversation-workspace directory if not exists
    workspace_path = f"/Users/adsingh/Desktop/R&D_Projects/self_improving_agents/datasets/workspace/{conversation_id}"
    is_workspace_created = ensure_directory(workspace_path)

    if not is_workspace_created:
        raise FileExistsError(f"Unable to create directory at '{workspace_path}' path")
    
    # Currently we have hardcoded the dataset-name
    dataset_name = "AirQualityUCI.csv"
    
    # Spawn the agent
    agent = CodeActAgent(conversation_id=conversation_id, system_prompt=system_prompt, llm_client=llm_client, code_executor=code_executor)

    while True:
        user_query = get_multiline_input()
        # Add the Workspace info to user-prompt
        user_query += f"""\n\nDataset Paths:
- Source: /Users/adsingh/Desktop/R&D_Projects/self_improving_agents/datasets/source/{dataset_name}
- Worksapce Directory: /Users/adsingh/Desktop/R&D_Projects/self_improving_agents/datasets/workspace/{conversation_id}"""

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