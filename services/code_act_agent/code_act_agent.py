import os
import re

from colorama import Fore, Style
from services.llm_client.llm_client import LLMClient
from services.code_executor.code_executor import CodeExecutor
from config import logging_config
import logging

class CodeActAgent:
    def __init__(self, conversation_id: str, system_prompt: str, llm_client: LLMClient, code_executor: CodeExecutor):
        # Validate the data types
        if not isinstance(conversation_id, str):
            raise ValueError(
                f"Expected 'conversation_id' to be of type str, but got {type(conversation_id).__name__}"
            )
        if not isinstance(system_prompt, str):
            raise ValueError(
                f"Expected 'system_prompt' to be of type str, but got {type(system_prompt).__name__}"
            )
        if not isinstance(llm_client, LLMClient):
            raise ValueError(
                f"Expected 'llm_client' to be an instance of LLMClient, but got {type(llm_client).__name__}"
            )
        if not isinstance(code_executor, CodeExecutor):
            raise ValueError(
                f"Expected 'code_executor' to be an instance of CodeExecutor, but got {type(code_executor).__name__}"
            )
        
        # Set the attributes
        self.system_prompt = system_prompt
        self.llm_client = llm_client
        self.code_executor = code_executor
        self.conversation_id = conversation_id
        self.workspace_dir_path = f"/Users/adsingh/Desktop/R&D_Projects/self_improving_agents/datasets/workspace/{self.conversation_id}"
        self.chat_history: list = [{"role": "system", "content": self.system_prompt}]
        self.iteration_num: int = 0
        self.max_iteration: int = 10

    def _get_last_modified_file(self, dir_path: str) -> str:
        if not os.path.isdir(dir_path):
            return ""
        
        # List all non-hidden files (exclude files starting with ".")
        files = [
            os.path.join(dir_path, f)
            for f in os.listdir(dir_path)
            if os.path.isfile(os.path.join(dir_path, f)) and not f.startswith(".")
        ]
        
        if not files:
            return ""
        
        # Get the file with the maximum modification time
        last_modified_file = max(files, key=os.path.getmtime)
        return os.path.abspath(last_modified_file)

    def _get_last_working_copy_abs_path(self)->str:
        return self._get_last_modified_file(self.workspace_dir_path)
        
    async def _get_response(self, **kwargs):
        # Make the messages list
        messages = self.chat_history

        # Get the LLM response
        response = await self.llm_client.send_request(messages=messages, **kwargs)
        return response
    
    def _parse_response(self, response: str) -> tuple[str, list[str]]:
        """
        Parse `response` and return a tuple (response_type, list_of_contents).

        - If one or more <code>...</code> tags are present: return ("code", [code1, code2, ...])
        - Else if one or more <response>...</response> tags are present: return ("text", [resp1, resp2, ...])
        - Else: raise XML Error
        """
        if response is None:
            raise ValueError("Invalid XML Response Format: expected <code>...</code> or <response>...</response>")

        if ("<code>" in response) and ("<response>" in response):
            raise ValueError("Invalid XML Response Format: <code> and <response> cannot both be present. Only one is allowed.")
        
        # look for all <code>...</code> blocks (non-greedy)
        code_pattern = r"<code>(.*?)</code>"
        codes = re.findall(code_pattern, response, flags=re.DOTALL | re.IGNORECASE)
        if codes:
            # strip leading/trailing whitespace from each block
            codes = [c.strip() for c in codes]
            return "code", codes

        # look for all <response>...</response> blocks
        resp_pattern = r"<response>(.*?)</response>"
        responses = re.findall(resp_pattern, response, flags=re.DOTALL | re.IGNORECASE)
        if responses:
            responses = [r.strip() for r in responses]
            return "text", responses

        # fallback: no tags found — return the whole input as a single text item
        raise ValueError("Invalid XML Response Format: expected <code>...</code> or <response>...</response>")

    async def run_agentic_loop(self, user_query: str, **kwargs)->str:
        self.iteration_num = 0
        final_response = ""
        if self._get_last_working_copy_abs_path():
            user_query = user_query + "\n" + f"- Last Working Copy: {self._get_last_working_copy_abs_path()}"
        else:
            user_query = user_query + "\n" + f"- Last Working Copy: None"
        user_query = user_query + f"\n- RUN {self.iteration_num+1}/{self.max_iteration}"
        self.chat_history.append({"role": "user", "content": user_query}) # Update Chat history
        logging.info(f"{Fore.GREEN}{Style.BRIGHT}[+] Modified Query:\n{Fore.BLACK}{user_query}{Style.RESET_ALL}")
        while True:
            # Break if exceeded iteration limit
            if self.iteration_num == self.max_iteration:
                break
            
            # Get response from LLM
            response = await self._get_response(**kwargs)
            self.chat_history.append({"role": "assistant", "content": response.choices[0].message.content}) # update chat-history
            logging.info(f"{Fore.BLUE}{Style.BRIGHT}[+] Agent Run: {Fore.BLACK}{self.iteration_num+1}/{self.max_iteration}{Style.RESET_ALL}")
            logging.info(f"{Fore.BLUE}{Style.BRIGHT}[+] Response Type: {Fore.BLACK}{type(response.choices[0].message.content)}, {Fore.BLUE}Length of Response: {Fore.BLACK}{len(response.choices[0].message.content)}{Style.RESET_ALL}")
            logging.info(f"{Fore.BLUE}{Style.BRIGHT}[+] Agent Response: {Style.RESET_ALL}{response.choices[0].message.content}\n\n")

            # Parse the response and execute code if present
            try:
                response_type, parsed_response_list = self._parse_response(response=response.choices[0].message.content)
                
                if response_type.lower() == "text": # Text Response(Exit)
                    final_response = "\n".join(parsed_response_list)
                    break
                else: # Code Response(Execute Code)
                    # Execute each block
                    agg_code_output = ""
                    code_execution_error = False
                    for idx, code_block in enumerate(parsed_response_list): # execute each code block
                        logging.info(f"{Fore.BLUE}{Style.BRIGHT}[+] Executing Code Block: {idx+1}{Style.RESET_ALL}")
                        try:
                            executor_output = self.code_executor.execute(code_block)
                            if executor_output["success"]:
                                agg_code_output = agg_code_output + f"Output of Code Block {idx+1}:\n{executor_output['output']}\n\n"
                                logging.info(f"{Fore.BLUE}{Style.BRIGHT}[+] Code Block {idx+1} Output:\n{Style.RESET_ALL}{executor_output['output']}\n\n")
                                logging.info(f"{Fore.BLACK}{Style.BRIGHT}***************************************************************************************************************************************************{Style.RESET_ALL}")
                            else:
                                raise Exception(f"Code Caused the following ERROR:\n{executor_output['error']}\n\nTraceback:\n{executor_output['traceback']}")
                        except Exception as e:
                            last_modified_file_path = self._get_last_working_copy_abs_path()
                            self.iteration_num += 1 # update the iteration counter
                            if last_modified_file_path:
                                user_response = f"**ERROR**:\n{e}\n\n- Last Working Copy: {last_modified_file_path}"
                            else:
                                user_response = f"**ERROR**:\n{e}\n\n- Last Working Copy: None"
                            user_response = user_response + f"\n- RUN {self.iteration_num+1}/{self.max_iteration}"
                            self.chat_history.append({"role": "user", "content": user_response}) # update chat history
                            code_execution_error = True
                            logging.error(f"{Fore.RED}{Style.BRIGHT}[-] Code Execution Error: {Fore.BLACK}{e}{Style.RESET_ALL}")
                            break
                    
                    if code_execution_error:
                        logging.error(f"{Fore.RED}{Style.BRIGHT}[-] Code Execution error encountered....{Style.RESET_ALL}")
                        continue
                    self.iteration_num += 1 # update the iteration counter
                    agg_code_output = agg_code_output.strip()
                    last_modified_file_path = self._get_last_working_copy_abs_path()
                    if last_modified_file_path:
                        user_response = f"Below is the output of your code:\n{agg_code_output}\n\n- Last Working Copy: {last_modified_file_path}"
                    else:
                        user_response = f"Below is the output of your code:\n{agg_code_output}\n\n- Last Working Copy: None"
                    user_response = user_response + f"\n- RUN {self.iteration_num+1}/{self.max_iteration}"
                    self.chat_history.append({"role": "user", "content": user_response}) # update chat history
            except ValueError as e: # Invalid XML Format Found
                last_modified_file_path = self._get_last_working_copy_abs_path()
                self.iteration_num += 1 # update the iteration counter
                if last_modified_file_path:
                    user_response = f"**ERROR**:\n{e}\n\n- Last Working Copy: {last_modified_file_path}"
                else:
                    user_response = f"**ERROR**:\n{e}\n\n- Last Working Copy: None"
                user_response = user_response + f"\n- RUN {self.iteration_num+1}/{self.max_iteration}"
                self.chat_history.append({"role": "user", "content": user_response}) # update chat history
                logging.error(f"{Fore.RED}{Style.BRIGHT}[-] Response Generation Error: {Fore.BLACK}{e}{Style.RESET_ALL}")
                continue
        
        logging.info(f"{Fore.GREEN}{Style.BRIGHT}[+] Agent Execution Completed, agent took {self.iteration_num} number of runs to complete task......{Style.RESET_ALL}")
        return final_response
