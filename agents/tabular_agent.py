import os
import json
import time
import traceback
import pandas as pd
import numpy as np
import scipy
import copy
import sklearn
import statsmodels.api as sm
from dotenv import load_dotenv, find_dotenv
import anthropic
import openai
import scripts.format_utils as format_utils
import sys
import math
from agents.tools.observe_tool import observe_tool, execute_observe_tool
from agents.tools.submit_answer_tool import submit_answer_tool, execute_submit_answer
from agents.tools.python_repl_tool import python_repl_tool, execute_python_repl
from colorama import Fore, Style, init
init()  # Initialize colorama
from generalscenarios.Binary import RowWiseResults
load_dotenv(find_dotenv())

class MessageLogItem:
    def __init__(self, text):
        self.content = text

class StepRecord:
    def __init__(self, tool, tool_input, message_log):
        self.tool = tool
        self.tool_input = tool_input
        self.message_log = message_log  # a list of MessageLogItem

class StepInfo:
    """
    A tiny container that has the attributes format_utils expects to read:
    step[0].tool
    step[0].tool_input
    step[0].message_log
    """
    def __init__(self, tool, tool_input, message_log):
        self.tool = tool
        self.tool_input = tool_input
        self.message_log = message_log

class Agent:
    """
    An AI agent designed to explore and analyze a given environment
    to discover physical laws or patterns.

    Attributes:
        environment (object): The environment the agent will interact with.
        model (str): The model used by the agent.
        row_wise (bool): Whether the agent observes row-wise or is given the full csv.
        max_observations_per_request (int): Maximum number of observations per request (only used if row_wise is True).
        max_observations_total (int): Maximum total number of observations allowed (only used if row_wise is True).
    """
    def __init__(self, environment, variation_name, model, row_wise, 
                 temperature, max_tokens_per_task, max_tool_calls_per_task, max_execution_time, # New params
                 max_observations_per_request=10, max_observations_total=10, reasoning_effort=None):
        self.environment = environment
        self.model = model
        self.row_wise = row_wise
        self.max_tokens = max_tokens_per_task 
        self.max_tool_calls = max_tool_calls_per_task 
        self.max_execution_time = max_execution_time 
        self.environment.binary_sim.number_of_observations_requested = 0
        self.max_observations_per_request = max_observations_per_request
        self.max_observations_total = max_observations_total
        self.reasoning_effort = reasoning_effort
        print(f'INTERNAL: Max tool calls: {self.max_tool_calls}')
        print(f'INTERNAL: Max execution time: {self.max_execution_time}')
        if self.reasoning_effort:
            print(f'INTERNAL: Reasoning effort: {self.reasoning_effort}')
        self.temperature = temperature 
        self.package_names = "numpy scipy sklearn statsmodels pandas"
        self.df = self.environment.binary_sim.df
        self.available_packages = {
            "np": np, "scipy": scipy, "sklearn": sklearn, "sm": sm, "pd": pd
        }
        self.columns = self.df.columns
        if row_wise:
            assert self.max_tool_calls > math.ceil(self.max_observations_total/self.max_observations_per_request), f"Max tool calls {self.max_tool_calls} must be larger than max observations {self.max_observations_total}"
            self.row_wise_results = RowWiseResults()
            self.environment.binary_sim.row_wise_results = self.row_wise_results  # Set row_wise_results for Binary
            self.available_packages.update({
                "row_wise_results": self.row_wise_results
            })
            self.exploration_prompt = self.environment.binary_sim.row_wise_prompt
        else:
            if "FOP" in variation_name:
                self.df = copy.deepcopy(self.df)
                self.df['star1_z'] = 0.0
                self.df['star2_z'] = 0.0
            self.available_packages.update({
                "df": self.df
            })
            self.exploration_prompt = self.environment.binary_sim.full_table_prompt
        
        print(f'INTERNAL: {self.exploration_prompt}')
        self.tools = []
        if row_wise:
            self.tools.append(observe_tool(maximum_observations_per_request=self.max_observations_per_request, metadata={'environment': self.environment}))
        self.tools.extend([
            python_repl_tool(_globals=self.available_packages, _locals=self.available_packages, package_names=self.package_names),
            submit_answer_tool()
        ])
        for tool in self.tools:
            print("INTERNAL: Tool info provided to agent:")
            print(f'INTERNAL Tool: {tool["function"]["name"]}')
            print(f'INTERNAL description: {tool["function"]["description"]}')
            print('-' * 50)

    def initialize_model(self, model, temperature):
        print(model.lower())
        if 'claude' in model.lower():
            return anthropic.Anthropic()
        elif 'gpt' in model.lower() or model.lower().startswith('o'):
            print(f"INTERNAL: Using OpenAI API")
            return openai.OpenAI()
        else:
            raise ValueError(f"Model {model} not recognized")

    def _convert_tools_for_anthropic(self, tools):
        """Convert OpenAI tools to Anthropic format"""
        return [{"name": t["function"]["name"], "description": t["function"]["description"], 
                "input_schema": t["function"]["parameters"]} for t in tools if t["type"] == "function"]

    def run(self, verbose=True):
        """Explore the environment following the ReAct protocol, with improved error handling."""
        print("INTERNAL: Starting agent run")
        self.environment.binary_sim.number_of_observations_requested = 0
        messages = [{"role": "system", "content": self.exploration_prompt}]
        llm = self.initialize_model(self.model, self.temperature)
        print(f"INTERNAL: Initialized {self.model} model")
        last_answer = None
        trace = {
            "input": self.exploration_prompt,
            "intermediate_steps": [],
            "output": None,
            "error_message": None,
        }
        total_input_tokens_used = 0
        total_output_tokens_used = 0

        try:
            if verbose:
                print(f"{Fore.CYAN}System: {self.exploration_prompt}{Style.RESET_ALL}")
            messages.append({"role": "user", "content": "Begin your analysis."})
            start_time = time.time()
            tool_calls_made = 0

            while (tool_calls_made < self.max_tool_calls and 
                   time.time() - start_time < self.max_execution_time - 2):  # 2s buffer
                print(f"INTERNAL: Tool calls made: {tool_calls_made}/{self.max_tool_calls}")
                print(f"INTERNAL: Time elapsed: {time.time() - start_time:.2f}/{self.max_execution_time}s")
                # print("INTERNAL: Entering action mode")
                # Request an Action
                try:
                    if 'claude' in self.model.lower():
                        # Extract system message for Anthropic API
                        system_msg = None
                        anthropic_messages = []
                        for msg in messages:
                            if msg["role"] == "system":
                                system_msg = msg["content"]
                            else:
                                anthropic_messages.append(msg)
                        
                        response = llm.messages.create(
                            model=self.model,
                            max_tokens=20000,
                            system=system_msg,
                            messages=anthropic_messages,
                            tools=self._convert_tools_for_anthropic(self.tools),
                            temperature=self.temperature
                        )
                        total_input_tokens_used += getattr(response.usage, "input_tokens", 0)
                        total_output_tokens_used += getattr(response.usage, "output_tokens", 0)
                        # Convert Anthropic response to OpenAI-like structure
                        class MockMsg: pass
                        assistant_message = MockMsg()
                        assistant_message.content = ""
                        assistant_message.tool_calls = []
                        for block in response.content:
                            if block.type == "text":
                                assistant_message.content += block.text
                            elif block.type == "tool_use":
                                class MockCall: pass
                                class MockFunc: pass
                                call = MockCall()
                                call.id = block.id
                                call.function = MockFunc()
                                call.function.name = block.name
                                call.function.arguments = json.dumps(block.input)
                                assistant_message.tool_calls.append(call)
                        assistant_message.tool_calls = assistant_message.tool_calls or None
                    elif self.model.lower().startswith('o'):
                        if self.reasoning_effort:
                            print(f"INTERNAL: Using model with reasoning effort: {self.reasoning_effort}")
                            response = llm.chat.completions.create(
                                model=self.model,
                                messages=messages,
                                tools=self.tools,
                                reasoning_effort=self.reasoning_effort
                            )
                        else:
                            print(f"INTERNAL: Using {self.model} model")
                            response = llm.chat.completions.create(
                                model=self.model,
                                messages=messages,
                                tools=self.tools
                            )
                        total_input_tokens_used += getattr(response.usage, "prompt_tokens", 0)
                        total_output_tokens_used += getattr(response.usage, "completion_tokens", 0)
                        assistant_message = response.choices[0].message
                    else:
                        response = llm.chat.completions.create(
                            model=self.model,
                            messages=messages,
                            tools=self.tools,
                            temperature=self.temperature
                        )
                        total_input_tokens_used += getattr(response.usage, "prompt_tokens", 0)
                        total_output_tokens_used += getattr(response.usage, "completion_tokens", 0)
                        assistant_message = response.choices[0].message
                    
                except Exception as api_error:
                    error_name = type(api_error).__name__
                    error_description = str(api_error)
                    error_traceback = traceback.format_exc()
                    print(f"INTERNAL: API Error occurred - {error_name}: {error_description}")
                    trace["error_message"] = f"API Error occurred: {error_name} - {error_description} - Traceback: {error_traceback}"
                    raise  # Re-raise the exception to be caught by the outer try-except

                content = assistant_message.content or ""
                tool_calls = assistant_message.tool_calls

                if content is not None and content.strip() and verbose:
                    print(f"{Fore.MAGENTA}Assistant: {content}{Style.RESET_ALL}")

                if tool_calls:
                    tool_calls_made += len(tool_calls)
                    print(f"INTERNAL: Processing {len(tool_calls)} tool call(s)")
                    
                    # Handle message format based on model type
                    if 'claude' in self.model.lower():
                        # Anthropic format: build content blocks
                        assistant_content = []
                        if content.strip():
                            assistant_content.append({"type": "text", "text": content})
                        for tool_call in tool_calls:
                            assistant_content.append({
                                "type": "tool_use",
                                "id": tool_call.id,
                                "name": tool_call.function.name,
                                "input": json.loads(tool_call.function.arguments)
                            })
                        messages.append({"role": "assistant", "content": assistant_content})
                    else:
                        # OpenAI format
                        messages.append({"role": "assistant", "content": content, "tool_calls": tool_calls})

                    for i, tool_call in enumerate(tool_calls):
                        # Check for malformed calls
                        if (not tool_call or
                            not getattr(tool_call, "function", None) or
                            not getattr(tool_call.function, "name", None)):
                            print("INTERNAL: Malformed tool call, skipping.")
                            continue

                        tool_name = tool_call.function.name
                        raw_args = tool_call.function.arguments
                        try:
                            tool_args = json.loads(raw_args)
                        except Exception as parse_error:
                            print(f"INTERNAL: Invalid JSON arguments: {raw_args} - {parse_error}")
                            continue

                        print(f"INTERNAL: Executing tool: {tool_name}")
                        if verbose:
                            print(f"{Fore.YELLOW}Action: {tool_name}")
                            print(f"Arguments: {json.dumps(tool_args, indent=2)}{Style.RESET_ALL}")

                        msg_log = []
                        if i == 0 and content.strip():
                            msg_log.append(MessageLogItem(content))

                        # Execute the appropriate tool
                        if tool_name == "Observe":
                            result = execute_observe_tool(
                                tool_args["times_requested"],
                                self.environment,
                                self.max_observations_per_request
                            )
                        elif tool_name == "PythonREPL":
                            result = execute_python_repl(
                                tool_args["input_code"],
                                self.available_packages,
                                self.available_packages
                            )
                        elif tool_name == "submit_answer":
                            print("INTERNAL: Submitting final answer")
                            result = execute_submit_answer(tool_args["answer"])
                            last_answer = result
                            trace["output"] = str(result)

                        if verbose:
                            print(f"{Fore.BLUE}Tool Output: {result}{Style.RESET_ALL}")

                        messages.append({
                            "role": "tool" if 'claude' not in self.model.lower() else "user",
                            "content": str(result) if 'claude' not in self.model.lower() else [{"type": "tool_result", "tool_use_id": tool_call.id, "content": str(result)}],
                            "tool_call_id": tool_call.id
                        } if 'claude' not in self.model.lower() else {
                            "role": "user",
                            "content": [{"type": "tool_result", "tool_use_id": tool_call.id, "content": str(result)}]
                        })

                        # Don't add tool output to message log
                        trace["intermediate_steps"].append([
                            StepRecord(
                                tool=tool_name,
                                tool_input=tool_args,
                                message_log=msg_log  # Only include agent messages
                            ),
                            str(result)  # Tool output stays separate
                        ])

                        if tool_name == "submit_answer":
                            print("INTERNAL: Answer submitted, ending run")
                            break

                    if last_answer is not None:
                        print("INTERNAL: Answer submitted, ending run")
                        break

                else:
                    # If no tool calls, treat as a regular message and continue
                    print("INTERNAL: No tool calls made, continuing")
                    if content.strip():
                        trace["intermediate_steps"].append([
                            StepRecord(
                                tool="Assistant Text",
                                tool_input="",
                                message_log=[MessageLogItem(content)]
                            ),
                            ""
                        ])
                    messages.append({"role": "assistant", "content": content})

                # Check token limit with buffer
                if (total_input_tokens_used + total_output_tokens_used) > (self.max_tokens - 50):
                    print(f"INTERNAL: Token limit of {self.max_tokens} exceeded")
                    raise Exception(f"Token limit of {self.max_tokens} exceeded")

        except Exception as e:
            error_name = type(e).__name__
            error_description = str(e)
            error_traceback = traceback.format_exc()
            print(f"INTERNAL: Error occurred - {error_name}: {error_description}")
            if verbose:
                print(f"\n{Fore.RED}Error occurred: {error_name} - {error_description}")
                print(f"Traceback:\n{error_traceback}{Style.RESET_ALL}")
            
            trace["intermediate_steps"].append([
                StepRecord(
                    tool="INTERNAL MESSAGE: Agent Error",
                    tool_input="",
                    message_log=[MessageLogItem(f"An error occurred: {error_name} - {error_description}")]
                ),
                f"Agent encountered an error: {error_name}. Trace saved, but no answer was submitted."
            ])
            trace['error_message'] = f"Error occurred: {error_name} - {error_description} - Traceback: {error_traceback}"
            last_answer = None

        finally:
            print("INTERNAL: Run complete")
            print(f"INTERNAL: Final token usage - Input: {total_input_tokens_used}, Output: {total_output_tokens_used}")
            trace["input_tokens_used"] = total_input_tokens_used
            trace["output_tokens_used"] = total_output_tokens_used
            return last_answer, format_utils.convert_result_to_json(trace)