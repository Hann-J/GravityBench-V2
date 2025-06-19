"""
Main script for running AI agents on physics scenarios with various configurations.
Handles scenario loading, parallel execution, timeouts, result collection and reporting.
"""

import os
import importlib
import argparse
import copy
import json
import time
import tqdm
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import agents.tabular_agent as TabularAgent
import datetime
import scripts.format_utils as format_utils
import traceback
from scripts.scenarios_config import get_all_scenarios, get_scenario
import multiprocessing
from multiprocessing import TimeoutError
from queue import Empty
import threading
import numpy as np
import pandas as pd
import scripts.geometry_config as geometry_config
from dotenv import load_dotenv
load_dotenv()

CONFIG_FILE_PATH = 'config.json'

try:
    with open(CONFIG_FILE_PATH, 'r') as f:
        config = json.load(f)
except FileNotFoundError:
    print(f"Error: Configuration file '{CONFIG_FILE_PATH}' not found. Please create it with necessary parameters.")
    print(f"Example: {{'TEMPERATURE': 0.5, 'MAX_ATTEMPTS': 3, 'MAX_TIME_PER_TASK': 600, 'MAX_TOKENS_PER_TASK': 4000, 'MAX_TOOL_CALLS_PER_TASK': 15}}")
    exit(1)
except json.JSONDecodeError:
    print(f"Error: Could not decode JSON from '{CONFIG_FILE_PATH}'. Please check its format.")
    exit(1)

# Configuration constants from environment variables
TEMPERATURE = float(config.get("TEMPERATURE", 0.0))
MAX_ATTEMPTS = int(config.get("MAX_ATTEMPTS", 3))
MAX_TIME_PER_TASK = int(config.get("MAX_TIME_PER_TASK", 12000))
MAX_TOKENS_PER_TASK = int(config.get("MAX_TOKENS_PER_TASK", 300000))
MAX_TOOL_CALLS_PER_TASK = int(config.get("MAX_TOOL_CALLS_PER_TASK", 100))

def output_writer(queue, output_dir):
    """
    Writer process that continuously saves results to disk.
    Handles both JSON and HTML output formats.
    
    Args:
        queue (multiprocessing.Queue): Queue to receive results from worker processes
        output_dir (str): Directory path to save output files
    """
    all_results = []
    while True:
        try:
            result = queue.get(timeout=1)
            if result is None:  # Stop signal
                break
            all_results.extend(result)
            save_run_output(all_results, output_dir)
        except Empty:
            continue

def load_scenarios_from_directory(directory='scenarios'):
    """
    Load scenario modules from a directory.
    
    Args:
        directory (str): Path to directory containing scenario Python files
    
    Returns:
        list: List of dictionaries with scenario names and filenames
    """
    scenarios = []
    for filename in os.listdir(directory):
        if filename.endswith('.py') and not filename.startswith('__'):
            scenario_name = filename[:-3]  # Remove .py extension
            scenarios.append({
                'scenario': scenario_name,
                'filename': filename
            })
    return scenarios

def save_run_output(run_results, output_dir):
    """
    Save collected results to JSON and generate HTML report.
    
    Args:
        run_results (list): List of scenario result dictionaries
        output_dir (str): Output directory path
    """
    def convert_to_native_types(obj):
        """Recursively convert numpy types to native Python types"""
        if isinstance(obj, np.generic):
            return obj.item()
        elif isinstance(obj, list):
            return [convert_to_native_types(item) for item in obj]
        elif isinstance(obj, dict):
            return {k: convert_to_native_types(v) for k, v in obj.items()}
        return obj

    output_filename = output_dir[8:]
    output_file = os.path.join(output_dir, output_filename + '.json')
    sorted_run_results = sorted(run_results, key=lambda x: (x['scenario_name']))
    scenarios_output = {"scenarios": convert_to_native_types(sorted_run_results)}
    
    with open(output_file, 'w') as f:
        json.dump(scenarios_output, f, indent=2)

    # Generate HTML report using format utilities
    output_html_file = os.path.join(output_dir, output_filename + '.html')
    html_result = format_utils.json_to_html(scenarios_output)
    with open(output_html_file, 'w') as f:
        f.write(html_result)

def agent_run_target(queue, scenario, variation_name, row_wise, model, max_observations_per_request=10, max_observations_total=10, reasoning_effort=None):
    """
    Target function for multiprocessing threads.
    Runs agent and puts results in a shared queue.
    
    Args:
        queue (multiprocessing.Queue): Shared queue for results
        scenario: Scenario object to run
        variation_name (str): Variation name
        row_wise (bool): Row-wise observation mode flag
        model (str): AI model name
        max_observations_per_request (int): Max obs per request
        max_observations_total (int): Total obs budget
        reasoning_effort (str): Reasoning effort for supported models (high, auto, none)
    """
    try:
        agent = TabularAgent.Agent(scenario, variation_name=variation_name, model=model, row_wise=row_wise,
                                   temperature=TEMPERATURE,
                                   max_tokens_per_task=MAX_TOKENS_PER_TASK,
                                   max_tool_calls_per_task=MAX_TOOL_CALLS_PER_TASK,
                                   max_execution_time=MAX_TIME_PER_TASK,
                                   max_observations_per_request=max_observations_per_request,
                                   max_observations_total=max_observations_total,
                                   reasoning_effort=reasoning_effort)
        result, json_chat_history = agent.run(verbose=True)
        queue.put((result, json_chat_history))
    except Exception as e:
        queue.put(e)

def run_agent_with_timeout(scenario, variation_name, row_wise, model, timeout, max_observations_per_request=10, max_observations_total=10, reasoning_effort=None):
    """
    Run agent with timeout protection using separate thread.
    
    Args:
        scenario: Scenario object to run
        variation_name (str): Variation name
        row_wise (bool): Row-wise observation mode
        model (str): AI model name
        timeout (int): Maximum execution time in seconds
        max_observations_per_request (int): Max obs per request
        max_observations_total (int): Total obs budget
        reasoning_effort (str): Reasoning effort for supported models (high, auto, none)
    
    Returns:
        tuple: (result, chat history) or raises exception
    """
    queue = multiprocessing.Queue()
    thread = threading.Thread(target=agent_run_target, 
                              args=(queue, scenario, variation_name, row_wise, model, max_observations_per_request, max_observations_total, reasoning_effort))
    thread.start()
    thread.join(timeout)

    if thread.is_alive():
        raise TimeoutError()

    result = queue.get()
    if isinstance(result, Exception):
        raise result

    return result

def run_agent_on_scenario(row_wise, scenario, scenario_name, variation_name, model='gpt-3.5-turbo', max_observations_total=10, timeout=MAX_TIME_PER_TASK, max_observations_per_request=10, req_successful_attempts_per_q=1, reasoning_effort=None):
    """
    Execute agent on a single scenario with retry logic.
    
    Args:
        row_wise (bool): Observation mode flag
        scenario: Scenario object to run
        scenario_name (str): Name of scenario
        variation_name (str): Variation name
        model (str): AI model name
        max_observations_total (int): Total obs budget
        timeout (int): Max execution time
        max_observations_per_request (int): Max obs per request
        req_successful_attempts_per_q (int): Required successful runs
        reasoning_effort (str): Reasoning effort for supported models (high, auto, none)
    
    Returns:
        list: Collected run results with metadata
    """
    run_results = []
    successfully_ran_attempts = 0
    total_attempts = 0
    
    # Get scenario-specific threshold
    scenarios = get_all_scenarios()
    threshold = scenarios[scenario_name]['correct_threshold_percentage_based_on_100_observations'] / 100.0 if row_wise else 0.05
    
    # Attempt loop with error handling and retries
    while successfully_ran_attempts < req_successful_attempts_per_q and total_attempts < MAX_ATTEMPTS*req_successful_attempts_per_q:
        result = None
        json_chat_history = None
        error_message = None
        try:
            start_time = time.time()
            result, json_chat_history = run_agent_with_timeout(scenario, variation_name, row_wise, model, timeout, 
                                                                max_observations_per_request, max_observations_total, reasoning_effort)
            error_message = json_chat_history['error_message']
            end_time = time.time()
            
            # Error logging
            if error_message is not None:
                print(f'INTERNAL: Error: {error_message}')
                with open('error_log.txt', 'a') as f:
                    f.write(f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] Scenario: {scenario_name} - Variation: {variation_name} - Error: {error_message}\n')
        except TimeoutError:
            print(f'Task timed out after {timeout} seconds. Retrying...')
            error_message = f'Task timed out after {timeout} seconds.'
            total_attempts += 1
            continue
        except Exception as e:
            print(f'INTERNAL: Error: {type(e).__name__}: {str(e)} \n{traceback.format_exc()}')
            error_message = str(type(e).__name__) + ": " + str(e)
            with open('error_log.txt', 'a') as f:
                f.write(f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] Scenario: {scenario_name} - Variation: {variation_name} - Error: {error_message}\n')
            if str(type(e).__name__) == 'RateLimitError' or str(type(e).__name__) == 'APIStatusError':
                print(f'INTERNAL: API error, waiting 60 seconds before restarting the agent.')
                time.sleep(60)
            total_attempts += 1
            continue

        # Result processing
        if result is None:
            total_attempts += 1
            continue

        # Handle rate limits
        if error_message is not None and ("RateLimitError" in error_message or "APIStatusError" in error_message):
            message = f'INTERNAL: Rate limit error encountered. Waiting 60 seconds before retrying...'
            print(message)
            with open('rate_limit_log.txt', 'a') as f:
                f.write(f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] {message} Error: {error_message}\n')
            time.sleep(60)
            continue

        end_time = time.time()

        # Convert and validate result
        result = format_utils.string_to_variable(result)
        correct_answer = None
        percent_error = None
        
        # Get ground truth
        print(f"INTERNAL: Assessing scenario '{scenario_name}' with variation '{variation_name}'")
        true_answer = scenario.true_answer(verification=True, return_empirical=True)
        
        # Answer validation
        if isinstance(true_answer, bool) and result is not None:
            correct_answer = (result == true_answer)
        elif isinstance(true_answer, float) and result is not None:
            if isinstance(result, dict) and 'answer' in result:
                result = float(result['answer'])
            percent_error = abs( (result - true_answer) / true_answer)
            correct_answer = (percent_error <= threshold)
        else:
            correct_answer = None

        # Build result record
        run_result = {
            "scenario_name": scenario_name,
            "variation_name": variation_name,
            "attempt": len(run_results) + 1,
            "error_message": error_message,
            "prompt": scenario.binary_sim.prompt,
            "units": scenario.binary_sim.final_answer_units,
            "model": model,
            "row_wise": row_wise,
            "max_observations_total": max_observations_total if row_wise else None,
            "max_observations_per_request": max_observations_per_request if row_wise else None,
            "observations_attempted": scenario.binary_sim.number_of_observations_requested if row_wise else None,
            "MAX_TIME_PER_TASK": MAX_TIME_PER_TASK,
            "MAX_TOKENS_PER_TASK": MAX_TOKENS_PER_TASK,
            "MAX_TOOL_CALLS_PER_TASK": MAX_TOOL_CALLS_PER_TASK,
            "temperature": TEMPERATURE,
            "result": result,
            "true_answer": true_answer,
            "threshold_used": threshold * 100,  # Store as percentage
            "correct": correct_answer,
            "percent_error": percent_error,
            "run_time": round(end_time - start_time, 2),
            "input_tokens_used": json_chat_history['input_tokens_used'],
            "output_tokens_used": json_chat_history['output_tokens_used'],
            "cost": calculate_cost(model, json_chat_history['input_tokens_used'], json_chat_history['output_tokens_used']),
            "chat_history": json_chat_history if json_chat_history else None,
            "total_attempts_so_far": total_attempts,
            "successfully_ran_attempts_so_far": successfully_ran_attempts + 1,
            "reasoning_effort": reasoning_effort
        }
        run_results.append(run_result)
        
        # Update attempt counters
        if result is not None:
            successfully_ran_attempts += 1
        
        # Reset observation counter for retries
        if successfully_ran_attempts < req_successful_attempts_per_q:
            scenario.binary_sim.number_of_observations_requested = 0

    return run_results

def main(row_wise, simulate_all=False, scenario_filenames=None, max_observations_total=10,  model='gpt-3.5-turbo', parallel=False, max_observations_per_request=10, req_successful_attempts_per_q=1, reasoning_effort=None, random_geometry=0):
    """
    Main execution function for running agent on scenarios.
    
    Args:
        row_wise (bool): Row-wise observation mode
        simulate_all (bool): Run all available scenarios
        scenario_filenames (list): Specific scenarios to run
        max_observations_total (int): Total obs budget
        model (str): AI model name
        parallel (bool): Enable parallel execution
        max_observations_per_request (int): Max obs per request
        req_successful_attempts_per_q (int): Required successful attempts
        reasoning_effort (str): Reasoning effort for supported models (high, auto, none)
        random_geometry (int): Each binary variation file is duplicated into n number of random geometry versions tested on the same scenarios
    
    Returns:
        list: All collected results
    """
    # Setup output directory
    datetime_now = datetime.datetime.now()
    formatted_datetime = datetime_now.strftime("%d-%m_%H_%M_%S")
    output_dir = f"outputs/{model}_{formatted_datetime}"
    if row_wise:
        output_dir += f"_row_wise_{max_observations_total}_{max_observations_per_request}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load scenarios
    scenarios = get_all_scenarios()
    all_results = []
    
    # Scenario selection
    if simulate_all:
        scenarios_to_run = scenarios
    elif scenario_filenames:
        scenarios_to_run = {name: scenarios[name] for name in scenario_filenames if name in scenarios}
    else:
        print("Please provide a list of scenarios to run or use --simulate-all.")
        return

    # Random geometry handling
    if random_geometry == 0:
        pass # No random geometry, proceed normally
    else:
        print("INTERNAL: Random geometry enabled. Duplicating files for different geometrical variations.")
        variations_set = {}
        transformed_variations = set() # Keep track of transformed variations so that it can be used for other scenarios, ensure uniqueness
        base_scenarios = copy.deepcopy(scenarios_to_run)

        # Loop over the variations to find all unique variations
        for scenario_name, scenario_set_ups in scenarios_to_run.items():
            for variation_name in scenario_set_ups['variations']:
                transformed_variations.add(variation_name)

        # Then apply random_geometry and set each randomly transformed variations into a dictionary with the original variation as key
        for variation in transformed_variations:
            df = pd.read_csv(f"scenarios/detailed_sims/{variation}.csv")  # Load the variation file
            variations_set[variation] = [] # Empty list for the unique variation
            for i in range(random_geometry):
                random_variation = geometry_config.random_geometry(df, file_name=variation, verification=False) # Returns a named random variation
                variations_set[variation].append(random_variation) # Update the dictionary with new random geometry variations

        # Loop over the original scenarios_to_run, if a variation is detected, append the randomly transformed variations
        for scenario_name, scenario_set_ups in base_scenarios.items():
            for variation_name in scenario_set_ups['variations']: # Every possible variations has been transformed, so no need to check for other variations
                scenarios_to_run[scenario_name]['variations'].extend(variations_set[variation_name])
        
        print("Random geometrical setup completed!")
        

    # Parallel execution setup
    if parallel:
        pool = None
        writer_process = None
        try:
            # Prepare task list
            tasks = [(row_wise, get_scenario(scenario_name=scenario_name, variation_name=variation_name, row_wise=row_wise, 
                      max_observations_total=max_observations_total, max_observations_per_request=max_observations_per_request, 
                      scenario_folder='scenarios'), 
                      scenario_name, variation_name, model, max_observations_total, MAX_TIME_PER_TASK, max_observations_per_request, req_successful_attempts_per_q, reasoning_effort)
                    for scenario_name in scenarios_to_run
                    for variation_name in scenarios_to_run[scenario_name]['variations']]

            # Start writer process
            result_queue = multiprocessing.Queue()
            writer_process = multiprocessing.Process(target=output_writer, args=(result_queue, output_dir))
            writer_process.start()

            # Process pool execution
            pool = multiprocessing.Pool()
            try:
                for result in tqdm.tqdm(pool.imap_unordered(run_agent_on_scenario_star, tasks), total=len(tasks)):
                    result_queue.put(result)
            finally:
                pool.close()
                pool.join()
                result_queue.put(None)
                writer_process.join(timeout=5)
                
        except Exception as e:
            print(f"Error in parallel processing: {e}")
            if pool:
                pool.terminate()
                pool.join()
            raise
        finally:
            # Cleanup processes
            if writer_process and writer_process.is_alive():
                writer_process.terminate()
                writer_process.join()
            if pool:
                pool.terminate()
                pool.join()

    # Sequential execution
    else:
        for scenario_name in tqdm.tqdm(scenarios_to_run):
            for variation_name in scenarios_to_run[scenario_name]['variations']:
                scenario_module = get_scenario(scenario_name=scenario_name, variation_name=variation_name, row_wise=row_wise, max_observations_total=max_observations_total, max_observations_per_request=max_observations_per_request, scenario_folder='scenarios')
                run_results = run_agent_on_scenario(row_wise, scenario_module, scenario_name, variation_name, model, 
                                                    max_observations_total, MAX_TIME_PER_TASK, max_observations_per_request, req_successful_attempts_per_q, reasoning_effort)
                all_results.extend(run_results)
            save_run_output(all_results, output_dir)
    
    # Update json file
    json_path = "scripts/scenarios_config.json"

    # Write updated data back to the json file
    with open(json_path, 'w') as f:
        json.dump(scenarios_to_run, f, indent=4)

    # Delete random geometry files for next run, use the variations_set dictionary from before to remove all randomly transformed variations
#    if random_geometry != 0:
#        for variation_list in variations_set.values(): # .values() return a lists of all the new variation names that is randomly transformed from an original variation
#            for variation in variation_list:
#                file_path_detailed = f"scenarios/detailed_sims/{variation}.csv"
#                file_path_sims = f"scenarios/sims/{variation}.csv"
#                if os.path.exists(file_path_detailed):
#                    os.remove(file_path_detailed)
#                if os.path.exists(file_path_sims):
#                    os.remove(file_path_sims)
#        print("INTERNAL: Random variation files has been deleted")
#
    return all_results

def run_agent_on_scenario_star(args):
    """Wrapper function for multiprocessing with star arguments."""
    return run_agent_on_scenario(*args)

def calculate_cost(model, input_tokens, output_tokens):
    """
    Calculate API costs based on token usage and model pricing.
    
    Args:
        model (str): Model name
        input_tokens (int): Number of input tokens
        output_tokens (int): Number of output tokens
    
    Returns:
        float: Total cost in USD
    """
    # Pricing information for various models (USD per million tokens)
    if 'gpt-4o' in model.lower() and 'mini' not in model.lower():
        input_cost = 2.50 / 1e6 * input_tokens
        output_cost = 10.00 / 1e6 * output_tokens
    elif 'gpt-4o-mini' in model.lower():
        input_cost = 0.15 / 1e6 * input_tokens
        output_cost = 0.60 / 1e6 * output_tokens
    elif 'sonnet' in model.lower():
        input_cost = 3.00 / 1e6 * input_tokens
        output_cost = 15.00 / 1e6 * output_tokens
    elif 'haiku' in model.lower():
        input_cost = 0.80 / 1e6 * input_tokens
        output_cost = 4.00 / 1e6 * output_tokens
    elif 'opus' in model.lower():
        input_cost = 15 / 1e6 * input_tokens
        output_cost = 75 / 1e6 * output_tokens
    elif 'o1' in model.lower():
        input_cost = 15 / 1e6 * input_tokens
        output_cost = 60 / 1e6 * output_tokens
    else:
        input_cost = 0
        output_cost = 0
        print(f"INTERNAL: Unknown model {model} for cost calculation")

    return input_cost + output_cost

if __name__ == "__main__":
    # Command-line argument parsing
    parser = argparse.ArgumentParser(description='Run agent on specified physics scenarios.')
    
    # Scenario configuration
    parser.add_argument('--row-wise', action='store_true', default=False,
                       help='Row-wise observation mode (vs full CSV access)')
    parser.add_argument('--simulate-all', action='store_true', default=False,
                       help='Run all available scenarios')
    parser.add_argument('--scenarios', nargs='*',
                       help='List of specific scenarios to run')
    parser.add_argument('--max-observations-total', type=int, default=100,
                       help='Total observation budget for row-wise mode')
    parser.add_argument('--random-geometry', type=int, default=0,
                        help='The number of random geometry transformation for each variation, default is set to 0 and 0 will not run this version')
    
    # Model selection
    parser.add_argument('--model', type=str, default='gpt-4o-mini',
                       help='AI model to use (supports various OpenAI/Anthropic models)')
    
    # Execution parameters
    parser.add_argument('--parallel', action='store_true', default=False,
                       help='Enable parallel execution')
    parser.add_argument('--max-observations-per-request', type=int, default=10,
                       help='Maximum observations per request in row-wise mode')
    parser.add_argument('--req-successful-attempts-per-q', type=int, default=1,
                       help='Required successful attempts per question')

    # Add reasoning_effort argument to the parser
    parser.add_argument('--reasoning-effort', type=str, default=None,
                       help='Reasoning effort for supported models (high, auto, none)')

    args = parser.parse_args()

    # Input validation
    if not args.row_wise:
        mot_action = next((action for action in parser._actions if action.dest == 'max_observations_total'), None)
        original_default_mot_value = mot_action.default if mot_action else 100
        if args.max_observations_total != original_default_mot_value and args.max_observations_total != 10:
            print(f"Warning: --max-observations-total (set to {args.max_observations_total}) is ignored without --row-wise. Using 10.")
        args.max_observations_total = 10  # Force value for non-row-wise mode
        mopr_action = next((action for action in parser._actions if action.dest == 'max_observations_per_request'), None)
        original_default_mopr_value = mopr_action.default if mopr_action else 10
        if args.max_observations_per_request != original_default_mopr_value:
            print(f"Warning: --max-observations-per-request (set to {args.max_observations_per_request}) is ignored without --row-wise. The default of {original_default_mopr_value} will be used by agent if applicable, but generally this param is for row-wise mode.")

    # Execute main program
    results = main(
        row_wise=args.row_wise,
        simulate_all=args.simulate_all,
        scenario_filenames=args.scenarios,
        max_observations_total=args.max_observations_total,
        model=args.model,
        parallel=args.parallel,
        max_observations_per_request=args.max_observations_per_request,
        req_successful_attempts_per_q=args.req_successful_attempts_per_q,
        reasoning_effort=args.reasoning_effort,
        random_geometry=args.random_geometry
    )