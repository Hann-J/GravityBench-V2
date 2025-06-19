import os
import argparse
import json
import time
import copy
import tqdm
import agents.tabular_agent as TabularAgent
import datetime
import format_utils
import traceback
from scenarios_config import get_all_scenarios, get_scenario
import multiprocessing
from multiprocessing import TimeoutError
import numpy as np
from queue import Empty
import scripts.geometry_config as geometry_config
import pandas as pd

# Load configuration from config.json (like run_agent.py)
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

TEMPERATURE = float(config.get("TEMPERATURE", 0.0))
MAX_ATTEMPTS = int(config.get("MAX_ATTEMPTS", 3))
MAX_TIME_PER_TASK = int(config.get("MAX_TIME_PER_TASK", 12000))
MAX_TOKENS_PER_TASK = int(config.get("MAX_TOKENS_PER_TASK", 300000))
MAX_TOOL_CALLS_PER_TASK = int(config.get("MAX_TOOL_CALLS_PER_TASK", 100))

def output_writer(queue, output_dir):
    all_results = []
    while True:
        try:
            result = queue.get(timeout=1)
            if result is None:  # This is our signal to stop
                break
            all_results.extend(result)
            save_run_output(all_results, output_dir)
        except Empty:
            continue

def load_scenarios_from_directory(directory='scenarios'):
    """
    Load scenarios from the specified directory.
    
    Parameters:
    directory (str): Path to the directory containing scenario files.
    
    Returns:
    list: A list of scenarios with their names and filenames.
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
    Save the output of runs to a JSON directory.
    """
    # Get just the base directory name without the 'outputs_range_of_N/' prefix
    output_filename = os.path.basename(output_dir)
    output_file = os.path.join(output_dir, output_filename + '.json')
    sorted_run_results = sorted(run_results, key=lambda x: (x['scenario_name']))
    scenarios_output = {"scenarios": sorted_run_results}
    with open(output_file, 'w') as f:
        json.dump(scenarios_output, f, indent=2)

    output_html_file = os.path.join(output_dir, output_filename + '.html')
    html_result = format_utils.json_to_html(scenarios_output)
    with open(output_html_file, 'w') as f:
        f.write(html_result)

import threading

def agent_run_target(queue, scenario, row_wise, model, max_observations_per_request=10, max_observations_total=10):
    try:
        agent = TabularAgent.Agent(scenario, model=model, row_wise=row_wise, 
                                   max_observations_per_request=max_observations_per_request,
                                   max_observations_total=max_observations_total)
        result, json_chat_history = agent.run(verbose=True)
        queue.put((result, json_chat_history))
    except Exception as e:
        queue.put(e)

def run_agent_with_timeout(scenario, row_wise, model, timeout, max_observations_per_request=10, max_observations_total=10):
    queue = multiprocessing.Queue()
    thread = threading.Thread(target=agent_run_target, 
                              args=(queue, scenario, row_wise, model, max_observations_per_request, max_observations_total))
    thread.start()
    thread.join(timeout)

    if thread.is_alive():
        # Terminate the thread
        raise TimeoutError()

    result = queue.get()
    if isinstance(result, Exception):
        raise result

    return result

def run_agent_on_scenario(row_wise, scenario, scenario_name, variation_name, model='gpt-3.5-turbo', max_observations_total=10, timeout=MAX_TIME_PER_TASK, max_observations_per_request=10):
    """
    Executes the agent on a given physics scenario module and returns the results.
    
    Parameters:
    row_wise (bool): Agent must observe row by row or given full csv
    scenario_module (module): The scenario module to run the agent on.
    scenario_name (str): The name or label of the scenario.
    max_observations_total (int): Maximum total number of observations allowed (only used if row_wise is True).
    max_observations_per_request (int): Maximum number of observations per request (only used if row_wise is True).
    timeout (int): Maximum time allowed for the agent to run (in seconds).
    """
    run_results = []
    result = None
    json_chat_history = None
    
    # Get scenario-specific threshold if row-wise, otherwise use default 5%
    scenarios = get_all_scenarios()
    threshold = scenarios[scenario_name]['correct_threshold_percentage_based_on_100_observations'] / 100.0 if row_wise else 0.05
    
    while len(run_results) < MAX_ATTEMPTS and result is None:
        error_message = None
        try:
            start_time = time.time()
            result, json_chat_history = run_agent_with_timeout(scenario, row_wise, model, timeout, 
                                                               max_observations_per_request, max_observations_total)
            error_message = json_chat_history['error_message']
            end_time = time.time()
        except TimeoutError:
            print(f'Task timed out after {timeout} seconds. Retrying...')
            error_message = f'Task timed out after {timeout} seconds.'
            continue
        except Exception as e:
            print(f'INTERNAL: Error: {type(e).__name__}: {str(e)} \n{traceback.format_exc()}')
            error_message = str(type(e).__name__) + ": " + str(e)
            if str(type(e).__name__) == 'RateLimitError' or str(type(e).__name__) == 'APIStatusError':
                print(f'INTERNAL: API error, waiting 60 seconds before restarting the agent.')
                time.sleep(60)
                continue
        if error_message is not None and ("RateLimitError" in error_message or "APIStatusError" in error_message):
            message = f'INTERNAL: Rate limit error encountered. Waiting 60 seconds before retrying...'
            print(message)
            with open('rate_limit_log.txt', 'a') as f:
                f.write(f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] {message} Error: {error_message}\n')
            time.sleep(60)
            continue
        end_time = time.time()

        result = format_utils.string_to_variable(result)
        correct_answer = None
        percent_error = None
        print(f"INTERNAL: Assessing scenario '{scenario_name}' with variation '{variation_name}'")
        if isinstance(scenario.true_answer(), bool) and result is not None:
            if result == scenario.true_answer():
                correct_answer = True
            else:
                correct_answer = False
        elif isinstance(scenario.true_answer(), float) and result is not None:
            # Check if within 5% of true answer
            print(f"INTERNAL: observations attempted: {scenario.binary_sim.number_of_observations_requested if row_wise else None} / {max_observations_total if row_wise else None}")
            if isinstance(result, dict) and 'answer' in result:
                result = float(result['answer'])
            percent_error = abs( (result - scenario.true_answer()) / scenario.true_answer())
            if percent_error <= threshold:
                correct_answer = True
            else:
                correct_answer = False
        else:
            correct_answer = None

        print('Agent answer:  ', result)
        print('True answer:   ', scenario.true_answer())
        if correct_answer and correct_answer is not None:
            print('-> Correct answer')
        else:
            print('-> Incorrect answer')

        # Get human performance metrics using same number of observations as agent
        human_empirical_answer = scenario.true_answer(N_obs=max_observations_total, verification=False, return_empirical=True)
        human_extra_info_answer = scenario.true_answer(verification=True, return_empirical=False)

        # Calculate human performance metrics
        if isinstance(human_empirical_answer, bool) or isinstance(human_extra_info_answer, bool):
            human_percent_diff = None
            human_correct = human_empirical_answer == human_extra_info_answer
        elif abs(human_extra_info_answer) < 1e-10:
            human_percent_diff = None
            human_correct = None
        else:
            human_percent_diff = abs(human_empirical_answer - human_extra_info_answer) / abs(human_extra_info_answer)
            human_correct = human_percent_diff <= threshold  # Using same threshold as agent

        # Convert NumPy types to native Python types
        human_empirical_answer = float(human_empirical_answer) if isinstance(human_empirical_answer, (float, np.floating)) else bool(human_empirical_answer)
        human_extra_info_answer = float(human_extra_info_answer) if isinstance(human_extra_info_answer, (float, np.floating)) else bool(human_extra_info_answer)
        human_percent_diff = float(human_percent_diff) if human_percent_diff is not None else None
        human_correct = bool(human_correct) if human_correct is not None else None

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
            "true_answer": scenario.true_answer(),
            "threshold_used": threshold,
            "correct": correct_answer,
            "percent_error": percent_error,
            "run_time": round(end_time - start_time, 2),
            "input_tokens_used": json_chat_history['input_tokens_used'],
            "output_tokens_used": json_chat_history['output_tokens_used'],
            "cost": calculate_cost(model, json_chat_history['input_tokens_used'], json_chat_history['output_tokens_used']),
            "human_empirical_answer": human_empirical_answer,
            "human_extra_info_answer": human_extra_info_answer,
            "human_percent_diff": human_percent_diff,
            "human_correct": human_correct,
            "chat_history": json_chat_history if json_chat_history else None,
        }
        run_results.append(run_result)
        if result is None:
            print(f'INTERNAL: Agent did not return a result. Restarting the agent.')
            scenario.binary_sim.number_of_observations_requested = 0
    return run_results

def main(simulate_all=False, scenario_filenames=None, model='gpt-3.5-turbo', parallel=False, skip_simulation=False, variation_name=None, observation_ranges=None, random_geometry=int(0)):
    """
    Main function to run the agent on scenarios with different max_observations_total values.
    """
    datetime_now = datetime.datetime.now()
    formatted_datetime = datetime_now.strftime("%d-%m_%H_%M_%S")
    output_dir = f"outputs_range_of_N/{model}_{formatted_datetime}"
    os.makedirs(output_dir, exist_ok=True)
    
    if observation_ranges is None:
        observation_ranges = [10, 10, 10, 20, 20, 20, 30, 30, 30, 50, 50, 50, 70, 70, 70, 100, 100, 100]
    max_observations_per_request = 10
    row_wise = True
    
    print("INTERNAL: Max attempts per task: ", MAX_ATTEMPTS)
    scenarios = get_all_scenarios()
    all_results = []
    
    if simulate_all:
        scenarios_to_run = scenarios
    elif scenario_filenames:
        scenarios_to_run = {}
        for name in scenario_filenames:
            if name in scenarios:
                scenarios_to_run[name] = scenarios[name]
            else:
                print(f"Warning: Scenario '{name}' not found in the provided dictionary.")
    else:
        print("Please provide a list of scenarios to run or use --simulate-all.")
        return

    # Modify the scenario variations if a specific variation is requested
    if variation_name:
        for scenario_name in scenarios_to_run:
            if variation_name in scenarios_to_run[scenario_name]['variations']:
                scenarios_to_run[scenario_name]['variations'] = [variation_name]
            else:
                scenarios_to_run[scenario_name]['variations'] = []

    # Random geometry handling
    if random_geometry == 0:
        pass # No random geometry, proceed normally
    else:
        print("INTERNAL: Random geometry enabled. Duplicating files for scenarios.")
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
        
    if parallel:
        tasks = []
        for max_obs in observation_ranges:
            for scenario_name in scenarios_to_run:
                for variation_name in scenarios_to_run[scenario_name]['variations']:
                    tasks.append((
                        row_wise,
                        get_scenario(scenario_name=scenario_name, variation_name=variation_name, 
                                   row_wise=row_wise, max_observations_total=max_obs,
                                   max_observations_per_request=max_observations_per_request,
                                   scenario_folder='scenarios', skip_simulation=skip_simulation),
                        scenario_name,
                        variation_name,
                        model,
                        max_obs,
                        MAX_TIME_PER_TASK,
                        max_observations_per_request
                    ))

        result_queue = multiprocessing.Queue()
        writer_process = multiprocessing.Process(target=output_writer, args=(result_queue, output_dir))
        writer_process.start()

        with multiprocessing.Pool() as pool:
            for result in tqdm.tqdm(pool.imap_unordered(run_agent_on_scenario_star, tasks), total=len(tasks)):
                result_queue.put(result)

        result_queue.put(None)  # Signal the writer to stop
        writer_process.join()

    else:
        for max_obs in tqdm.tqdm(observation_ranges, desc="Testing observation ranges"):
            print(f"\nTesting with max_observations_total = {max_obs}")
            for scenario_name in tqdm.tqdm(scenarios_to_run, desc="Scenarios", leave=False):
                for variation_name in scenarios_to_run[scenario_name]['variations']:
                    scenario_module = get_scenario(
                        scenario_name=scenario_name,
                        variation_name=variation_name,
                        row_wise=row_wise,
                        max_observations_total=max_obs,
                        max_observations_per_request=max_observations_per_request,
                        scenario_folder='scenarios',
                        skip_simulation=skip_simulation
                    )
                    run_results = run_agent_on_scenario(
                        row_wise,
                        scenario_module,
                        scenario_name,
                        variation_name,
                        model,
                        max_obs,
                        MAX_TIME_PER_TASK,
                        max_observations_per_request
                    )
                    all_results.extend(run_results)
                save_run_output(all_results, output_dir)
    
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

    return all_results

def run_agent_on_scenario_star(args):
    return run_agent_on_scenario(*args)

def calculate_cost(model, input_tokens, output_tokens):
    if 'gpt-4o' in model.lower() and 'mini' not in model.lower():
        input_cost = 2.50 / 1e6 * input_tokens
        output_cost = 10.00 / 1e6 * output_tokens
    elif 'gpt-4o-mini' in model.lower():
        input_cost = 0.15 / 1e6 * input_tokens
        output_cost = 0.60 / 1e6 * output_tokens
    elif 'claude-3-5-sonnet' in model.lower():
        input_cost = 3.00 / 1e6 * input_tokens
        output_cost = 15.00 / 1e6 * output_tokens
    else:
        input_cost = 0
        output_cost = 0
        print(f"INTERNAL: Unknown model {model} for cost calculation")

    return input_cost + output_cost

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Run agent on specified scenarios with different max_observations_total values.'
    )
    parser.add_argument(
        '--simulate-all', action='store_true', default=False,
        help='Simulate all scenarios listed in the JSON file.'
    )
    parser.add_argument(
        '--scenarios', nargs='*',
        help='List of scenario filenames to run.'
    )
    parser.add_argument(
        '--observation-ranges', type=str,
        default='10,10,10,20,20,20,30,30,30,50,50,50,70,70,70,100,100,100',
        help='Comma-separated list of observation counts to test (default: 10,10,10,20,20,20,30,30,30,50,50,50,70,70,70,100,100,100)'
    )
    parser.add_argument(
        '--model', type=str, default='gpt-4o-mini',
        help='Model to use. Supported: gpt-3.5-turbo, gpt-4o, gpt-4o-mini, claude-3-5-sonnet-20240620, etc.'
    )
    parser.add_argument(
        '--parallel', action='store_true', default=False,
        help='Run scenarios in parallel with multiprocessing.'
    )
    parser.add_argument(
        '--skip-simulation', action='store_true', default=False,
        help='Skip simulation and use pre-computed data if available.'
    )
    parser.add_argument(
        '--variation', type=str,
        help='Specific variation name to run (e.g., "21.3 M, 3.1 M")'
    )
    parser.add_argument(
        '--random-geometry', type=int, default=0,
        help='The number of random geometry transformation for each variation, default is set to 0 and 0 will not run this version')

    args = parser.parse_args()

    if args.simulate_all:
        print("Running all scenarios.")
    if args.scenarios:
        print("Running scenarios:", args.scenarios)
    if args.variation:
        print("Running specific variation:", args.variation)
    if args.parallel:
        print("Running scenarios in parallel.")
    if args.skip_simulation:
        print("Skipping simulation and using pre-computed data if available.")
        
    # Parse observation ranges from command line argument
    observation_ranges = [int(x) for x in args.observation_ranges.split(',')]
    
    results = main(
        simulate_all=args.simulate_all,
        scenario_filenames=args.scenarios,
        model=args.model,
        parallel=args.parallel,
        skip_simulation=args.skip_simulation,
        variation_name=args.variation,
        observation_ranges=observation_ranges,  # Pass the ranges to main
        random_geometry=args.random_geometry
    )