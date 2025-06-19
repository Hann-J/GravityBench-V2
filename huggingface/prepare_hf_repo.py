import json
import os
import importlib
from scripts.scenarios_config import get_all_scenarios as load_scenarios_config_json, variations as scenario_creator_variations # Renamed to avoid conflict

# Assuming scripts.scenarios_config.get_scenario can be adapted or we use its logic
# For simplicity, let's directly use the scenarios_config.json and BinaryScenario for true_answer and prompt
# And we'll need a way to get the scenario-specific class (e.g., Apoastron) to call true_answer

def get_scenario_instance(scenario_name, variation_name, scenario_folder='scenarios'):
    """
    Simplified way to get a scenario instance for its methods.
    """
    if variation_name not in scenario_creator_variations:
        print(f"Warning: Variation '{variation_name}' not found in scenario_creator_variations. Skipping.")
        return None
    
    scenario_creator = scenario_creator_variations[variation_name]
    
    # Dynamically import the scenario module (e.g., scenarios.apoastron)
    try:
        scenario_module = importlib.import_module(f"{scenario_folder}.{scenario_name}")
        # Instantiate the scenario class (e.g., Apoastron.Scenario)
        # This requires that the __init__ of these classes can handle scenario_creator directly
        # The original get_scenario in scenarios_config.py does more setup.
        # We might need to simplify the Scenario class constructors or replicate parts of get_scenario.
        # For now, let's assume a way to get the prompt and true_answer.

        # A more robust way, reusing more of existing structure:
        from scripts.scenarios_config import get_scenario as get_full_scenario_object
        return get_full_scenario_object(scenario_name, variation_name)

    except ImportError:
        print(f"Could not import scenario module: {scenario_folder}.{scenario_name}")
        return None
    except AttributeError:
        print(f"Scenario class not found or structured as expected in {scenario_folder}.{scenario_name}")
        return None


def create_dataset_item(scenario_name, config_data, variation_name, include_full_csv=True):
    """
    Create a dataset item for a given scenario and variation.
    
    Args:
        scenario_name: Name of the scenario
        config_data: Configuration data for the scenario
        variation_name: Name of the variation
        include_full_csv: If True, include full CSV content. If False, use placeholder.
    
    Returns:
        Dictionary representing the dataset item, or None if creation failed.
    """
    print(f"Processing: {scenario_name} - {variation_name}")

    scenario_instance = get_scenario_instance(scenario_name, variation_name)
    if not scenario_instance:
        print(f"Could not get instance for {scenario_name} - {variation_name}. Skipping.")
        return None

    task_prompt = scenario_instance.binary_sim.task
    expected_units = scenario_instance.binary_sim.final_answer_units
    
    try:
        # Use the non-empirical, non-verifying version for the "cleanest" true answer
        true_answer = scenario_instance.true_answer(verification=False, return_empirical=False)
        # Convert to string to ensure consistent data type across all scenarios
        true_answer = str(true_answer)
    except Exception as e:
        print(f"Error getting true_answer for {scenario_name} - {variation_name}: {e}")
        return None

    sim_csv_filename = f"{variation_name}.csv"
    sim_csv_path = os.path.join("scenarios", "sims", sim_csv_filename)

    if not os.path.exists(sim_csv_path):
        print(f"CSV file not found: {sim_csv_path}. Skipping.")
        return None

    # Handle CSV content based on split type
    if include_full_csv:
        with open(sim_csv_path, 'r', encoding='utf-8') as f_csv:
            simulation_csv_content = f_csv.read()
    else:
        # For preview split, use placeholder
        simulation_csv_content = "[CSV data omitted for preview]"

    scenario_id = f"{scenario_name}_{variation_name.replace(' ', '_').replace(',', '').replace('.', 'p')}"
    budget_obs_threshold = config_data.get("correct_threshold_percentage_based_on_100_observations", 5.0)

    return {
        "scenario_id": scenario_id,
        "scenario_name": scenario_name,
        "variation_name": variation_name,
        "task_prompt": task_prompt,
        "expected_units": expected_units,
        "true_answer": true_answer,
        "full_obs_threshold_percent": 5.0,
        "budget_obs_threshold_percent": float(budget_obs_threshold),
        "simulation_csv_filename": sim_csv_filename,
        "simulation_csv_content": simulation_csv_content,
    }


def main():
    all_scenario_configs = load_scenarios_config_json() # This is from your scenarios_config.json
    
    output_dir = "huggingface"
    os.makedirs(output_dir, exist_ok=True)

    # Create both splits
    test_items = []
    preview_items = []

    for scenario_name, config_data in all_scenario_configs.items():
        for variation_name in config_data.get("variations", []):
            
            # Create test item (with full CSV)
            test_item = create_dataset_item(scenario_name, config_data, variation_name, include_full_csv=True)
            if test_item:
                test_items.append(test_item)
            
            # Create preview item (with CSV placeholder)
            preview_item = create_dataset_item(scenario_name, config_data, variation_name, include_full_csv=False)
            if preview_item:
                preview_items.append(preview_item)

    # Write test split
    test_jsonl_path = os.path.join(output_dir, "test.jsonl")
    with open(test_jsonl_path, 'w', encoding='utf-8') as f_out:
        for item in test_items:
            f_out.write(json.dumps(item) + "\n")
    print(f"test dataset written to {test_jsonl_path} ({len(test_items)} items)")

    # Write preview split
    preview_jsonl_path = os.path.join(output_dir, "preview.jsonl")
    with open(preview_jsonl_path, 'w', encoding='utf-8') as f_out:
        for item in preview_items:
            f_out.write(json.dumps(item) + "\n")
    print(f"Preview dataset written to {preview_jsonl_path} ({len(preview_items)} items)")


if __name__ == "__main__":
    main()