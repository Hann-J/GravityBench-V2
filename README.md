# Gravity-Bench: A Benchmark for AI Discovery of Gravitational Physics

[![Paper](https://img.shields.io/badge/arXiv-2501.18411-B31B1B)](https://arxiv.org/abs/2501.18411v2)
[![Website](https://img.shields.io/badge/Website-gravitybench.github.io-blue)](https://gravitybench.github.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 
[![Hugging Face](https://img.shields.io/badge/HuggingFace-GravityBench-orange)](https://huggingface.co/datasets/GravityBench/GravityBench)

This repository contains the code and benchmark for the ICML 2025 paper: **"Gravity-Bench-v1: A Benchmark on Gravitational Physics Discovery for Agents"** (arXiv:2501.18411). Visit our project website at [gravitybench.github.io](https://gravitybench.github.io/) for discussion on the results.

Gravity-Bench is a benchmark that evaluates AI agents on solving gravitational physics problems by providing them tools to **observe** a two-body gravitational system and **write code** to solve the problem. 

Our range of tasks include difficult problems, such as determining how we modified gravity and determining the coefficient of drag that has been added to the system (problems not often seen in textbooks).

These questions challenge AI agents with skills that mirror real-world science, including iterative reasoning, planning, and generalization.

![Flowchart](analysis/plots/flowchart5.png)


## Key Features

-   **Physics-based Environment:**
    Uses high-precision simulations capturing realistic two-body orbital dynamics. Simulations are configured in `scripts/scenarios_config.py` and are simulated in `generalscenarios/Binary.py`. Observational data is stored in `scenarios/sims/` (for agent use) and detailed simulation data for verification is in `scenarios/detailed_sims/`.

-   **Partial Observability:**
    Agents can operate in two modes:
    1.  **Full-Observation (full-obs):** Access the complete simulation data at once via a `pandas` dataframe (e.g., via `scripts/run_agent.py`).
    2.  **Budgeted Observation (budget-obs):** Agent must collect observations incrementally under a defined observation budget, usually 100 observations (e.g., via `scripts/run_agent.py --row-wise`). This mode emphasizes strategic observation planning.

-   **Tasks:**
    Each task is assigned to multiple simulation variations, configured in `scripts/scenarios_config.json`. See Appendix B of our paper for a detailed description of benchmark problems.

-   **Expert Reference Solutions & Ground Truth:**
    -   **Expert Reference Solutions:** Implemented in each task's `.py` file (e.g., `scenarios/mass_star1.py`), these solutions emulate an expert's approach using only the observable data (full-obs or budget-obs with uniform sampling). They serve as a  baseline for agent performance and are generated using `scripts/run_expert_solution.py`.
    -   The same `true_answer` methods can also return exact values derived from simulation inputs or the simulations internal calculations (when `verification=True` and `return_empirical=False`). This is used for verifying the correctness of the environment and the empirical solutions.

-   **Agent:**
    The benchmark is designed for iterative, tool-augmented agents. The provided baseline agent (`agents/tabular_agent.py`) uses:
    -   An `Observe` tool for taking measurements at specific times (`agents/tools/observe_tool.py`).
    -   A Python REPL tool for data analysis (`agents/tools/python_repl_tool.py`).
        - Warning: The Python REPL tool executes arbitrary code without containerization and should be used with caution.
    -   A `submit_answer` tool for providing the final numeric or boolean answer (`agents/tools/submit_answer_tool.py`).
    This agent is executed using `scripts/run_agent.py` or `scripts/run_agent_range_of_budgets.py` (for varying observation budgets). The agent prompts are detailed in Appendix D of our paper.

## Getting Started

1.  **Install Dependencies:**

    ### Option A: uv (Recommended)
    
    [uv](https://docs.astral.sh/uv/) is a fast Python package manager that automatically handles virtual environments.
    
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Or: pip install uv
    
    # Install dependencies (automatically creates virtual environment)
    uv sync
    
    # Activate the environment (optional - uv run will auto-activate)
    source .venv/bin/activate
    ```

    ### Option B: Conda/Mamba
    ```bash
    # Install dependencies using conda
    conda env create -f environment.yml
    conda activate gravitybench
    ```
    ### Option C: pip
    ```bash
    # Create and activate virtual environment
    python -m venv .venv
    source .venv/bin/activate
    
    # Install dependencies
    pip install -r requirements.txt
    ```

**Note for Windows Users:** The benchmark code relies on the `rebound` physics simulation library, which does not support Windows. Windows users should use our standalone `quick_start.py` script instead, which works with our [Huggingface dataset](https://huggingface.co/datasets/GravityBench/GravityBench) and doesn't require `rebound`. See the "Running your own models and agents" section below for details.

2.  **Configure API Keys:**
    
    Create a `.env` file with your API keys using `.env.example`.

3. **Run a Scenario (Full Observations):**
    This example runs the agent on the `period.py` scenario task using the `gpt-4o` model with full observation access.
    
    ```bash
    python scripts/run_agent.py --scenarios modified_gravity_power_law --model gpt-4.1
    ```
    
    Outputs (a JSON file and an HTML report) will be saved in a new subdirectory under `outputs/`, named like `outputs/gpt-4o_<timestamp>/`.

4.  **Run a Scenario (Budgeted Observations):**
    This example runs the agent on the `max_velocity_star1.py` scenario task, using a Claude model, with a budget of 100 total observations, requesting up to 10 at a time.
    
    ```bash
    python scripts/run_agent.py \
        --scenarios max_velocity_star1 \
        --model gpt-4.1 \
        --row-wise \
        --max-observations-total 100 \
        --max-observations-per-request 10
    ```

5.  **Explore Results:**
    After each run, navigate to the `outputs/` (or `outputs_range_of_N/`) directory. Inside the run-specific subfolder (e.g., `outputs/gpt-4o_<timestamp>/`), you will find:
    -   A `<model>_<timestamp>.json` file containing detailed run data.
    -   A `<model>_<timestamp>.html` file, which is an interactive report. Open this in your browser to see a summary and step-by-step logs for each scenario attempt.

## Running your own models and agents

To run your own model you can modify `agents/tabular_agent.py`. To run your own agent framework, we recommend adapting our `quick_start.py` which contains a minimal all-in-one example for a simple agent framework using our [Huggingface dataset](https://huggingface.co/datasets/GravityBench/GravityBench).

## Reproducing Paper Results

To reproduce the main results presented in the paper (arXiv:2501.18411), follow these general steps:

1.  **Run Experiments with Different Agents and Settings:**
    Refer to Section 4 of the paper for details on models and configurations tested.
    *   **Full-Observation Mode:**
        ```bash
        python scripts/run_agent.py --simulate-all --model gpt-4o-mini-2024-07-18
        ```
    *   **Budgeted-Observation Mode (e.g., 100 observations):**
        ```bash
        python scripts/run_agent.py --simulate-all --model gpt-4o-mini-2024-07-18 --row-wise --max-observations-total 100
        ```
    *   **Varying Observation Budgets (for specific scenarios/models, see Figure 2 in paper):**
        ```bash
        python scripts/run_agent_range_of_budgets.py --model gpt-4o-2024-11-20 --scenarios max_velocity_star1 periastron --variation "9.6 M, 3.1 M" --variation "3.1 M, 0.18 M, Elliptical, Single Orbit"
        ```

2.  **Generate Expert Baseline Data:**
    This script calculates the performance of the expert-defined empirical solutions.
    ```bash
    python scripts/run_expert_solution.py
    ```
    This will create `outputs/expert_baseline_results.csv`.

3.  **Aggregate Raw Experiment Results:**
    After running the agent experiments, combine their JSON outputs:
    ```bash
    python outputs/combine_results.py
    ```
    This creates `outputs/combined_results.csv` and `outputs/chat_histories.csv`.

    For results from `run_agent_range_of_budgets.py`:
    ```bash
    python outputs_range_of_N/aggregate.py
    ```
    This creates `outputs_range_of_N/aggregated_results.csv`.

4.  **Generate Figures and Tables:**
    Run the analysis scripts located in the `analysis/` directory. These scripts use the aggregated CSV files.
    ```bash
    python analysis/table1_scores.py       # For Table 1
    python analysis/table2_massassumption.py # For Table 2 in Appendix
    python analysis/fig_casestudy.py       # For Figure 3 (Traces) - Note: this script uses specific hardcoded observation data.
    python analysis/plot_100_obs_human_performance.py # For Figure 4 (Thresholds)
    python outputs_range_of_N/fig_maxvelocity_and_periastron.py # For Figure 2 (Varying Budgets)
    python outputs_range_of_N/fig_periastron.py # For Figure 5 (Periastron Case Study)
    ```
    The generated figures will typically be saved in `analysis/plots/` or `outputs_range_of_N/plots/`, and LaTeX tables in `analysis/tables/`.

## Citation

If you use Gravity-Bench in your research, please cite our paper:

```bibtex
@misc{koblischke2025gravitybenchv1,
      title={Gravity-Bench-v1: A Benchmark on Gravitational Physics Discovery for Agents}, 
      author={Nolan Koblischke and Hyunseok Jang and Kristen Menou and Mohamad Ali-Dib},
      year={2025},
      eprint={2501.08411},
      archivePrefix={arXiv},
      primaryClass={cs.AI}
}
```
![Simulation Example](analysis/plots/Simulations.png)

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.
