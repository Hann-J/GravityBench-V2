import numpy as np
import pandas as pd
from dataclasses import dataclass

@dataclass
class ModelStats:
    """Container for model performance statistics across evaluation scenarios.
    
    Attributes:
        score: Percentage of correct answers (mean ± SE)
        score_se: Standard error of score percentage
        num_questions: Number of unique scenario variations evaluated
        total_cost: Total API cost in USD (mean ± SE across runs)
        cost_se: Standard error of total cost
        total_time: Total runtime in minutes (mean ± SE across runs)
        time_se: Standard error of total time
        obs_attempted: Mean number of observations used per question (row-wise mode)
        obs_se: Standard error of observations attempted
        num_runs: Number of experimental runs (consistent across scenarios)
    """
    score: float
    score_se: float
    num_questions: int
    total_cost: float
    cost_se: float
    total_time: float
    time_se: float
    obs_attempted: float
    obs_se: float
    num_runs: int

def compute_stats_for_scenarios(df_sub, model_name) -> ModelStats:
    """
    Calculate aggregate statistics for a model's performance across scenario variations.
    
    Approach:
    - Treat each scenario variation as an independent question
    - Handle potential multiple runs by:
      1. Ensuring consistent number of runs across scenarios
      2. Calculating run-wise totals for cost/time metrics
      3. Calculating scenario-wise averages for accuracy/observation metrics
    
    Args:
        df_sub: Filtered DataFrame containing data for one model and evaluation mode
        model_name: Identifier for debugging output

    Returns:
        ModelStats object containing aggregated performance metrics
    """
    # Handle empty input case (e.g., missing model data)
    if len(df_sub) == 0:
        return ModelStats(
            score=np.nan, score_se=np.nan,
            num_questions=0,
            total_cost=np.nan, cost_se=np.nan,
            total_time=np.nan, time_se=np.nan,
            obs_attempted=np.nan, obs_se=np.nan,
            num_runs=0
        )
    
    # Create copy to avoid modifying original data and prevent pandas warnings
    df_sub = df_sub.copy()
    
    # Ensure required columns exist (fill missing with defaults)
    for col in ['cost', 'run_time', 'observations_attempted']:
        if col not in df_sub.columns:
            # Default to 0 for metrics, NaN for optional fields
            default_val = 0.0 if col in ['cost', 'run_time'] else np.nan
            df_sub[col] = default_val

    # Convert runtime from seconds to minutes using in-place modification
    # Note: Using .loc to avoid chained assignment warnings
    df_sub.loc[:, 'run_time'] = df_sub['run_time'] / 60.0
    
    # Debugging output for data validation
    print(f"\nDEBUG for {model_name}:")
    print(f"Total rows in df_sub: {len(df_sub)}")
    print(f"Unique scenario variations: {df_sub['scenario_variation'].nunique()}")
    
    # Group by scenario variation to calculate per-question metrics
    grouped = df_sub.groupby('scenario_variation', dropna=False)
    
    # Lists to collect scenario-level metrics
    scenario_correct = []  # Mean accuracy per scenario
    scenario_obs = []      # Mean observations used per scenario
    n_runs_list = []       # Track number of runs per scenario

    # First pass: Determine number of consistent runs across all scenarios
    for _, scenario_group in grouped:
        n_runs_list.append(len(scenario_group))
    
    # Use minimum number of complete runs available across all scenarios
    actual_runs = min(set(n_runs_list)) if n_runs_list else 0
    print(f"Actual runs: {actual_runs}")

    # Calculate cost and time metrics (run-level aggregation)
    costs_per_run = []
    times_per_run = []
    
    # Aggregate costs/times across scenarios for each run index
    for run_idx in range(actual_runs):
        run_cost = 0
        run_time = 0
        # Sum across all scenarios for this run index
        for _, scenario_group in grouped:
            if run_idx < len(scenario_group):
                run_cost += scenario_group.iloc[run_idx]['cost']
                run_time += scenario_group.iloc[run_idx]['run_time']
        costs_per_run.append(run_cost)
        times_per_run.append(run_time)

    # Second pass: Calculate accuracy and observation metrics (scenario-level)
    for _, scenario_group in grouped:
        # Use only consistent number of runs
        scenario_runs = scenario_group.iloc[:actual_runs]
        
        # Calculate mean correctness across runs for this scenario
        correctness_mean = scenario_runs['correct'].astype(float).mean()
        scenario_correct.append(correctness_mean)

        # Calculate mean observations attempted (handle missing data)
        if 'observations_attempted' in scenario_runs.columns:
            obs_val = scenario_runs['observations_attempted'].dropna().mean()
        else:
            obs_val = np.nan
        scenario_obs.append(obs_val)

    # Convert collections to numpy arrays for vectorized calculations
    scenario_correct = np.array(scenario_correct)
    scenario_obs = np.array(scenario_obs)
    costs_per_run = np.array(costs_per_run)
    times_per_run = np.array(times_per_run)

    # Calculate final statistics with error propagation
    # Score: Mean accuracy across scenarios converted to percentage
    final_percentage = scenario_correct.mean() * 100 if len(scenario_correct) > 0 else np.nan
    # Standard error of mean score across scenarios
    final_se = np.sqrt(scenario_correct.var(ddof=1) / len(scenario_correct)) * 100 if len(scenario_correct) > 1 else 0.0

    # Observation metrics (only relevant for row-wise mode)
    obs_attempted = np.nanmean(scenario_obs) if len(scenario_obs) > 0 else np.nan
    valid_obs = scenario_obs[~np.isnan(scenario_obs)]
    obs_attempted_se = np.sqrt(valid_obs.var(ddof=1) / len(valid_obs)) if len(valid_obs) > 1 else 0.0
    
    # Cost metrics (mean and SE across runs)
    final_cost = np.mean(costs_per_run) if len(costs_per_run) > 0 else np.nan
    cost_se = np.sqrt(costs_per_run.var(ddof=1) / len(costs_per_run)) if len(costs_per_run) > 1 else 0.0
    
    # Time metrics (mean and SE across runs)
    final_time = np.mean(times_per_run) if len(times_per_run) > 0 else np.nan
    time_se = np.sqrt(times_per_run.var(ddof=1) / len(times_per_run)) if len(times_per_run) > 1 else 0.0
    
    # Number of unique scenario variations processed
    num_questions = len(scenario_correct)

    return ModelStats(
        score=final_percentage,
        score_se=final_se,
        num_questions=num_questions,
        total_cost=final_cost,
        cost_se=cost_se,
        total_time=final_time,
        time_se=time_se,
        obs_attempted=obs_attempted,
        obs_se=obs_attempted_se,
        num_runs=actual_runs
    )


if __name__ == "__main__":
    # Load experimental results from CSV
    df = pd.read_csv("outputs/combined_results.csv")
    
    # Data validation: Ensure correctness is boolean
    if 'correct' in df.columns:
        df['correct'] = df['correct'].astype(bool)
    
    # Initialize result containers
    all_models = df['model'].unique()
    results = {}          # Full table access results (row_wise=False)
    results_row_wise = {} # Row-wise access results (row_wise=True)
    
    # Process each model's results
    for model_name in all_models:
        print(f"Processing {model_name}")
        
        # Full table access mode (row_wise=False)
        df_full = df[(df['model'] == model_name) & (df['row_wise'] == False)]
        results[model_name] = compute_stats_for_scenarios(df_full, model_name)
        
        # Row-wise access mode (row_wise=True)
        df_row = df[(df['model'] == model_name) & (df['row_wise'] == True)]
        results_row_wise[model_name] = compute_stats_for_scenarios(df_row, model_name)
    
    # Print human-readable results tables
    # Full Table Results
    print("\nModel Performance Results - Full Table Access")
    print("-" * 200)
    print(f"{'Model':<30} {'Score':>18} {'Questions':>12} {'Total Cost':>20} {'Total Time (min)':>20} {'Runs':>8}")
    print("-" * 200)
    for model, stats in results.items():
        print(f"{model:<30} {stats.score:>8.1f}% ± {stats.score_se:>4.1f}% {stats.num_questions:>12} {stats.total_cost:>10.2f} ± {stats.cost_se:>4.2f} {stats.total_time:>10.1f} ± {stats.time_se:>4.1f} {stats.num_runs:>8}")
    print("-" * 200)
    
    # Row-wise Results
    print("\nModel Performance Results - Sequential Observations (100 Observation Budget)")
    print("-" * 200)
    print(f"{'Model':<30} {'Score':>18} {'Questions':>12} {'Total Cost':>20} {'Total Time (min)':>20} {'Mean Obs':>18} {'Runs':>8}")
    print("-" * 200)
    for model, stats in results_row_wise.items():
        print(f"{model:<30} {stats.score:>8.1f}% ± {stats.score_se:>4.1f}% {stats.num_questions:>12} {stats.total_cost:>10.2f} ± {stats.cost_se:>4.2f} {stats.total_time:>10.1f} ± {stats.time_se:>4.1f} {stats.obs_attempted:>8.2f} {stats.num_runs:>8}")
    print("-" * 200)

    # Generate LaTeX table for paper submission
    latex_table = r"""
\begin{table*}[!htb]
    \centering
    \footnotesize
    \caption{Model performance. Each model: 2 runs.}\label{tab:performance}
    \begin{tabular}{lcccc}
    \toprule
    & \textbf{Score} & \textbf{Total Cost (\$)} & \textbf{Total Time (min)} & \textbf{Mean Observations Used} \\
    \midrule
    \multicolumn{5}{l}{\textbf{Full Table Access}} \\
"""
    # Determine maximum scores for highlighting
    max_score_full = max(stats.score for model, stats in results.items() if model != 'human')
    max_score_row = max(stats.score for model, stats in results_row_wise.items() if model != 'human')

    # Sort models by descending score for both tables
    sorted_models_full = sorted(results.items(), key=lambda x: x[1].score, reverse=True)
    sorted_models_row = sorted(results_row_wise.items(), key=lambda x: x[1].score, reverse=True)

    # Full table entries
    for model, stats in sorted_models_full:
        if model == 'human':
            # Human baseline has different metric availability
            latex_table += f"    {model} & {stats.score:.1f}\\% & - & - & - \\\\\n"
        else:
            # Highlight best performing non-human model
            score_str = (f"\\textbf{{{stats.score:.1f}\\% $\\pm$ {stats.score_se:.1f}\\%}}" 
                        if stats.score == max_score_full 
                        else f"{stats.score:.1f}\\% $\\pm$ {stats.score_se:.1f}\\%")
            latex_table += f"    {model} & {score_str} & {stats.total_cost:.2f} $\\pm$ {stats.cost_se:.2f} & {stats.total_time:.1f} $\\pm$ {stats.time_se:.1f} & - \\\\\n"

    # Row-wise table entries
    latex_table += r"""\\[0.5em]
    \multicolumn{5}{l}{\textbf{Sequential Observations - 100 Observation Budget}} \\
"""
    for model, stats in sorted_models_row:
        if model == 'human':
            latex_table += f"    {model} & {stats.score:.1f}\\% & - & - & {stats.obs_attempted:.1f} \\\\\n"
        else:
            score_str = (f"\\textbf{{{stats.score:.1f}\\% $\\pm$ {stats.score_se:.1f}\\%}}" 
                        if stats.score == max_score_row 
                        else f"{stats.score:.1f}\\% $\\pm$ {stats.score_se:.1f}\\%")
            latex_table += f"    {model} & {score_str} & {stats.total_cost:.2f} $\\pm$ {stats.cost_se:.2f} & {stats.total_time:.1f} $\\pm$ {stats.time_se:.1f} & {stats.obs_attempted:.1f} $\\pm$ {stats.obs_se:.1f} \\\\\n"

    # Finalize LaTeX table
    latex_table += r"""    \bottomrule
    \end{tabular}
\end{table*}
"""
    
    # Write LaTeX output to file
    with open('analysis/tables/table1_scores.tex', 'w') as f:
        f.write(latex_table)