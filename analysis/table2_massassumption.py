import pandas as pd
import numpy as np
import re

# Read the chat histories and results
df = pd.read_csv('outputs/chat_histories.csv')
# Read the scenarios requiring mass
# these are defined by having mass somewhere in their true_answer solutions
with open('analysis/scenarios_requiring_mass.txt', 'r') as f:
    mass_required_scenarios = set(f.read().splitlines())
n_mass_problems = len(mass_required_scenarios)
# Define patterns to search for mass assumptions
mass_patterns = [
    r"\(df\[ star1_x \] \+ df\[ star2_x \]\) \/ 2",
    r'star1_mass = 1.0',
    r'star1_mass = 1',
    r'star2_mass = 1.0',
    r'star2_mass = 1',
    r'm1 = m2',
    r'm1 = 1.0',
    r'm2 = 1.0',
    r'm1 = 1',
    r'm2 = 1',
    r'mass_ratio = 1',
    r'q = 1',
    r'mu = 1',
    r'star1_mass = total_mass / 2',
    r'assume m1 = m2 = 1'
]

def has_mass_assumption(text):
    if pd.isna(text):
        return False, {}
    pattern_counts = {pattern: 0 for pattern in mass_patterns}
    found_any = False
    for pattern in mass_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        pattern_counts[pattern] = len(matches)
        if len(matches) > 0:
            found_any = True
    return found_any, pattern_counts

# Initialize results dictionary
results = {}

# Get all unique models
all_models = df['model'].unique()
# For demonstration, we'll hardcode the models
all_models = ['gpt-4o-mini-2024-07-18', 'claude-3-5-sonnet-20241022', 'gpt-4o-2024-11-20', 'claude-3-5-haiku-20241022', 'o4-mini']

for model_name in all_models:
    model_df = df[(df['model'] == model_name) & (df['row_wise'] == False)].copy()
    
    # Initialize pattern counts for this model
    model_pattern_counts = {pattern: 0 for pattern in mass_patterns}
    # Filter for scenarios that require mass
    model_df = model_df[model_df['scenario_variation'].apply(
        lambda x: any(s in x for s in mass_required_scenarios)
    )]

    n_mass_problems = len(model_df['scenario_variation'].unique())
    print(f"Model {model_name} has {n_mass_problems} mass problems")
    
    # Group by scenario_variation to handle multiple runs
    grouped = model_df.groupby('scenario_variation', dropna=False)
    n_runs_list = []
    for _, g in grouped:
        n_runs_list.append(len(g))
    
    if len(set(n_runs_list)) > 1:
        print(f"Inconsistent number of runs for {model_name}: {n_runs_list}")
        n_runs = min(n_runs_list)
        print(f"Using first {n_runs} runs")
    else:
        # In case the model has no scenarios or no data
        if len(n_runs_list) == 0:
            results[model_name] = {
                'incorrect_mass_mean': 0,
                'incorrect_mass_se': 0,
                'incorrect_total_mean': 0,
                'incorrect_total_se': 0,
                'correct_mass_mean': 0,
                'correct_mass_se': 0,
                'correct_total_mean': 0,
                'correct_total_se': 0,
                'n_runs': 0,
                'pattern_counts': model_pattern_counts
            }
            continue
        n_runs = n_runs_list[0]
    
    # Initialize arrays to store per-run stats
    incorrect_with_mass = np.zeros(n_runs)
    incorrect_total = np.zeros(n_runs)
    correct_with_mass = np.zeros(n_runs)
    correct_total = np.zeros(n_runs)
    
    # Process each run
    for run_idx in range(n_runs):
        run_data = []
        for _, g in grouped:
            run_row = g.iloc[run_idx:run_idx+1]
            run_data.append(run_row)
        run_df = pd.concat(run_data, axis=0)
        
        for _, row in run_df.iterrows():
            has_mass, pattern_counts = has_mass_assumption(str(row['chat_history']))
            for pattern, count in pattern_counts.items():
                model_pattern_counts[pattern] += count
                
            if row['correct']:
                correct_total[run_idx] += 1
                if has_mass:
                    correct_with_mass[run_idx] += 1
            else:
                incorrect_total[run_idx] += 1
                if has_mass:
                    incorrect_with_mass[run_idx] += 1
    
    # Compute means and standard errors across runs
    def mean_se(values):
        mean_val = values.mean()
        if n_runs > 1:
            se_val = np.sqrt(values.var(ddof=1) / n_runs)
        else:
            se_val = 0.0
        return mean_val, se_val
    
    inc_mass_m, inc_mass_se = mean_se(incorrect_with_mass)
    inc_tot_m, inc_tot_se = mean_se(incorrect_total)
    cor_mass_m, cor_mass_se = mean_se(correct_with_mass)
    cor_tot_m, cor_tot_se = mean_se(correct_total)
    
    results[model_name] = {
        'incorrect_mass_mean': inc_mass_m,
        'incorrect_mass_se': inc_mass_se,
        'incorrect_total_mean': inc_tot_m,
        'incorrect_total_se': inc_tot_se,
        'correct_mass_mean': cor_mass_m,
        'correct_mass_se': cor_mass_se,
        'correct_total_mean': cor_tot_m,
        'correct_total_se': cor_tot_se,
        'n_runs': n_runs,
        'pattern_counts': model_pattern_counts
    }

print("\nMass Assumption Analysis Results")
print("-" * 100)
print(f"{'Model':<30} {'Incorrect':<30} {'Correct':<30} {'Total'}")
print("-" * 100)

for model in results:
    stats = results[model]
    if stats['n_runs'] == 0:
        print(f"{model:<30} No data")
        continue
    
    # We still show the old breakdown in the console for clarity,
    # though the new LaTeX table will only show percentages.
    incorrect_text = (
        f"{stats['incorrect_mass_mean']:.1f} ± {stats['incorrect_mass_se']:.1f}/"
        f"{stats['incorrect_total_mean']:.1f} ± {stats['incorrect_total_se']:.1f} "
        f"({100*stats['incorrect_mass_mean']/stats['incorrect_total_mean'] if stats['incorrect_total_mean']>0 else 0:.1f}%)"
    )
    correct_text = (
        f"{stats['correct_mass_mean']:.1f} ± {stats['correct_mass_se']:.1f}/"
        f"{stats['correct_total_mean']:.1f} ± {stats['correct_total_se']:.1f} "
        f"({100*stats['correct_mass_mean']/stats['correct_total_mean'] if stats['correct_total_mean']>0 else 0:.1f}%)"
    )
    total_mass = stats['incorrect_mass_mean'] + stats['correct_mass_mean']
    total_mass_se = np.sqrt(stats['incorrect_mass_se']**2 + stats['correct_mass_se']**2)
    
    total_resp = stats['incorrect_total_mean'] + stats['correct_total_mean']
    # We'll avoid a divide-by-zero just in case
    total_pct = 100*total_mass/total_resp if total_resp>0 else 0
    # For the console, an approximate combined se for the fraction
    # (using standard ratio-propagation):
    if total_resp > 0:
        ratio = total_mass / total_resp
        # partial derivative approach for ratio error
        # ratio_se^2 = (σX^2 / Y^2) + (X^2 * σY^2 / Y^4)
        ratio_se = np.sqrt(
            (total_mass_se**2)/(total_resp**2) +
            (total_mass**2 * 0.0)/(total_resp**4)  # ignoring separate total_resp SE for console
        )
        total_pct_se = 100*ratio_se
    else:
        total_pct_se = 0
    total_text = (
        f"{total_mass:.1f} ± {total_mass_se:.1f}/"
        f"{total_resp:.1f} "
        f"({total_pct:.1f}±{total_pct_se:.1f}%)"
    )
    
    print(f"{model:<30} {incorrect_text:<30} {correct_text:<30} {total_text}")

print("-" * 100)
print(f"\nResults averaged over {stats['n_runs']} runs with standard errors.")
print("\nPattern Counts by Model:")
print("-" * 100)
print(f"{'Pattern':<50} {'Count':<10}")
print("-" * 100)
for model in results:
    print(f"\nModel: {model}")
    for pattern, count in results[model]['pattern_counts'].items():
        print(f"{pattern:<50} {count:<10}")

def ratio_with_error(x_mean, x_se, y_mean, y_se):
    """
    Computes x_mean/y_mean with error propagation.
    Returns ratio, ratio_se.
    If y_mean == 0, returns (0, 0) to avoid division by zero.
    """
    if y_mean == 0:
        return 0.0, 0.0
    ratio = x_mean / y_mean
    # Standard error propagation for ratio
    ratio_se = np.sqrt(
        (x_se**2)/(y_mean**2) +
        (x_mean**2 * y_se**2)/(y_mean**4)
    )
    return ratio, ratio_se

latex_table = (
    r"""
\begin{table*}[!htb]
    \centering
    \footnotesize
    \caption{\textbf{Analysis of Mass-Related Assumptions in Model Responses.} 
    For each model, we analyze responses to problems that explicitly require determining stellar masses as an intermediate step or final answer (""" +
    str(n_mass_problems) +
    r""" problems). 
    The percentages show what fraction of each category (incorrect/correct) contained explicit mass assumptions. 
    Mass assumptions were identified by searching for the following patterns: center-of-mass calculation ``(df['star1\_x'] + df['star2\_x'])/2'', 
    explicit mass assignments (``star1\_mass = 1.0'', ``star2\_mass = 1.0'', ``m1 = m2'', ``m1 = 1.0'', ``m2 = 1.0''), 
    and other variations of unit mass assumptions. All matches were manually verified. 
    Results averaged over """ +
    str(n_runs) +
    r""" runs with standard errors shown.}\label{tab:mass_assumptions}
    \begin{tabular}{lcc}
    \toprule
    & \textbf{\% of Incorrect Solutions} & \textbf{\% of Correct Solutions} \\
    & \textbf{that include a mass assumption} & \textbf{that include a mass assumption} \\
    \midrule
"""
)

for model in results:
    stats = results[model]
    n_runs = stats['n_runs']
    if n_runs == 0:
        # If there's no data for that model
        latex_table += f"    {model} & -- & -- \\\\\n"
        continue

    # Incorrect fraction and SE
    inc_ratio, inc_ratio_se = ratio_with_error(
        stats['incorrect_mass_mean'], stats['incorrect_mass_se'],
        stats['incorrect_total_mean'], stats['incorrect_total_se']
    )
    inc_percent = 100.0 * inc_ratio
    inc_percent_se = 100.0 * inc_ratio_se

    # Correct fraction and SE
    cor_ratio, cor_ratio_se = ratio_with_error(
        stats['correct_mass_mean'], stats['correct_mass_se'],
        stats['correct_total_mean'], stats['correct_total_se']
    )
    cor_percent = 100.0 * cor_ratio
    cor_percent_se = 100.0 * cor_ratio_se

    incorrect_str = f"{inc_percent:.1f} $\\pm$ {inc_percent_se:.1f} ({int(stats['incorrect_mass_mean'])}/{int(stats['incorrect_total_mean'])})"
    correct_str = f"{cor_percent:.1f} $\\pm$ {cor_percent_se:.1f} ({int(stats['correct_mass_mean'])}/{int(stats['correct_total_mean'])})"

    latex_table += f"    {model} & {incorrect_str} & {correct_str} \\\\\n"

latex_table += r"""    \bottomrule
    \end{tabular}
\end{table*}
"""

# Write the new LaTeX table to file
with open('analysis/tables/table2_massassumption.tex', 'w') as f:
    f.write(latex_table)

print("\nGenerated a new LaTeX table (tables/table2.tex) that shows only percentages ± errors.")