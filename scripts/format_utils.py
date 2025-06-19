from typing import Dict, Any
import numpy as np

def convert_result_to_json(result: Dict[str, Any]) -> Dict[str, Any]:
    
    # Extract intermediate steps
    intermediate_steps = []
    for step in result['intermediate_steps']:
        tool_info = {
            "tool": step[0].tool,
            "tool_input": step[0].tool_input,
            "message_log": [{"content": msg.content} for msg in step[0].message_log] if hasattr(step[0], 'message_log') else None
        }
        tool_output = step[1]
        intermediate_steps.append({"tool_info": tool_info, "tool_output": tool_output})
    
    # Construct the final dictionary
    json_data = {
        "input": result['input'],
        "output": result['output'],
        "intermediate_steps": intermediate_steps,
        'input_tokens_used': result['input_tokens_used'],
        'output_tokens_used': result['output_tokens_used'],
        'error_message': result['error_message'],
    }
    
    return json_data


from html import escape
import json

def safe_escape(s):
    """Safely escape a value that might be a string or another type."""
    if isinstance(s, str):
        return escape(s)
    return escape(str(s))

def format_scientific(value):
    """Format a number in scientific notation with 3 decimal places."""
    return f"{value:.2e}" if value is not None else "None"

def json_to_html(data):
    # Calculate correct predictions ratio and percentage
    # Get unique scenario-variation pairs by creating tuples and using set
    unique_pairs = {(scenario['scenario_name'], scenario['variation_name']) 
                   for scenario in data['scenarios']}
    total_scenarios = len(unique_pairs)
    # Group by scenario-variation pairs and calculate average correct value
    scenario_var_results = {}
    for scenario in data['scenarios']:
        if scenario['correct'] is not None:  # Only count if True or False
            key = (scenario['scenario_name'], scenario['variation_name'])
            if key not in scenario_var_results:
                scenario_var_results[key] = []
            scenario_var_results[key].append(1 if scenario['correct'] else 0)
    
    # Calculate average correct for each scenario-variation pair and sum
    correct_predictions = sum(
        sum(results) / len(results)
        for results in scenario_var_results.values()
    )
    # correct_predictions = sum(1 for scenario in data['scenarios'] if scenario['correct'])
    if total_scenarios > 0:
        correct_ratio = f"{correct_predictions:.1f}/{total_scenarios}"
        correct_percentage = f"{(correct_predictions / total_scenarios) * 100:.1f}%"
    else:
        correct_ratio = "0/0"
        correct_percentage = "0.00%"

    # Get the model name and other parameters (assuming all scenarios use the same model)
    model_name = data['scenarios'][0]['model'] if data['scenarios'] else "Unknown Model"
    row_wise = data['scenarios'][0]['row_wise'] if data['scenarios'] else 'Unknown'
    max_observations_total = data['scenarios'][0]['max_observations_total'] if data['scenarios'] else 'Unknown'
    max_observations_per_request = data['scenarios'][0]['max_observations_per_request'] if data['scenarios'] else 'Unknown'
    temperature = data['scenarios'][0]['temperature'] if data['scenarios'] else 'Unknown'

    # Calculate total cost
    total_cost = sum(scenario['cost'] for scenario in data['scenarios'])

    scenarios_with_none_result = [
        scenario["scenario_name"] + f' ({scenario["variation_name"]})' + f" (Run {scenario['attempt']})"
        for scenario in data["scenarios"]
        if scenario["result"] is None and scenario["attempt"] == max(scn["attempt"] for scn in data["scenarios"] if scn["scenario_name"] == scenario["scenario_name"])
    ]

    # Generate scenario links for the directory
    previous_scenario_name = None
    previous_variation_name = None
    scenario_links = ""

    for i, scenario in enumerate(data['scenarios']):
        current_scenario_name = scenario["scenario_name"]
        current_variation_name = scenario["variation_name"]
        current_attempt = scenario["attempt"]
        
        # Add pass/fail indicator
        status_indicator = 'N/A' if scenario["correct"] is None else 'P' if scenario["correct"] else 'X'
        
        scenario_name_html = f'{current_scenario_name}<br>' if current_scenario_name != previous_scenario_name else ''
        variation_html = f'<span class="tab-indent">{current_variation_name}</span>' if current_variation_name != previous_variation_name or current_attempt == 1 else ''
        attempt_html = f'<span class="tab-indent">(Run {current_attempt})</span>' if current_attempt > 1 else ''
        
        scenario_links += (
            f'<li><a href="#scenario-{i}">'
            f'{scenario_name_html}'
            f'{variation_html} {attempt_html} ({status_indicator})'
            f'</a></li>'
        )
        
        previous_scenario_name = current_scenario_name
        previous_variation_name = current_variation_name
    # HTML template
    # Within your existing HTML template, update the <style> and <script> sections, and add the "Expand All" button in the sidebar:
    html_template = '''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Run Results</title>
        <link href="https://cdnjs.cloudflare.com/ajax/libs/prism/1.24.1/themes/prism.min.css" rel="stylesheet" />
        <style>
            body {{
                font-family: Arial, sans-serif;
                line-height: 1.6;
                color: #333;
                margin: 0;
                padding: 0;
                display: flex;
            }}
            .sidebar {{
                width: 400px;
                height: 100vh;
                overflow-y: auto;
                position: fixed;
                background-color: #f4f4f4;
                padding: 20px;
                box-sizing: border-box;
            }}
            .main-content {{
                margin-left: 400px;
                padding: 20px;
                max-width: 800px;
            }}
            h1, h2, h3 {{
                color: #2c3e50;
            }}
            pre {{
                background-color: #f4f4f4;
                border: 1px solid #ddd;
                border-radius: 4px;
                padding: 10px;
                overflow-x: auto;
                white-space: pre-wrap;       /* CSS3 */
                white-space: -moz-pre-wrap;  /* Firefox */
                white-space: -pre-wrap;      /* Opera <7 */
                white-space: -o-pre-wrap;    /* Opera 7 */
                word-wrap: break-word;       /* IE */
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin-bottom: 20px;
                text-align: left;
            }}
            th, td {{
                border: 1px solid #ddd;
                padding: 8px;
            }}
            th {{
                background-color: #f2f2f2;
            }}
            td.attribute {{
                font-weight: bold;
            }}
            .tool-info {{
                background-color: #fff8dc;
                padding: 10px;
                margin-bottom: 10px;
                border: 1px solid #e6d9a5;
                border-radius: 4px;
            }}
            .tool-output {{
                background-color: #f0fff0;
                padding: 10px;
                margin-bottom: 10px;
                border: 1px solid #98fb98;
                border-radius: 4px;
            }}
            .summary {{
                background-color: #e6f3ff;
                padding: 15px;
                margin-bottom: 20px;
                border-radius: 4px;
            }}
            .collapsible {{
                background-color: #f9f9f9;
                border: none;
                color: #444;
                cursor: pointer;
                padding: 10px;
                width: 100%;
                text-align: left;
                outline: none;
                font-size: 18px;
                font-weight: bold;
                text-decoration: underline;
            }}
            .active, .collapsible:hover {{
                background-color: #ddd;
            }}
            .collapsible::after {{
                content: '\\25BC'; /* Downward pointing triangle */
                font-size: 12px;
                color: #777;
                float: left;
                margin-left: 5px;
                margin-right: 5px;
            }}
            .collapsible.active::after {{
                content: '\\25B2'; /* Upward pointing triangle */
            }}
            .content {{
                padding: 0 18px;
                display: none;
                overflow: hidden;
                background-color: #f9f9f9;
            }}
            hr {{
                border: none;
                height: 3px; /* Adjust the thickness as needed */
                background-color: #333; /* Adjust the color as needed */
                margin: 20px 0; /* Adjust the spacing as needed */
            }}
            .tab-indent {{
                display: inline-block;
                margin-left: 1em; /* Adjust as needed */
            }}
            li {{
                list-style-type: none; /* Remove default list bullet */
            }}
            li a {{
                text-decoration: none; /* Remove underline */
            }}
            .filter {{
                margin-top: 10px;
                margin-bottom: 10px;
                display: flex;
                flex-wrap: wrap;
                align-items: center;
                font-size: 0.9em;
            }}
            .filter label {{
                margin-right: 10px;
                display: flex;
                align-items: center;
            }}
            .filter input[type="radio"] {{
                margin-right: 3px;
            }}
        </style>
    </head>
    <body>
        <div class="sidebar">
            <h2>Scenarios</h2>
                <li><a href="#top" style="">Go to top</a></li>
            <button id="expand-all">Expand All Steps</button>
            <div class="filter">
                <label><input type="radio" name="filter" value="all" checked> All</label>
                <label><input type="radio" name="filter" value="passed"> Passed</label>
                <label><input type="radio" name="filter" value="failed"> Failed</label>
                <label><input type="radio" name="filter" value="error"> Error</label>
            </div>
            <ul>
                {scenario_links}
            </ul>
        </div>
        <div class="main-content">
            <h1 id="top">Run Summary</h1>
            <div class="summary">
                <h2>Summary</h2>
                <p><strong>Model:</strong> {model_name}</p>
                {row_wise_info}
                <p><strong>Total Run Time (all runs):</strong> {total_run_time:.2f} minutes</p>
                <p><strong>Total Input Tokens Used (all runs):</strong> {input_tokens_used}</p>
                <p><strong>Total Output Tokens Used (all runs):</strong> {output_tokens_used}</p>
                <p><strong>Total Cost (all runs):</strong> ${total_cost:.4f}</p>
                <p><strong>Correct Predictions (avg across runs):</strong> {correct_ratio} ({correct_percentage})</p>
                {if_scenario_has_none_results}
            </div>
            {content}
        </div>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.24.1/components/prism-core.min.js"></script>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.24.1/plugins/autoloader/prism-autoloader.min.js"></script>
        <script>
            document.addEventListener('DOMContentLoaded', (event) => {{
                const coll = document.getElementsByClassName('collapsible');
                for (let i = 0; i < coll.length; i++) {{
                    coll[i].addEventListener('click', function() {{
                        this.classList.toggle('active');
                        const content = this.nextElementSibling;
                        if (content.style.display === 'block') {{
                            content.style.display = 'none';
                        }} else {{
                            content.style.display = 'block';
                        }}
                    }});
                }}
                document.getElementById('expand-all').addEventListener('click', () => {{
                    for (let i = 0; i < coll.length; i++) {{
                        coll[i].classList.add('active');
                        coll[i].nextElementSibling.style.display = 'block';
                    }}
                }});

                // Add filter functionality
                const filterRadios = document.querySelectorAll('input[name="filter"]');
                filterRadios.forEach(radio => {{
                    radio.addEventListener('change', function() {{
                        const filter = this.value;
                        const scenarioLinks = document.querySelectorAll('.sidebar ul li a');
                        scenarioLinks.forEach(link => {{
                            const status = link.textContent.slice(-2, -1); // Get the status indicator
                            if (filter === 'all' || 
                                (filter === 'passed' && status === 'P') ||
                                (filter === 'failed' && status === 'X') ||
                                (filter === 'error' && link.textContent.slice(-4, -1) === 'N/A')) {{
                                link.parentElement.style.display = 'block';
                            }} else {{
                                link.parentElement.style.display = 'none';
                            }}
                        }});
                    }});
                }});
            }});
        </script>
    </body>
    </html>
    '''
    content = ""
    for i, scenario in enumerate(data['scenarios']):
        try:
            content += f'<h2 id="scenario-{i}">{safe_escape(scenario.get("scenario_name", "Unknown"))} {safe_escape(scenario.get("variation_name", "Unknown"))}</h2>'
            
            # Handle units - only display if not None
            units = scenario.get('units', '')
            units_display = f" {safe_escape(units)}" if units is not None and units != '' else ""
            
            content += '''
            <table>
            <tr>
                <td class="attribute">Agent Answer</td>
                <td>{result}{final_answer_units}</td>
            </tr>
            <tr>
                <td class="attribute">True Answer</td>
                <td>{true_answer}{final_answer_units}</td>
            </tr>
            <tr>
                <td class="attribute">Percent Error</td>
                <td>{percent_error}</td>
            </tr>
            <tr>
                <td class="attribute">Within {threshold}%?</td>
                <td>{correct}</td>
            </tr>
            <tr>
                <td class="attribute">Run Time</td>
                <td>{run_time}</td>
            </tr>
            <tr>
                <td class="attribute">Input Tokens Used</td>
                <td>{input_tokens_used}</td>
            </tr>
            <tr>
                <td class="attribute">Output Tokens Used</td>
                <td>{output_tokens_used}</td>
            </tr>
            <tr>
                <td class="attribute">Run</td>
                <td>{attempt}</td>
            </tr>
            <tr>
                <td class="attribute">Variation Name</td>
                <td>{variation_name}</td>
            </tr>
            <tr>
                <td class="attribute">Cost</td>
                <td>${cost:.4f}</td>
            </tr>
            {observations_attempted_row}
            </table>
            '''.format(
                result=scenario.get('result', 'N/A') if isinstance(scenario.get('result'), (bool, str)) 
                    else safe_escape(format_scientific(scenario.get('result'))),
                true_answer=scenario.get('true_answer', 'N/A') if isinstance(scenario.get('true_answer'), (bool, str)) 
                    else safe_escape(format_scientific(scenario.get('true_answer'))),
                final_answer_units=units_display,
                threshold=safe_escape(scenario.get('threshold_used', 'N/A')),
                correct=safe_escape(scenario.get('correct', 'N/A')),
                percent_error=safe_escape(f"{scenario.get('percent_error', 0) * 100 :.1f}") + '%' 
                    if scenario.get('percent_error') is not None else 'N/A',
                run_time=safe_escape(round(scenario.get('run_time', 0))) + ' seconds',
                input_tokens_used=safe_escape(scenario.get('input_tokens_used', 0)),
                output_tokens_used=safe_escape(scenario.get('output_tokens_used', 0)),
                attempt=safe_escape(scenario.get('attempt', 'N/A')),
                variation_name=safe_escape(scenario.get('variation_name', 'N/A')),
                cost=scenario.get('cost', 0.0),
                observations_attempted_row=f'<tr><td class="attribute">Number of Observations Attempted</td><td>{scenario.get("observations_attempted", "N/A")}/{scenario.get("max_observations_total", "N/A")} (Observational Budget)</td></tr>' 
                    if scenario.get('row_wise') and 'observations_attempted' in scenario else ''
            )

            if scenario.get('error_message'):
                content += f'<p><strong>Error:</strong> {safe_escape(scenario["error_message"])}</p>'

            # Process chat history if it exists
            chat_history = scenario.get('chat_history', {})
            if chat_history:
                input_text = chat_history.get('input', '')
                if input_text:
                    content += f"<h3>Input</h3><pre>{safe_escape(input_text)}</pre>"
                
                agent_output = chat_history.get('output', '')
                if isinstance(agent_output, list):
                    agent_output = agent_output[0].get('text', '')
                if agent_output:
                    content += f"<h3>Output</h3><pre>{safe_escape(agent_output)}</pre>"

                if 'intermediate_steps' in chat_history:
                    content += '''
                    <button type="button" class="collapsible">Intermediate Steps</button>
                    <div class="content">
                    '''

                    for j, step in enumerate(chat_history['intermediate_steps'], 1):
                        tool_info = step.get('tool_info', {})
                        tool_input = tool_info.get('tool_input', '')
                        formatted_input = tool_input.get('input_code', str(tool_input)) if isinstance(tool_input, dict) else str(tool_input)
                        formatted_input = formatted_input.replace('; ', ';\n')
                        
                        content += f'''
                        <h4>Step {j}</h4>
                        <div class="tool-info">
                        '''
                        
                        if tool_info.get('message_log'):
                            try:
                                message_lines = []
                                for msg in tool_info['message_log']:
                                    msg_content = msg.get('content', '')
                                    if isinstance(msg_content, list):
                                        message_lines.extend(
                                            part['text'] 
                                            for part in msg_content 
                                            if isinstance(part, dict) and part.get('type') == 'text'
                                        )
                                    else:
                                        message_lines.append(str(msg_content))
                                if message_lines:
                                    content += f'<strong>Agent Message:</strong><pre>{safe_escape(chr(10).join(message_lines))}</pre>'
                            except Exception as e:
                                print(f"Error processing message log: {e}")
                                print(tool_info['message_log'])
                        
                        content += f'''
                            <strong>Tool:</strong> {safe_escape(tool_info.get('tool', 'Unknown'))}<br>
                            <strong>Tool Input:</strong><br>
                            <pre><code class="language-python">{safe_escape(formatted_input)}</code></pre>
                        </div>
                        <div class="tool-output">
                            <strong>Tool Output:</strong>
                            <pre>{safe_escape(step.get('tool_output', ''))}</pre>
                        </div>
                        '''
                    content += '</div>'
            content += '<hr>'
        except Exception as e:
            print(f"Error processing scenario {i}: {e}")
            continue

    # Generate final HTML
    html_content = html_template.format(
        scenario_links=scenario_links,
        model_name=model_name,
        row_wise_info=f'<p><strong>Budget-obs</strong></p>'
                      f'<p><strong>Max observations total:</strong> {max_observations_total}</p>'
                      f'<p><strong>Max observations per request:</strong> {max_observations_per_request}</p>' if row_wise else '<p><strong>Full-obs</strong></p>',
        temperature=temperature,
        if_scenario_has_none_results=f'<p><strong>Agent failed to answer for scenarios:</strong> {", ".join(scenarios_with_none_result)}</p>' if scenarios_with_none_result else '',
        total_run_time=sum(scenario['run_time'] for scenario in data['scenarios'])/60,
        input_tokens_used=sum(scenario['input_tokens_used'] for scenario in data['scenarios']),
        output_tokens_used=sum(scenario['output_tokens_used'] for scenario in data['scenarios']),
        correct_ratio=correct_ratio,
        correct_percentage=correct_percentage,
        total_cost=total_cost,
        content=content
    )
    return html_content

def string_to_variable(var):
    if isinstance(var, (bool, np.bool_)):
        return bool(var)
    elif isinstance(var, str):
        if var == "True":
            return True
        elif var == "False":
            return False
        try:
            return float(var)
        except ValueError:
            return None
    elif isinstance(var, (int, float)):
        return float(var)
    return var
