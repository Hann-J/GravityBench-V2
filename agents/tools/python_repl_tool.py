import functools
import logging
import multiprocessing
import re
import sys
import traceback
from io import StringIO
import json

logger = logging.getLogger(__name__)

@functools.lru_cache(maxsize=None)
def warn_once() -> None:
    """Warn once about the dangers of PythonREPL."""
    logger.warning("Python REPL can execute arbitrary code. Use with caution.")

def python_repl_tool(_globals, _locals, package_names):
    """Create a Python REPL tool definition for function calling."""
    return {
        "type": "function",
        "function": {
            "name": "PythonREPL",
            "description": f"A Python REPL. Use this to execute python code. Input should be a valid python command. If you want to see the output of a value, you should print it out with `print(...)`. You cannot use matplotlib. No plotting is allowed.\nPackages you can import: {package_names}.",
            "parameters": {
                "type": "object",
                "properties": {
                    "input_code": {
                        "type": "string",
                        "description": "A valid python command."
                    }
                },
                "required": ["input_code"]
            }
        }
    }

def worker(command: str, namespace: dict, queue: multiprocessing.Queue) -> None:
    """Worker function to execute Python code in a separate process."""
    old_stdout = sys.stdout
    sys.stdout = mystdout = StringIO()
    try:
        # Sanitize input
        cleaned_command = sanitize_input(command)
        cleaned_command = wrap_last_line_with_print(cleaned_command)

        exec(cleaned_command, namespace)
        sys.stdout = old_stdout
        queue.put(mystdout.getvalue())
    except Exception as e:
        sys.stdout = old_stdout
        tb = traceback.extract_tb(sys.exc_info()[2])
        user_tb = [frame for frame in tb if frame.filename == "<string>"]
        tb_str = "Error Traceback:\n"
        
        command_lines = cleaned_command.split('\n')
        
        for frame in user_tb:
            tb_str += f'  line {frame.lineno}:\n'
            if 0 < frame.lineno <= len(command_lines):
                tb_str += f'    {command_lines[frame.lineno - 1].strip()}\n'
        tb_str += f'{type(e).__name__}: {str(e)}'
        queue.put(f"{tb_str}")

def sanitize_input(query: str) -> str:
    """Sanitize input to the Python REPL."""
    # Remove any leading/trailing backticks and 'python' prefix
    query = re.sub(r"^(\s|`)*(?i:python)?\s*", "", query)
    query = re.sub(r"(\s|`)*$", "", query)
    
    # Process the string character by character
    result = []
    i = 0
    in_string = False
    string_char = None  # Keep track of the string delimiter (' or ")
    
    while i < len(query):
        if not in_string:
            # Outside string
            if query[i] in '"\'':
                in_string = True
                string_char = query[i]
                result.append(query[i])
            elif i < len(query) - 1 and query[i:i+2] == '\\n':
                result.append('\n')
                i += 1  # Skip the next character
            else:
                result.append(query[i])
        else:
            # Inside string
            if query[i] == '\\' and i + 1 < len(query):
                # Handle escape sequences inside strings
                result.append(query[i:i+2])
                i += 1  # Skip the next character
            elif query[i] == string_char:
                # End of string
                in_string = False
                result.append(query[i])
            else:
                result.append(query[i])
        i += 1
    
    return ''.join(result)

def wrap_last_line_with_print(code: str) -> str:
    """If last line of code is a single word then wrap it in print."""
    lines = code.strip().split('\n')
    last_line = lines[-1].strip()
    if re.match(r'^[^\s,()]+$', last_line):
        lines[-1] = f'print({last_line})'
    return '\n'.join(lines)

def execute_python_repl(input_code: str, _globals: dict, _locals: dict, timeout: int = None) -> str:
    """Execute Python code in the REPL."""
    warn_once()
    if 'matplotlib' in input_code:
        return "No plotting is allowed. Code was not executed since it contained 'matplotlib'."
    
    namespace = {**_globals, **_locals}
    queue = multiprocessing.Queue()
    
    if timeout is not None:
        p = multiprocessing.Process(target=worker, args=(input_code, namespace, queue))
        p.start()
        p.join(timeout)
        if p.is_alive():
            p.terminate()
            return "Execution timed out"
    else:
        worker(input_code, namespace, queue)
    
    result = queue.get()
    
    _globals.update({k: v for k, v in namespace.items() if k not in _locals})
    _locals.update({k: v for k, v in namespace.items() if k in _locals})
    
    if len(result) > 5000:
        result = result[:5000] + "...(output truncated)"
    if len(result) == 0:
        result = "No output. You likely forgot to print the result. Please use `print(...)` to see any output."
    return result