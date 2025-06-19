def submit_answer_tool():
    """Create a submit answer tool definition for function calling."""
    return {
        "type": "function",
        "function": {
            "name": "submit_answer",
            "description": "Use to submit the answer. Must be a number for quantitative answers or boolean True/False for non-quantitative answers. Do not submit a string.",
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {
                        "type": ["number", "boolean"],
                        "description": "The answer to submit. Must be a number for quantitative answers or boolean True/False for non-quantitative answers. Do not submit a string."
                    }
                },
                "required": ["answer"]
            }
        }
    }

def execute_submit_answer(answer):
    """Execute the submit answer tool with the given parameters."""
    # Handle dictionary input first
    if isinstance(answer, dict):
        try:
            return execute_submit_answer(answer['answer'])  # Recursively process the extracted value
        except KeyError:
            raise ValueError("Answer must be a number for quantitative answers or boolean True/False for non-quantitative answers. Do not submit a string.")
        
    # Then handle string conversion
    if isinstance(answer, str):
        try:
            answer = float(answer)
        except ValueError:
            if answer.lower() == "true":
                answer = True
            elif answer.lower() == "false":
                answer = False
            else:
                raise ValueError("Answer cannot be a string. Must be a number for quantitative answers or boolean True/False for non-quantitative answers.")
    
    return answer