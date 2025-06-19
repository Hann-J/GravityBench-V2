def observe_tool(maximum_observations_per_request, metadata):
    """Create an observe tool definition for function calling."""
    return {
        "type": "function",
        "function": {
            "name": "Observe",
            "description": f"Observe the system at a list of times. Maximum length of times_requested (max number of observations per request): {maximum_observations_per_request}.",
            "parameters": {
                "type": "object",
                "properties": {
                    "times_requested": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Observation time or list of observation times."
                    }
                },
                "required": ["times_requested"]
            }
        }
    }

def execute_observe_tool(times_requested, environment, maximum_observations_per_request):
    """Execute the observe tool with the given parameters."""
    return str(environment.binary_sim.observe_row(times_requested, maximum_observations_per_request))