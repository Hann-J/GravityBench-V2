import json
from io import StringIO
import pandas as pd, numpy as np
import openai, datasets
from dotenv import load_dotenv

from agents.tools.observe_tool import observe_tool
from agents.tools.python_repl_tool import python_repl_tool, execute_python_repl
from agents.tools.submit_answer_tool import submit_answer_tool, execute_submit_answer

load_dotenv()
class BinarySim:
    def __init__(self, csv, task, units, truth):
        self.df = pd.read_csv(StringIO(csv))
        self.truth = float(truth)
        self.req = 0  # total observations used
        t0, t1, n = self.df.time.min(), self.df.time.max(), len(self.df)
        self.prompt = (
            f"You are a physics discovery agent. Your task is: {task}\n\n"
            f"Dataset spans {t0:.2f}–{t1:.2f} with {n} rows.\n"
            "Use Observe(times_requested) to sample <=100 rows (<=10 per call).\n"
            "Analyse with PythonREPL where 'row_wise_results' stores your samples.\n"
            f"Expected answer units: {units}. Submit with submit_answer(answer)."
        )

    def observe(self, times, limit):
        if len(times) > limit:
            return f"Error: ask <={limit} rows"
        self.req += len(times)
        rows = [self.df.iloc[(self.df.time - t).abs().idxmin()] for t in times]
        return pd.DataFrame(rows)

class Agent:
    def __init__(self, sim, model="gpt-4.1", per_req=10):
        self.sim, self.per_req = sim, per_req
        self.row_wise_results = pd.DataFrame()
        self.pkgs = dict(
            np=np, pd=pd, scipy=__import__('scipy'), sklearn=__import__('sklearn'),
            sm=__import__('statsmodels.api', fromlist=['api']),
            row_wise_results=self.row_wise_results,
        )
        self.tools = [
            observe_tool(per_req, metadata={"env": sim}),
            python_repl_tool(self.pkgs, self.pkgs, "numpy pandas scipy sklearn statsmodels"),
            submit_answer_tool(),
        ]
        self.client = openai.OpenAI()
        self.msg = [
            {"role": "system", "content": sim.prompt},
            {"role": "user", "content": "Begin your analysis."},
        ]

    # keep observations deduplicated and expose to REPL
    def add_obs(self, df):
        self.row_wise_results = (
            pd.concat([self.row_wise_results, df]).drop_duplicates("time").reset_index(drop=True)
        )
        self.pkgs["row_wise_results"] = self.row_wise_results  # refresh pointer

    def run(self):
        for _ in range(20):
            rsp = self.client.chat.completions.create(
                model="gpt-4.1", messages=self.msg, tools=self.tools
            ).choices[0].message
            if rsp.content: print(f"Content: {rsp.content}")
            self.msg.append({"role": "assistant", "content": rsp.content, "tool_calls": rsp.tool_calls})
            if not rsp.tool_calls:
                print("No tool calls, continuing...")
                continue
            for call in rsp.tool_calls:
                args = json.loads(call.function.arguments)
                name = call.function.name
                if name == "Observe":
                    print(f"Observe args: {args['times_requested']}")
                    res = self.sim.observe(args["times_requested"], self.per_req)
                    print(f"Observe result: {res}\n\n")
                    self.add_obs(res)
                elif name == "PythonREPL":
                    print(f"PythonREPL args: {args['input_code']}")
                    res = execute_python_repl(args["input_code"], self.pkgs, self.pkgs)
                    print(f"PythonREPL result: {res}\n\n")
                else:
                    print(f"submit_answer args: {args['answer']}")
                    return execute_submit_answer(args["answer"])
                self.msg.append({"role": "tool", "content": str(res), "tool_call_id": call.id})
        return None

def main():
    data = datasets.load_dataset("GravityBench/GravityBench")["test"][0]
    sim = BinarySim(data['simulation_csv_content'], data['task_prompt'], data['expected_units'], data['true_answer'])
    agent = Agent(sim)
    ans = agent.run()
    print("Ans:", ans, "Truth:", sim.truth, f"Obs {sim.req}/100")

    err = abs(float(ans) - sim.truth) / sim.truth * 100
    print(f"Error {err:.2f}% (threshold {data['budget_obs_threshold_percent']}%)")

if __name__ == "__main__":
    main()
