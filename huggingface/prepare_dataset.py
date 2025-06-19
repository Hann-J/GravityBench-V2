# File: hf_dataset_files/phys_discovery_simple.py (or at the root of your HF dataset repo)
import json
import datasets

_CITATION = """\
@misc{koblischke2025gravitybenchv1benchmarkgravitationalphysics,
      title={Gravity-Bench-v1: A Benchmark on Gravitational Physics Discovery for Agents}, 
      author={Nolan Koblischke and Hyunseok Jang and Kristen Menou and Mohamad Ali-Dib},
      year={2025},
      eprint={2501.18411},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2501.18411}, 
}
"""

_DESCRIPTION = """\
A dataset for evaluating AI agents on physics discovery tasks in simulated binary star systems.
Each instance provides a task prompt, simulation data (as a CSV string), the true answer,
and parameters for evaluating correctness.

The dataset includes two splits:
- 'validation': Contains all metadata but CSV data is replaced with '[CSV data omitted for preview]' 
  to enable the HuggingFace dataset viewer
- 'test': Contains the full dataset with complete simulation CSV data for actual use
"""

_HOMEPAGE = "https://huggingface.co/datasets/GravityBench/GravityBench"

_LICENSE = "apache-2.0"

class GravityBenchSimpleConfig(datasets.BuilderConfig):
    """BuilderConfig for GravityBenchSimple."""
    def __init__(self, **kwargs):
        super(GravityBenchSimpleConfig, self).__init__(**kwargs)

class GravityBenchSimple(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("1.0.0")
    DEFAULT_CONFIG_NAME = "default"

    def _info(self):
        features = datasets.Features({
            "scenario_id": datasets.Value("string"),
            "scenario_name": datasets.Value("string"),
            "variation_name": datasets.Value("string"),
            "task_prompt": datasets.Value("string"),
            "expected_units": datasets.Value("string"),
            "true_answer": datasets.Value("string"),
            "full_obs_threshold_percent": datasets.Value("float32"),
            "budget_obs_threshold_percent": datasets.Value("float32"),
            "simulation_csv_filename": datasets.Value("string"),
            "simulation_csv_content": datasets.Value("string"),
        })
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"filepath": "test.jsonl"},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"filepath": "validation.jsonl"},
            ),
        ]

    def _generate_examples(self, filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            for id_, row in enumerate(f):
                data = json.loads(row)
                yield id_, data