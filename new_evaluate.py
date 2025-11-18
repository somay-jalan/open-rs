import os
from datetime import timedelta
from lighteval.logging.evaluation_tracker import EvaluationTracker
from lighteval.models.vllm.vllm_model import VLLMModelConfig
from lighteval.pipeline import ParallelismManager, Pipeline, PipelineParameters
from lighteval.utils.imports import is_package_available

if is_package_available("accelerate"):
    from accelerate import Accelerator, InitProcessGroupKwargs
    accelerator = Accelerator(kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(seconds=3000))])
else:
    accelerator = None


# ----------------------
# Experiment Config
# ----------------------
TASKS = ["aime24", "math_500", "amc23"]

REVISIONS = {
    1: {100: "0439c48c3686728b5cc5a20820d601d2908c6ee1",
        200: "26f1b83b6c41b2c1b6407648b0de08c09f92687c",
        300: "52bf539b00ab80ff4f9b09092b242d56560e7259",
        400: "998f5ed67e0457c504c49580d85d82de1bcb6f35"},
    2: {50: "21e477217641fec6cd81f9caa220f5cccfb6a874",
        100: "f653a8b948a6047011ee97b181ae0ad1d78d6e6a",
        150: "e25175ada88bb8ef4187480b35631bf397196538",
        200: "2cb5c97863c5936d88b571b4b7eaf7cbbaf5af2f",
        250: "9d6ebc12f8d98824b3a072a930260a62f9189285",
        300: "5e24d1cabac521742d1e6ebd244af9fc2fa91a89",
        350: "195beba1db4d7a3539c0dc0223df9ce2486744d8",
        400: "f3f07cd9f9b2abd8f8760dc4bef5b7165a940ffe"},
    3: {50: "1793695d9e827f1024422af3698f9a070b935698",
        100: "b224d0ff3f66ced97e621b2ad1eeafd53235240c",
        150: "8c574c77d411bd3d39107a03d4f79db3838c5577",
        200: "d91ed4d06af444d7213f8fc225a01c9d43e1ce18",
        250: "7f3c01cb25ca20730f85a079787d859fd5c2be8f",
        300: "6f19980625be4c3316c6919ef12bd168a97c7cf4",
        350: "86ba8acb918d462e6a26f6866f573ded4afdde4a",
        400: "e88c97c2c4035abf0e00091ce214d055aad2c270"},
}

STEPS = {
    1: [100, 200, 300, 400],
    2: [50, 100, 150, 200, 250, 300, 350, 400],
    3: [50, 100, 150, 200, 250, 300, 350, 400],
}


# ----------------------
# Run Evaluation
# ----------------------
def run_experiment(exp_num):
    if exp_num not in REVISIONS:
        print(f"Error: Experiment {exp_num} not defined")
        return

    for step in STEPS[exp_num]:
        revision = REVISIONS[exp_num][step]
        output_dir = f"logs/evals/Exp{exp_num}_{step}"
        os.makedirs(output_dir, exist_ok=True)

        model_config = VLLMModelConfig(
            model_name="data/OpenRS-GRPO",
            dtype="bfloat16",
            revision=revision
        )

        evaluation_tracker = EvaluationTracker(
            output_dir=output_dir,
            save_details=True,
            push_to_hub=True,
            hub_results_org="your_username"
        )

        pipeline_params = PipelineParameters(
            launcher_type=ParallelismManager.ACCELERATE,
            max_samples=10,
            custom_tasks_directory=None
        )

        print(f"Running Experiment {exp_num}, Step {step}, Revision {revision}")
        for task in TASKS:
            print(f"Evaluating task: {task}")
            pipeline = Pipeline(
                tasks=f"custom|{task}|0|0",
                pipeline_parameters=pipeline_params,
                evaluation_tracker=evaluation_tracker,
                model_config=model_config
            )
            pipeline.evaluate()
            pipeline.save_and_push_results()
            pipeline.show_results()


# ----------------------
# Main
# ----------------------
def main():
    import sys
    if len(sys.argv) < 2:
        print("Usage: python evaluate.py <experiment_number1> [experiment_number2 ...]")
        print("Available experiments: 1, 2, 3")
        return

    for exp in sys.argv[1:]:
        try:
            exp_num = int(exp)
            run_experiment(exp_num)
        except ValueError:
            print(f"Invalid experiment number: {exp}")


if __name__ == "__main__":
    main()
