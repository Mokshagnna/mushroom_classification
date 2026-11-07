import argparse

from .inference import main as inference_main
from .pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Mushroom classification MLOps workflow.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("train", help="Validate data, compare models, train, evaluate, and register.")
    subparsers.add_parser("predict", help="Run batch inference using the saved end-to-end pipeline.")
    arguments, remaining = parser.parse_known_args()

    if arguments.command == "train":
        result = run_pipeline()
        print(f"Selected model: {result['selected_model']}")
        print(f"Metrics: {result['metrics']}")
        print(f"Model saved to: {result['model_path']}")
        if not result["model_replaced"]:
            print(
                "The candidate did not replace the incumbent because its "
                "poisonous recall was lower on the shared holdout."
            )
        if result["model_backup_path"]:
            print(f"Previous model backed up to: {result['model_backup_path']}")
        if result["registry_result"]:
            registered = result["registry_result"]["registered_model"]
            print(f"MLflow registered model: {registered}")
    else:
        import sys

        sys.argv = [sys.argv[0], *remaining]
        inference_main()


if __name__ == "__main__":
    main()
