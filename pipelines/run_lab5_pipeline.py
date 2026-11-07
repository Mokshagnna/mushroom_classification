from _runner import run_stages


if __name__ == "__main__":
    raise SystemExit(
        run_stages(
            [
                "01_load_data.py",
                "02_validate_data.py",
                "03_preprocessing.py",
                "04_validate_outputs.py",
                "05_train_registry.py",
                "06_model_evaluation.py",
                "07_quality_gate.py",
            ]
        )
    )
