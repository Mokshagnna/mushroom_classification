from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml

from src.mlops_pipeline.mushroom_ops import run_stage
from src.mushroom_classification.config import PROJECT_ROOT


class MushroomStageTests(unittest.TestCase):
    def test_load_validate_preprocess_and_validate_outputs_in_isolation(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mushroom-stage-tests-") as directory:
            temp = Path(directory)
            config = yaml.safe_load(
                (PROJECT_ROOT / "config.yaml").read_text(encoding="utf-8")
            )
            config["data"]["raw_path"] = str(
                PROJECT_ROOT / "data" / "raw" / "mushroom.csv"
            )
            config["data"]["processed_dir"] = str(temp / "processed")
            for name in config["paths"]:
                config["paths"][name] = str(temp / Path(config["paths"][name]).name)
            config["paths"]["preprocessor"] = str(temp / "preprocessor.joblib")
            config["paths"]["preprocessing_validation_report"] = str(
                temp / "preprocessing_validation.json"
            )
            config_path = temp / "config.yaml"
            config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

            self.assertEqual(run_stage(1, config_path)["row_count"], 8124)
            self.assertTrue(run_stage(2, config_path)["valid"])
            preprocessing_report = run_stage(3, config_path)
            self.assertEqual(
                preprocessing_report["train_shape"][0]
                + preprocessing_report["test_shape"][0],
                8124,
            )

            output_report = run_stage(4, config_path)
            self.assertEqual(output_report["status"], "PASSED")
            self.assertTrue(
                np.load(temp / "processed" / "X_train_final.npy").shape[0]
                > np.load(temp / "processed" / "X_test_final.npy").shape[0]
            )


if __name__ == "__main__":
    unittest.main()
