from __future__ import annotations

import pandas as pd

from api.schemas import MushroomFeatures
from src.mushroom_classification.inference import predict_frame


def build_feature_frame(record: MushroomFeatures) -> pd.DataFrame:
    return pd.DataFrame([record.model_dump(by_alias=True)])


def predict_records(model: object, records: list[MushroomFeatures]) -> pd.DataFrame:
    frame = pd.DataFrame(
        [record.model_dump(by_alias=True) for record in records]
    )
    return predict_frame(model, frame)
