from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class MushroomFeatures(BaseModel):
    """The 22 categorical fields accepted by the trained mushroom pipeline."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    cap_shape: Literal["b", "c", "f", "k", "s", "x"] = Field(alias="cap-shape")
    cap_surface: Literal["f", "g", "s", "y"] = Field(alias="cap-surface")
    cap_color: Literal["b", "c", "e", "g", "n", "p", "r", "u", "w", "y"] = Field(
        alias="cap-color"
    )
    bruises: Literal["f", "t"]
    odor: Literal["a", "c", "f", "l", "m", "n", "p", "s", "y"]
    gill_attachment: Literal["a", "f"] = Field(alias="gill-attachment")
    gill_spacing: Literal["c", "w"] = Field(alias="gill-spacing")
    gill_size: Literal["b", "n"] = Field(alias="gill-size")
    gill_color: Literal["b", "e", "g", "h", "k", "n", "o", "p", "r", "u", "w", "y"] = (
        Field(alias="gill-color")
    )
    stalk_shape: Literal["e", "t"] = Field(alias="stalk-shape")
    stalk_root: Literal["?", "b", "c", "e", "r"] = Field(alias="stalk-root")
    stalk_surface_above_ring: Literal["f", "k", "s", "y"] = Field(
        alias="stalk-surface-above-ring"
    )
    stalk_surface_below_ring: Literal["f", "k", "s", "y"] = Field(
        alias="stalk-surface-below-ring"
    )
    stalk_color_above_ring: Literal["b", "c", "e", "g", "n", "o", "p", "w", "y"] = (
        Field(alias="stalk-color-above-ring")
    )
    stalk_color_below_ring: Literal["b", "c", "e", "g", "n", "o", "p", "w", "y"] = (
        Field(alias="stalk-color-below-ring")
    )
    veil_type: Literal["p"] = Field(alias="veil-type")
    veil_color: Literal["n", "o", "w", "y"] = Field(alias="veil-color")
    ring_number: Literal["n", "o", "t"] = Field(alias="ring-number")
    ring_type: Literal["e", "f", "l", "n", "p"] = Field(alias="ring-type")
    spore_print_color: Literal["b", "h", "k", "n", "o", "r", "u", "w", "y"] = Field(
        alias="spore-print-color"
    )
    population: Literal["a", "c", "n", "s", "v", "y"]
    habitat: Literal["d", "g", "l", "m", "p", "u", "w"]


class PredictionResponse(BaseModel):
    prediction: Literal[0, 1]
    predicted_class: Literal["edible", "poisonous"]
    edible_probability: float
    poisonous_probability: float
    request_id: str


class BatchPredictionRequest(BaseModel):
    records: list[MushroomFeatures] = Field(min_length=1, max_length=1000)


class BatchPredictionResponse(BaseModel):
    count: int
    predictions: list[PredictionResponse]


class HealthResponse(BaseModel):
    status: Literal["healthy"]
    model_loaded: bool


class ModelMetadataResponse(BaseModel):
    service: str
    model_loaded: bool
    feature_count: int
    class_mapping: dict[str, str]
    safety_notice: str
