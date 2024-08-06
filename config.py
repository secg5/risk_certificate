from dataclasses import dataclass


@dataclass
class Config:
    number_arms: list[float]
    first_stage_size: int
    distribution: list[float]
    sample_size: int
    random_seed: int = 42
