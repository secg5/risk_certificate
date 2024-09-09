from dataclasses import dataclass


@dataclass
class Config:
    number_arms: list[float]
    first_stage_size: int
    distribution: list[float]
    sample_size: int
    delta: float 
    run_all_k: bool
    reward_parameters: dict
    arm_distribution: str 
    random_seed: int = 42
