from typing import Sequence, Optional, Union, Literal, List
from dataclasses import dataclass


@dataclass
class GeneratorConfig:
    latent_channels: int = 128
    hidden_channels: Sequence[int] = (64, 64, 48, 48, 32, 16, 3)
    upsamplings: Sequence[int] = (32, 2, 2, 2, 2, 2)
    k: int = 10
    aggr: Literal['mean', 'add', 'max'] = 'mean'
    n_classes: int = 10

    freq: int = 5

    lr: float = 1e-4
    lr_decay: float = 0.5
    multistep_milestones: List[int] = (int(1e9),)


@dataclass
class DiscriminatorConfig:
    hidden_channels: Sequence[int] = (3, 16, 32, 64, 128, 256)
    k: int = 10
    aggr: Literal['mean', 'add', 'max'] = 'mean'
    n_classes: int = 10

    freq: int = 1

    lr: float = 1e-4
    lr_decay: float = 0.5
    multistep_milestones: List[int] = (int(1e9),)


@dataclass
class Config:
    generator: GeneratorConfig = GeneratorConfig()
    discriminator: DiscriminatorConfig = DiscriminatorConfig()

    batch_size: int = 4
    gradient_penalty: float = 10.
    center: float = 1.
    regulate_difference: bool = False
    lambda2: float = 1e-2

    n_points: int = 1024
    label: Literal[
            'bathtub', 'bed', 'chair', 'desk', 'dresser', 'monitor',
            'night_stand', 'sofa', 'table', 'toilet'
        ] = 'toilet'
