from typing import Sequence, Optional, Union, Literal
from dataclasses import dataclass


@dataclass
class GeneratorConfig:
    latent_channels: int = 128
    hidden_channels: Sequence[int] = (96, 64, 48, 32, 16, 3)
    upsamplings: Sequence[int] = (64, 2, 2, 2, 2)
    k: int = 20
    aggr: Literal['mean', 'add', 'max'] = 'mean'

    lr: float = 1e-4


@dataclass
class DiscriminatorConfig:
    hidden_channels: Sequence[int] = (3, 32, 64, 64, 128)
    k: int = 20
    aggr: Literal['mean', 'add', 'max'] = 'mean'

    lr: float = 1e-4


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
