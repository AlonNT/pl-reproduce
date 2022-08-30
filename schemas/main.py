from typing import Literal
from pydantic import root_validator
from pydantic.types import PositiveInt

from schemas.architecture import ArchitectureArgs
from schemas.data import DataArgs
from schemas.environment import EnvironmentArgs
from schemas.optimization import OptimizationArgs
from schemas.utils import ImmutableArgs, MyBaseModel, NonNegativeInt, Fraction


class Args(MyBaseModel):
    opt = OptimizationArgs()
    arch = ArchitectureArgs()
    env = EnvironmentArgs()
    data = DataArgs()
