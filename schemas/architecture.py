from typing import Optional
from schemas.utils import ImmutableArgs


class ArchitectureArgs(ImmutableArgs):

    #: Use pretrained model, or train from scratch.
    use_pretrained: bool = False

    #: Pretrained model path (in wandb).
    pretrained_path: Optional[str] = None

    #: If it's true, use adaptive-average-pooling to 1x1 before feeding to the final MLP (like done in ResNets).
    adaptive_avg_pool_before_mlp: bool = True

    #: If it's true, after first conv->bn->relu use 3x3 max-pool with stride=2 and padding (like done in ResNets).
    max_pool_after_first_conv: bool = False
