from typing import Any, List, Optional

from zencfg import ConfigBase
from .distributed import DistributedConfig
from .models import ModelConfig, FNO_Medium2d, FNO_Small2d
from .opt import OptimizationConfig, PatchingConfig
from .wandb import WandbConfig


class NavierStokesOptConfig(OptimizationConfig):
    n_epochs: int = 600
    learning_rate: float = 1e-5  # Further reduced to stabilize gradients and improve convergence
    training_loss: str = "l2"  # Changed from h1 to l2 for more stable initial training
    weight_decay: float = 1e-4
    optimizer: str = "AdamW"  # Options: "AdamW" or "SGD" - SGD often finds flatter minima
    sgd_momentum: float = 0.9  # Momentum for SGD optimizer
    scheduler: str = "CosineAnnealingLR"  # Changed to CosineAnnealingLR for warm restarts to escape sharp minima
    step_size: int = 100
    gamma: float = 0.5
    scheduler_patience: int = 3  # For ReduceLROnPlateau
    scheduler_T_max: int = 50  # CosineAnnealingLR period - warm restart every 50 epochs
    grad_clip: Optional[float] = 3.0  # Increased from 2.0 - allow more gradient flow while preventing explosion
    grad_noise_scale: Optional[float] = 0.01  # Gradient noise scale (None to disable) - helps escape sharp minima


class NavierStokesDatasetConfig(ConfigBase):
    folder: str = "~/data/navier_stokes/"
    batch_size: int = 8
    n_train: int = 10000
    train_resolution: int = 128
    n_tests: List[int] = [2000]
    test_resolutions: List[int] = [128]
    test_batch_sizes: List[int] = [8]
    encode_input: bool = False
    encode_output: bool = True


class NavierStokesCheckpointConfig(ConfigBase):
    """Configuration for checkpoint saving during training."""
    save_dir: str = "./ckpt"  # Directory to save checkpoints
    save_best: Optional[str] = "128_l2"  # Save best model based on this metric (None to disable)
    save_every: Optional[int] = None  # Save checkpoint every N epochs (None to disable)


class Default(ConfigBase):
    n_params_baseline: Optional[Any] = None
    verbose: bool = True
    distributed: DistributedConfig = DistributedConfig()
    model: ModelConfig = FNO_Small2d()  # Reduced from FNO_Medium2d() to reduce model size
    opt: OptimizationConfig = NavierStokesOptConfig()
    data: NavierStokesDatasetConfig = NavierStokesDatasetConfig()
    patching: PatchingConfig = PatchingConfig()
    wandb: WandbConfig = WandbConfig()
    checkpoint: NavierStokesCheckpointConfig = NavierStokesCheckpointConfig()
