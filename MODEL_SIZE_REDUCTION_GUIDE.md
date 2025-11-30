# Guide to Reducing Model Size

## Current Model Configuration

Your current model uses `FNO_Medium2d()` with:
- **Parameters**: ~69,265,345 (~69M)
- **n_modes**: [64, 64]
- **hidden_channels**: 64
- **projection_channel_ratio**: 4
- **lifting_channel_ratio**: 2 (default)
- **n_layers**: 4 (default)

## Option 1: Use Smaller Preset (Easiest) ✅ APPLIED

I've already updated your config to use `FNO_Small2d()` instead of `FNO_Medium2d()`.

**FNO_Small2d parameters:**
- **n_modes**: [16, 16] (reduced from [64, 64])
- **hidden_channels**: 24 (reduced from 64)
- **projection_channel_ratio**: 2 (reduced from 4)
- **Expected parameters**: ~500K - 2M (much smaller!)

**Impact:**
- Model size reduced by ~30-100x
- Better match to your 10K training samples
- Should train faster and more stably

## Option 2: Custom Model Configuration

If you want more control, you can create a custom model config. Here's how:

### Step 1: Create Custom Model Class

Add this to your `navier_stokes_config.py`:

```python
from .models import SimpleFNOConfig

class NavierStokesFNOConfig(SimpleFNOConfig):
    """Custom FNO config optimized for Navier-Stokes with reduced size."""
    data_channels: int = 1
    out_channels: int = 1
    n_modes: List[int] = [32, 32]  # Reduced from [64, 64]
    hidden_channels: int = 32  # Reduced from 64
    projection_channel_ratio: int = 2  # Reduced from 4
    # These use defaults from FNOConfig:
    # lifting_channel_ratio: int = 2
    # n_layers: int = 4
```

### Step 2: Update Default Config

Change line 39 in `navier_stokes_config.py`:

```python
model: ModelConfig = NavierStokesFNOConfig()  # Custom config
```

## Parameter Impact on Model Size

Here's how each parameter affects model size (from most to least impact):

### 1. `n_modes` (Highest Impact)
- **Current**: [64, 64]
- **What it does**: Number of Fourier modes kept in each dimension
- **Impact**: Quadratic - affects spectral convolution size
- **Recommendations**:
  - [16, 16] → ~16x smaller spectral convs
  - [32, 32] → ~4x smaller spectral convs
  - [48, 48] → ~1.8x smaller spectral convs

### 2. `hidden_channels` (High Impact)
- **Current**: 64
- **What it does**: Width of the FNO (number of channels)
- **Impact**: Quadratic - affects all layer sizes
- **Recommendations**:
  - 24 → ~7x smaller (FNO_Small2d)
  - 32 → ~4x smaller
  - 48 → ~1.8x smaller

### 3. `n_layers` (Medium Impact)
- **Current**: 4 (default)
- **What it does**: Number of FNO blocks
- **Impact**: Linear - each layer adds parameters
- **Recommendations**:
  - 3 → ~25% reduction
  - 2 → ~50% reduction (may hurt performance)

### 4. `projection_channel_ratio` (Medium Impact)
- **Current**: 4
- **What it does**: Multiplier for projection layer channels
- **Impact**: Affects final projection layer size
- **Recommendations**:
  - 2 → ~50% smaller projection (FNO_Small2d uses this)
  - 3 → ~25% smaller projection

### 5. `lifting_channel_ratio` (Lower Impact)
- **Current**: 2 (default)
- **What it does**: Multiplier for lifting layer channels
- **Impact**: Affects initial lifting layer size
- **Recommendations**:
  - 1 → ~50% smaller lifting
  - 1.5 → ~25% smaller lifting

## Recommended Configurations

### Configuration A: Small (Recommended for 10K samples)
```python
n_modes: [16, 16]
hidden_channels: 24
projection_channel_ratio: 2
n_layers: 4
lifting_channel_ratio: 2
```
**Expected size**: ~500K - 2M parameters
**Use case**: Limited data, faster training

### Configuration B: Medium-Small (Balance)
```python
n_modes: [32, 32]
hidden_channels: 32
projection_channel_ratio: 2
n_layers: 4
lifting_channel_ratio: 2
```
**Expected size**: ~2M - 5M parameters
**Use case**: Good balance of size and capacity

### Configuration C: Custom Medium (Current with reductions)
```python
n_modes: [48, 48]  # Reduced from [64, 64]
hidden_channels: 48  # Reduced from 64
projection_channel_ratio: 3  # Reduced from 4
n_layers: 4
lifting_channel_ratio: 2
```
**Expected size**: ~10M - 20M parameters
**Use case**: Still large but more manageable

## How to Implement Custom Config

### Full Example for Configuration B:

```python
# In navier_stokes_config.py
from typing import Any, List, Optional
from zencfg import ConfigBase
from .distributed import DistributedConfig
from .models import ModelConfig, SimpleFNOConfig  # Add SimpleFNOConfig
from .opt import OptimizationConfig, PatchingConfig
from .wandb import WandbConfig

# Create custom model config
class NavierStokesFNOConfig(SimpleFNOConfig):
    """Custom FNO config optimized for Navier-Stokes."""
    data_channels: int = 1
    out_channels: int = 1
    n_modes: List[int] = [32, 32]  # Reduced from [64, 64]
    hidden_channels: int = 32  # Reduced from 64
    projection_channel_ratio: int = 2  # Reduced from 4

# ... rest of your config classes ...

class Default(ConfigBase):
    n_params_baseline: Optional[Any] = None
    verbose: bool = True
    distributed: DistributedConfig = DistributedConfig()
    model: ModelConfig = NavierStokesFNOConfig()  # Use custom config
    opt: OptimizationConfig = NavierStokesOptConfig()
    data: NavierStokesDatasetConfig = NavierStokesDatasetConfig()
    patching: PatchingConfig = PatchingConfig()
    wandb: WandbConfig = WandbConfig()
```

## Verifying Model Size

After changing the config, run your training script and check the output. You should see:

```
n_params: [new parameter count]
```

Compare this to the original 69,265,345 parameters.

## Expected Performance Impact

### Smaller Models:
- **Pros**: 
  - Faster training
  - Less memory usage
  - Better generalization with limited data
  - More stable gradients
- **Cons**:
  - May have lower capacity
  - Might need more epochs to converge

### Trade-offs:
- **FNO_Small2d** (24 channels, [16,16] modes): Good for initial experiments, may need fine-tuning
- **Custom Medium** (32-48 channels, [32,32] modes): Better balance, closer to documented FNO-2D size
- **Original FNO_Medium2d** (64 channels, [64,64] modes): Too large for 10K samples

## Next Steps

1. ✅ **Already done**: Switched to `FNO_Small2d()` 
2. **Test training**: Run training and check:
   - New parameter count
   - Training stability (gradient norms)
   - Loss convergence
3. **If too small**: Try custom config with Configuration B
4. **If still unstable**: Further reduce `hidden_channels` or `n_modes`

## Reference: Documented FNO-2D

According to the documentation, the standard FNO-2D for Navier-Stokes has:
- **Parameters**: 414,517 (~414K)
- **n_modes**: Likely [32, 32] or smaller
- **hidden_channels**: Likely 32-48
- **L2 error**: ~0.0128 (your current: ~0.66-0.69)

Your goal should be to get closer to this size and performance.

