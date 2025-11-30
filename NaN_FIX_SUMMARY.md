# NaN Issue Fix Summary

## Problem
Training was producing NaN values in loss, preventing the model from learning.

## Root Causes Identified
1. **Loss function argument mismatch**: The trainer was calling `training_loss(out, **sample)` which unpacks the sample dictionary, but loss functions expect `(y_pred, y)` as positional arguments.
2. **High learning rate**: The default learning rate of `3e-4` may have been too high, causing gradient explosion.
3. **No gradient clipping**: No protection against gradient explosion.

## Fixes Applied

### 1. Fixed Loss Function Calls
**File**: `clone/neuraloperator/neuralop/training/trainer.py`

- Changed `training_loss(out, **sample)` to `training_loss(out, y)` in:
  - `train_one_batch()` method (line ~554)
  - `eval_one_batch()` method (line ~616)
  - `eval_one_batch_autoreg()` method (line ~700)

This ensures the loss function receives `y` as a positional argument matching its signature: `loss(y_pred, y)`.

### 2. Added NaN Detection and Debugging
**File**: `clone/neuraloperator/neuralop/training/trainer.py`

Added checks in `train_one_batch()` to detect NaN/Inf values:
- Check input data (`x`, `y`) for NaN/Inf on first batch
- Check model output for NaN/Inf on first batch
- Check loss value for NaN/Inf and print diagnostics

### 3. Lowered Learning Rate
**File**: `clone/neuraloperator/config/navier_stokes_config.py`

- Reduced learning rate from `3e-4` to `1e-4` to prevent gradient explosion

### 4. Added Gradient Clipping
**Files**: 
- `clone/neuraloperator/neuralop/training/trainer.py`
- `clone/neuraloperator/config/opt.py`
- `clone/neuraloperator/scripts/train_navier_stokes.py`

- Added `grad_clip` parameter to `Trainer.__init__()` (default: `None`)
- Added gradient clipping in `train_one_epoch()` after `loss.backward()`:
  ```python
  if self.grad_clip is not None:
      torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
  ```
- Added `grad_clip: Optional[float] = 1.0` to `OptimizationConfig`
- Passed `grad_clip` to trainer in training script

### 5. Created Diagnostic Script
**File**: `tests/debug_nan.py`

Created a diagnostic script to check for NaN values in:
- Raw data
- Processed/normalized data
- Model parameters
- Model outputs
- Loss values

## Testing Recommendations

1. **Run the diagnostic script** (requires environment setup):
   ```bash
   cd /Users/ryanlin/Desktop/NeuralOperatorProject
   source NO_env/bin/activate
   python tests/debug_nan.py
   ```

2. **Monitor training output** for:
   - Warning messages about NaN/Inf values
   - Loss values should be finite (not NaN)
   - Training should progress normally

3. **If NaN persists**, try:
   - Further reduce learning rate to `5e-5` or `1e-5`
   - Increase gradient clipping to `0.5` or `0.1`
   - Check data normalization is working correctly
   - Verify data files are not corrupted

## Expected Behavior After Fix

- Loss values should be finite numbers (not NaN)
- Training should progress with decreasing loss
- No warnings about unexpected keyword arguments
- Model should learn and improve over epochs

## Files Modified

1. `clone/neuraloperator/neuralop/training/trainer.py` - Fixed loss calls, added NaN checks, added gradient clipping
2. `clone/neuraloperator/config/navier_stokes_config.py` - Lowered learning rate
3. `clone/neuraloperator/config/opt.py` - Added grad_clip config option
4. `clone/neuraloperator/scripts/train_navier_stokes.py` - Pass grad_clip to trainer
5. `tests/debug_nan.py` - Created diagnostic script



