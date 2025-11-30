# Training Fixes - Round 2

## Issues Identified from Latest Training Output

### 1. Learning Rate Not Adapting
- **Problem**: LR stayed at 2e-05 throughout all epochs despite validation loss plateauing
- **Root Cause**: 
  - Validation loss was fluctuating (0.5869 → 0.6127 → 0.6011 → 0.5884)
  - Scheduler patience was 5 epochs, but loss was improving slightly at epoch 5
  - No visibility into when/if scheduler was reducing LR

### 2. Gradients Still Growing
- **Problem**: Gradient norms reaching 18+ despite clipping at 2.0
- **Evidence**:
  - Epoch 0: mostly < 2.0
  - Epoch 1-2: reaching 5-7
  - Epoch 3-4: reaching 12-16
  - Epoch 5-6: reaching 12-18
- **Root Cause**: Learning rate (2e-05) was still too high for the model size and loss landscape

### 3. Validation Loss Fluctuating
- **Problem**: Validation loss not consistently improving
- **Evidence**:
  - Epoch 0: 0.8294
  - Epoch 1: 0.5869 (best)
  - Epoch 2: 0.6093 (worse)
  - Epoch 3: 0.6127 (worse)
  - Epoch 4: 0.6011 (better but still worse than epoch 1)
  - Epoch 5: 0.5884 (close to epoch 1)
- **Root Cause**: Instability from high learning rate and frequent gradient clipping

## Fixes Applied

### 1. Further Reduced Learning Rate
**Change**: `2e-5` → `1e-5`
**Rationale**:
- Gradients were still growing, indicating LR was too high
- Lower LR will provide more stable training
- With smaller model (671K params), lower LR is appropriate
- Will reduce gradient magnitudes and improve convergence

### 2. Increased Gradient Clipping Threshold
**Change**: `grad_clip: 2.0` → `grad_clip: 3.0`
**Rationale**:
- Clipping at 2.0 was happening too frequently (almost every batch)
- This was preventing the model from learning effectively
- 3.0 allows more gradient flow while still preventing explosion
- Gradients reaching 18+ suggest we need more headroom

### 3. Reduced Scheduler Patience
**Change**: `scheduler_patience: 5` → `scheduler_patience: 3`
**Rationale**:
- With patience=5, scheduler waits too long before reducing LR
- Validation loss was fluctuating, so scheduler wasn't responding quickly enough
- Patience=3 will trigger LR reduction sooner when loss plateaus
- More aggressive scheduling should help with convergence

### 4. Added LR Reduction Logging
**Change**: Added logging in trainer.py to show when LR is reduced
**Rationale**:
- Previously no visibility into when scheduler was reducing LR
- Now will print: "Learning rate reduced: X -> Y (validation loss: Z)"
- Helps debug scheduler behavior
- Confirms scheduler is working correctly

## Expected Improvements

### Immediate Effects:
1. **More stable gradients**: Lower LR (1e-5) should reduce gradient magnitudes
2. **Less frequent clipping**: Higher threshold (3.0) allows more gradient flow
3. **Faster LR adaptation**: Patience=3 will reduce LR sooner when needed
4. **Better visibility**: LR reduction logging shows scheduler activity

### Short-term (10-20 epochs):
1. **Gradient norms should stabilize**: Should stay mostly < 3.0, with occasional clips
2. **LR should reduce**: After 3 epochs without improvement, LR will drop
3. **Validation loss should improve**: More stable training should lead to better convergence
4. **Training loss should decrease**: Lower LR allows finer optimization

### Long-term (100+ epochs):
1. **Better convergence**: Lower LR and adaptive scheduling should reach better minima
2. **Stable training**: Gradients should remain controlled throughout training
3. **Improved validation metrics**: Should see consistent improvement in validation loss

## Configuration Summary

**Before:**
- Learning rate: 2e-5
- Gradient clipping: 2.0
- Scheduler patience: 5
- LR reduction logging: None

**After:**
- Learning rate: 1e-5
- Gradient clipping: 3.0
- Scheduler patience: 3
- LR reduction logging: Enabled

## Monitoring Points

When you restart training, watch for:

1. **Gradient norms**: Should mostly stay < 3.0, with occasional clips up to 5-6
2. **LR reduction messages**: Should appear after 3 epochs without validation improvement
3. **Validation loss trend**: Should show consistent improvement (not just fluctuation)
4. **Training loss**: Should decrease steadily, not plateau

## If Issues Persist

If gradients still grow or loss doesn't improve:

1. **Further reduce LR**: Try 5e-6
2. **Increase batch size**: If memory allows, try 16 or 32
3. **Add gradient accumulation**: Accumulate over 2-4 batches
4. **Consider different scheduler**: Try CosineAnnealingLR with warm restarts
5. **Check data normalization**: Ensure input/output normalization is correct

