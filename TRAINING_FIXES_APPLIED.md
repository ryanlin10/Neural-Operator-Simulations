# Training Fixes Applied

## Issues Identified from Terminal Output

1. **Loss plateauing**: Training loss stuck around 0.86-0.88, not decreasing
2. **Gradient norms still growing**: Reaching 18+ despite clipping at 1.0
3. **Learning rate not adapting**: LR stayed at 5e-05 throughout training
4. **Evaluation metrics not improving**: L2 loss stuck around 0.66
5. **Over-aggressive gradient clipping**: Clipping at 1.0 was preventing learning

## Fixes Applied

### 1. Reduced Learning Rate Further
**Change**: `5e-5` → `2e-5`
**Rationale**: 
- Current LR was still too high, causing gradient instability
- Lower LR will allow more stable learning and better convergence
- With smaller model (671K params), lower LR is appropriate

### 2. Switched Training Loss from H1 to L2
**Change**: `training_loss: "h1"` → `training_loss: "l2"`
**Rationale**:
- H1 loss includes gradient terms which are more sensitive
- L2 loss is simpler and more stable for initial training
- Can switch back to H1 later for fine-tuning if needed
- L2 loss typically leads to better initial convergence

### 3. Increased Gradient Clipping Threshold
**Change**: `grad_clip: 1.0` → `grad_clip: 2.0`
**Rationale**:
- Clipping at 1.0 was too aggressive - almost every batch was being clipped
- This was preventing the model from learning effectively
- 2.0 allows more gradient flow while still preventing explosion
- Gradients were reaching 18+, so 2.0 is still conservative

### 4. Reduced Scheduler Patience
**Change**: `scheduler_patience: 10` → `scheduler_patience: 5`
**Rationale**:
- With patience=10, scheduler waits too long before reducing LR
- Loss was plateauing but scheduler wasn't responding
- Shorter patience will trigger LR reduction sooner when loss plateaus

### 5. Fixed Scheduler to Use Validation Loss
**Change**: Modified `trainer.py` to use validation L2 loss for ReduceLROnPlateau
**Rationale**:
- Previously scheduler was monitoring training loss (`train_err`)
- Training loss can fluctuate and isn't a good indicator of generalization
- Validation loss is a better metric for learning rate scheduling
- This ensures LR reduces when validation loss plateaus, not training loss

## Expected Improvements

### Immediate Effects:
1. **More stable gradients**: With grad_clip=2.0, gradients won't be clipped as frequently
2. **Better learning**: L2 loss is less sensitive than H1, allowing smoother optimization
3. **Adaptive learning rate**: Scheduler will now respond to validation loss and reduce LR when needed

### Short-term (10-20 epochs):
1. **Loss should start decreasing**: With lower LR and L2 loss, training should be more stable
2. **Gradient norms should stabilize**: Less frequent clipping means more consistent updates
3. **Learning rate will adapt**: When validation loss plateaus, LR will reduce automatically

### Long-term (100+ epochs):
1. **Better convergence**: Lower LR and adaptive scheduling should lead to better final performance
2. **Improved validation metrics**: Using validation loss for scheduling should improve generalization

## Configuration Summary

**Before:**
- Learning rate: 5e-5
- Training loss: H1
- Gradient clipping: 1.0
- Scheduler patience: 10
- Scheduler metric: Training loss

**After:**
- Learning rate: 2e-5
- Training loss: L2
- Gradient clipping: 2.0
- Scheduler patience: 5
- Scheduler metric: Validation L2 loss

## Next Steps

1. **Restart training** with the new configuration
2. **Monitor gradient norms**: Should see less frequent clipping (norms staying < 2.0 more often)
3. **Watch for LR reduction**: After 5 epochs without validation improvement, LR should reduce
4. **Check loss trends**: Both training and validation loss should start decreasing

## If Issues Persist

If training still doesn't improve:
1. **Further reduce learning rate**: Try 1e-5
2. **Increase batch size**: If memory allows, try 16 or 32
3. **Consider gradient accumulation**: Accumulate over 2-4 batches for effective larger batch size
4. **Monitor specific layers**: Check if certain layers have much larger gradients

