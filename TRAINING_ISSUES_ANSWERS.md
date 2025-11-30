# Answers to Training Investigation Questions

## 1. Why is the loss seemingly not decreasing?

**Answer:** The loss is actually decreasing slightly in the first epoch (0.8695 → 0.8653), but then it plateaus and even increases slightly (0.8658). More critically, **the evaluation metrics are getting worse**:

- **H1 loss**: 0.8613 → 0.8641 → 0.8657 (increasing)
- **L2 loss**: 0.6578 → 0.6753 → 0.6866 (increasing)

This indicates:
1. **Overfitting**: The model is memorizing training data but not generalizing
2. **Training instability**: The growing gradient norms (see below) suggest the optimization is unstable
3. **Learning rate too high**: The model may be overshooting optimal points in the loss landscape

The training loss metric (`train_err`) shows values around 6.9-7.0, which is the **summed loss per batch**. This is different from `avg_loss` which is averaged per sample. Both metrics should decrease, but they're plateauing.

## 2. Why do the gradients become larger as batches progress?

**Answer:** This is a classic sign of **gradient explosion** and training instability. The gradient norms are growing exponentially:

- **Epoch 0, Batch 0**: 0.078 (normal)
- **Epoch 0, Batch 1100**: 18.26 (CLIPPED at 10.0)
- **Epoch 1, Batch 1100**: 35.12 (CLIPPED at 10.0)
- **Epoch 2, Batch 500**: 38.94 (CLIPPED at 10.0)

**Root causes:**
1. **Learning rate too high**: With LR=1e-4, the optimizer makes large steps, which can push the model into regions with steeper gradients
2. **Gradient clipping too lenient**: The threshold of 10.0 is insufficient when gradients reach 35-38. Clipping should be more aggressive (1.0 or lower)
3. **Loss landscape instability**: As training progresses, the model may be entering regions where the loss surface is steeper, causing gradients to accumulate
4. **No learning rate decay early enough**: The StepLR scheduler only reduces LR every 100 epochs, which is too infrequent for this unstable training

**Why this happens:**
- High learning rate → large weight updates → model moves to new regions of loss landscape
- If the new region has steeper gradients, the next update will be even larger
- This creates a positive feedback loop: larger gradients → larger updates → steeper gradients → even larger gradients
- Gradient clipping helps, but if the threshold is too high, it doesn't prevent the instability

## 3. Is this behavior normal given the dataset and model size?

**Answer: No, this behavior is NOT normal.** Here's why:

### Expected Performance
According to the documentation, for Navier-Stokes with ν=1e-3:
- **FNO-2D** (414K parameters) should achieve **L2 error ≈ 0.0128**
- Your model has **L2 error ≈ 0.66-0.69**, which is **50x worse** than expected

### Model Size Analysis
- **Your model**: 69,265,345 parameters (~69M)
- **Documented FNO-2D**: 414,517 parameters (~414K)
- **Your model is 167x larger** than the documented configuration

### Dataset Size
- **Training samples**: 10,000
- **Parameter-to-sample ratio**: ~6,900:1 (very high, suggesting overparameterization)

### Normal Behavior Should Be:
1. **Loss decreasing steadily** over epochs (not plateauing)
2. **Gradient norms stable or decreasing** (not growing exponentially)
3. **Evaluation metrics improving** (not getting worse)
4. **Reaching L2 error < 0.1** within 50-100 epochs for this problem

### What's Abnormal:
1. **Gradient explosion** (norms growing from 0.078 to 38.94)
2. **Evaluation metrics worsening** (overfitting)
3. **Loss plateauing** after just 1-2 epochs
4. **Frequent gradient clipping** (almost every batch after epoch 0)

## 4. How long will it take to reach optimal loss?

**Answer:** With the current configuration, **the model will likely NEVER reach optimal loss** because:

1. **Training is unstable**: Gradient explosion will prevent convergence
2. **Overfitting**: Evaluation metrics are already worsening
3. **Hyperparameters are suboptimal**: Learning rate and gradient clipping need adjustment

### With Current Settings:
- **Expected outcome**: Loss will plateau around 0.86-0.87, evaluation will continue to worsen
- **Time to plateau**: Already reached (within 2-3 epochs)
- **Optimal loss**: Will not be reached

### With Recommended Fixes:
- **Time to see improvement**: 10-20 epochs
- **Time to reach good loss (< 0.1 L2)**: 100-200 epochs (if fixes work)
- **Time to reach optimal loss (< 0.02 L2)**: 200-400 epochs (if model size is appropriate)

## 5. What should I do to improve this training?

### Immediate Actions (Do First):

1. **Reduce Learning Rate**
   - Change from `1e-4` to `5e-5` or `2e-5`
   - This will reduce gradient magnitudes and improve stability
   - **Status**: ✅ Already fixed in config

2. **Tighten Gradient Clipping**
   - Change from `10.0` (or `30.0` in config) to `1.0`
   - More aggressive clipping will prevent gradient explosion
   - **Status**: ✅ Already fixed in config

3. **Switch to Adaptive Scheduler**
   - Change from `StepLR` to `ReduceLROnPlateau`
   - This will automatically reduce LR when loss plateaus
   - **Status**: ✅ Already fixed in config

### Short-term Actions (Next Run):

4. **Add Learning Rate Warmup** (if possible)
   - Gradually increase LR over first 10-20 epochs
   - Prevents early instability

5. **Consider L2 Loss for Initial Training**
   - H1 loss includes gradient terms which can be more sensitive
   - Start with L2, switch to H1 later for fine-tuning

6. **Increase Batch Size** (if memory allows)
   - Current: 8
   - Recommended: 16 or 32
   - Provides more stable gradient estimates

### Long-term Considerations:

7. **Reduce Model Size**
   - Current: 69M parameters (167x larger than documented)
   - Consider reducing `hidden_channels`, `n_layers`, or `n_modes`
   - Better match to dataset size (10K samples)

8. **Monitor Early Stopping**
   - Stop training when validation loss starts increasing
   - Current training shows this happening after epoch 0

9. **Data Augmentation** (if applicable)
   - With only 10K samples, augmentation can help
   - For PDEs: rotation, reflection (if physically valid)

## Summary

The training is **unstable and not converging** due to:
- Learning rate too high (1e-4)
- Gradient clipping too lenient (10.0)
- Fixed learning rate schedule (StepLR)

**Fixes have been applied** to the config file:
- Learning rate: 1e-4 → 5e-5
- Gradient clipping: 30.0 → 1.0
- Scheduler: StepLR → ReduceLROnPlateau

**Next steps:**
1. Restart training with the updated config
2. Monitor gradient norms (should stay < 5.0)
3. Monitor evaluation metrics (should improve, not worsen)
4. If issues persist, consider reducing model size or switching to L2 loss

**Expected improvement timeline:**
- Within 10-20 epochs: Should see stable gradients and decreasing loss
- Within 100-200 epochs: Should reach reasonable performance (< 0.1 L2 error)
- May need model size reduction to reach optimal performance (< 0.02 L2 error)

