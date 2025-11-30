"""
Debug script to identify NaN sources in Navier-Stokes training.
Run this before training to check data and model initialization.
"""

import sys
from pathlib import Path
import torch

# Add neuraloperator to path
sys.path.insert(0, str(Path(__file__).parent.parent / "clone" / "neuraloperator"))

from neuralop.data.datasets.navier_stokes import load_navier_stokes_pt
from neuralop import get_model
from zencfg import make_config_from_cli
from config.navier_stokes_config import Default

def check_tensor(name, tensor):
    """Check tensor for NaN, inf, and value ranges."""
    if not isinstance(tensor, torch.Tensor):
        print(f"{name}: Not a tensor (type: {type(tensor)})")
        return
    
    has_nan = torch.isnan(tensor).any().item()
    has_inf = torch.isinf(tensor).any().item()
    min_val = tensor.min().item()
    max_val = tensor.max().item()
    mean_val = tensor.mean().item()
    std_val = tensor.std().item()
    
    print(f"\n{name}:")
    print(f"  Shape: {tensor.shape}")
    print(f"  Has NaN: {has_nan}")
    print(f"  Has Inf: {has_inf}")
    print(f"  Min: {min_val:.6f}")
    print(f"  Max: {max_val:.6f}")
    print(f"  Mean: {mean_val:.6f}")
    print(f"  Std: {std_val:.6f}")
    
    if has_nan or has_inf:
        print(f"  ⚠️  WARNING: {name} contains NaN or Inf values!")
    
    return has_nan or has_inf

def main():
    print("=" * 60)
    print("Navier-Stokes Training NaN Diagnostic")
    print("=" * 60)
    
    # Load config
    config = make_config_from_cli(Default)
    config = config.to_dict()
    
    print("\n1. Loading dataset...")
    data_dir = Path(config.data.folder).expanduser()
    
    train_loader, test_loaders, data_processor = load_navier_stokes_pt(
        data_root=data_dir,
        train_resolution=config.data.train_resolution,
        n_train=min(100, config.data.n_train),  # Just check first 100 samples
        batch_size=config.data.batch_size,
        test_resolutions=config.data.test_resolutions,
        n_tests=[100],  # Just check 100 test samples
        test_batch_sizes=config.data.test_batch_sizes,
        encode_input=config.data.encode_input,
        encode_output=config.data.encode_output,
        num_workers=0,
        pin_memory=False,
    )
    
    print("\n2. Checking raw data...")
    sample = next(iter(train_loader))
    x_raw = sample['x']
    y_raw = sample['y']
    
    check_tensor("Raw Input (x)", x_raw)
    check_tensor("Raw Target (y)", y_raw)
    
    print("\n3. Checking data processor normalization...")
    data_processor.training = True
    data_processor = data_processor.to('cpu')
    
    sample_processed = data_processor.preprocess(sample.copy(), batched=True)
    x_processed = sample_processed['x']
    y_processed = sample_processed['y']
    
    check_tensor("Processed Input (x)", x_processed)
    check_tensor("Processed Target (y)", y_processed)
    
    print("\n4. Checking model initialization...")
    device = torch.device('cpu')
    model = get_model(config)
    model = model.to(device)
    
    # Check model parameters
    total_params = 0
    nan_params = 0
    inf_params = 0
    
    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            nan_params += param.numel()
            print(f"  ⚠️  Parameter {name} contains NaN!")
        if torch.isinf(param).any():
            inf_params += param.numel()
            print(f"  ⚠️  Parameter {name} contains Inf!")
        total_params += param.numel()
    
    print(f"  Total parameters: {total_params:,}")
    print(f"  NaN parameters: {nan_params:,}")
    print(f"  Inf parameters: {inf_params:,}")
    
    print("\n5. Checking model forward pass...")
    model.eval()
    with torch.no_grad():
        out = model(x_processed)
        check_tensor("Model Output", out)
        
        # Check if output matches target shape
        if out.shape != y_processed.shape:
            print(f"  ⚠️  Shape mismatch! Output: {out.shape}, Target: {y_processed.shape}")
    
    print("\n6. Checking loss computation...")
    from neuralop import H1Loss
    h1loss = H1Loss(d=2)
    
    try:
        loss = h1loss(out, y_processed)
        check_tensor("H1 Loss", loss)
    except Exception as e:
        print(f"  ❌ Error computing loss: {e}")
    
    print("\n" + "=" * 60)
    print("Diagnostic complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()



