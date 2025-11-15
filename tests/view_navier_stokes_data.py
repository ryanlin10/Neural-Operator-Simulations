"""
Script to view and visualize Navier-Stokes training data.

This script loads the downloaded Navier-Stokes dataset and displays:
- Sample input/output pairs
- Data statistics
- Visualization of vorticity fields
"""

import sys
from pathlib import Path
import torch
import matplotlib.pyplot as plt
import numpy as np

# Add the neuraloperator to path
sys.path.insert(0, str(Path(__file__).parent / "clone" / "neuraloperator"))

from neuralop.data.datasets.navier_stokes import NavierStokesDataset


def view_data(data_dir="~/data/navier_stokes/", resolution=128, n_samples=5):
    """
    Load and visualize Navier-Stokes data.
    
    Parameters
    ----------
    data_dir : str
        Directory where data is stored
    resolution : int
        Resolution of data to load (128 or 1024)
    n_samples : int
        Number of samples to visualize
    """
    data_dir = Path(data_dir).expanduser()
    
    print(f"Loading Navier-Stokes data from: {data_dir}")
    print(f"Resolution: {resolution}")
    print(f"Visualizing {n_samples} samples\n")
    
    # Check if data exists
    train_file = data_dir / f"nsforcing_train_{resolution}.pt"
    test_file = data_dir / f"nsforcing_test_{resolution}.pt"
    
    if not train_file.exists():
        print(f"ERROR: Training data file not found: {train_file}")
        print("\nThe data will be downloaded automatically when you run the training script.")
        print("Or you can manually download from: https://zenodo.org/records/12825163")
        return
    
    # Load a small dataset to visualize
    dataset = NavierStokesDataset(
        root_dir=data_dir,
        n_train=n_samples,
        n_tests=[n_samples],
        batch_size=1,
        test_batch_sizes=[1],
        train_resolution=resolution,
        test_resolutions=[resolution],
        encode_input=False,  # Don't normalize for visualization
        encode_output=False,
        download=False,  # Assume already downloaded
    )
    
    # Get training data
    train_db = dataset.train_db
    
    print(f"Dataset loaded successfully!")
    print(f"Training samples: {len(train_db)}")
    print(f"Test samples: {len(dataset.test_dbs[resolution])}\n")
    
    # Visualize samples
    fig, axes = plt.subplots(n_samples, 2, figsize=(12, 3 * n_samples))
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle(f"Navier-Stokes Data (Resolution: {resolution}x{resolution})", 
                 fontsize=14, fontweight='bold')
    
    for i in range(min(n_samples, len(train_db))):
        sample = train_db[i]
        
        # Get input (x) and output (y)
        # Data format: [batch, channels, height, width]
        x = sample['x']  # Input: initial vorticity field
        y = sample['y']  # Output: evolved vorticity field
        
        # Remove batch and channel dimensions for visualization
        if x.ndim == 4:
            x = x.squeeze(0).squeeze(0)  # Remove batch and channel dims
        elif x.ndim == 3:
            x = x.squeeze(0)  # Remove channel dim
        
        if y.ndim == 4:
            y = y.squeeze(0).squeeze(0)
        elif y.ndim == 3:
            y = y.squeeze(0)
        
        # Convert to numpy for plotting
        x_np = x.cpu().numpy() if torch.is_tensor(x) else x
        y_np = y.cpu().numpy() if torch.is_tensor(y) else y
        
        # Plot input (initial condition)
        im1 = axes[i, 0].imshow(x_np, cmap='RdBu_r', origin='lower')
        axes[i, 0].set_title(f'Sample {i+1}: Input (Initial Vorticity)', fontsize=10)
        axes[i, 0].set_xlabel('x')
        axes[i, 0].set_ylabel('y')
        plt.colorbar(im1, ax=axes[i, 0], fraction=0.046, pad=0.04)
        
        # Plot output (evolved state)
        im2 = axes[i, 1].imshow(y_np, cmap='RdBu_r', origin='lower')
        axes[i, 1].set_title(f'Sample {i+1}: Output (Evolved Vorticity)', fontsize=10)
        axes[i, 1].set_xlabel('x')
        axes[i, 1].set_ylabel('y')
        plt.colorbar(im2, ax=axes[i, 1], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.show()
    
    # Print statistics
    print("\n" + "="*60)
    print("DATA STATISTICS")
    print("="*60)
    
    # Collect all data for statistics
    all_inputs = []
    all_outputs = []
    
    for i in range(min(100, len(train_db))):  # Sample up to 100 for stats
        sample = train_db[i]
        x = sample['x']
        y = sample['y']
        
        if torch.is_tensor(x):
            all_inputs.append(x.cpu().numpy())
        else:
            all_inputs.append(x)
            
        if torch.is_tensor(y):
            all_outputs.append(y.cpu().numpy())
        else:
            all_outputs.append(y)
    
    all_inputs = np.array(all_inputs)
    all_outputs = np.array(all_outputs)
    
    print(f"\nInput (Initial Vorticity) Statistics:")
    print(f"  Shape: {all_inputs.shape}")
    print(f"  Min: {all_inputs.min():.4f}")
    print(f"  Max: {all_inputs.max():.4f}")
    print(f"  Mean: {all_inputs.mean():.4f}")
    print(f"  Std: {all_inputs.std():.4f}")
    
    print(f"\nOutput (Evolved Vorticity) Statistics:")
    print(f"  Shape: {all_outputs.shape}")
    print(f"  Min: {all_outputs.min():.4f}")
    print(f"  Max: {all_outputs.max():.4f}")
    print(f"  Mean: {all_outputs.mean():.4f}")
    print(f"  Std: {all_outputs.std():.4f}")
    
    print("\n" + "="*60)
    print(f"\nData location: {data_dir}")
    print(f"Files:")
    for f in sorted(data_dir.glob("*.pt")):
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  - {f.name} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="View Navier-Stokes dataset")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="~/data/navier_stokes/",
        help="Directory containing the data (default: ~/data/navier_stokes/)"
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=128,
        choices=[128, 1024],
        help="Resolution of data to load (default: 128)"
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=5,
        help="Number of samples to visualize (default: 5)"
    )
    
    args = parser.parse_args()
    
    view_data(
        data_dir=args.data_dir,
        resolution=args.resolution,
        n_samples=args.n_samples
    )

