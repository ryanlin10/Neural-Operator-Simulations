from timeit import default_timer
from pathlib import Path
from typing import Union
import sys
import warnings

import torch
from torch.cuda import amp
from torch import nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Only import wandb and use if installed
wandb_available = False
try:
    import wandb

    wandb_available = True
except ModuleNotFoundError:
    wandb_available = False

import neuralop.mpu.comm as comm
from neuralop.losses import LpLoss
from .training_state import load_training_state, save_training_state


class Trainer:
    """
    A general Trainer class to train neural-operators on given datasets.

    .. note ::
        Our Trainer expects datasets to provide batches as key-value dictionaries, ex.:
        ``{'x': x, 'y': y}``, that are keyed to the arguments expected by models and losses.
        For specifics and an example, check ``neuralop.data.datasets.DarcyDataset``.

    Parameters
    ----------
    model : nn.Module
    n_epochs : int
    wandb_log : bool, default is False
        whether to log results to wandb
    device : torch.device, or str 'cpu' or 'cuda'
    mixed_precision : bool, default is False
        whether to use torch.autocast to compute mixed precision
    data_processor : DataProcessor class to transform data, default is None
        if not None, data from the loaders is transform first with data_processor.preprocess,
        then after getting an output from the model, that is transformed with data_processor.postprocess.
    eval_interval : int, default is 1
        how frequently to evaluate model and log training stats
    log_output : bool, default is False
        if True, and if wandb_log is also True, log output images to wandb
    use_distributed : bool, default is False
        whether to use DDP
    verbose : bool, default is False
    """

    def __init__(
        self,
        *,
        model: nn.Module,
        n_epochs: int,
        wandb_log: bool = False,
        device: str = "cpu",
        mixed_precision: bool = False,
        data_processor: nn.Module = None,
        eval_interval: int = 1,
        log_output: bool = False,
        use_distributed: bool = False,
        verbose: bool = False,
        grad_clip: float = None,
    ):
        """ """

        self.model = model
        self.n_epochs = n_epochs
        # only log to wandb if a run is active
        self.wandb_log = False
        if wandb_available:
            self.wandb_log = wandb_log and wandb.run is not None
        self.eval_interval = eval_interval
        self.log_output = log_output
        self.verbose = verbose
        self.use_distributed = use_distributed
        self.device = device
        self.grad_clip = grad_clip  # Gradient clipping value (None to disable)
        # handle autocast device
        if isinstance(self.device, torch.device):
            self.autocast_device_type = self.device.type
        else:
            if "cuda" in self.device:
                self.autocast_device_type = "cuda"
            else:
                self.autocast_device_type = "cpu"
        self.mixed_precision = mixed_precision
        self.data_processor = data_processor

        # Track starting epoch for checkpointing/resuming
        self.start_epoch = 0

    def _clip_gradients(self):
        """Clip gradients, handling both real and complex-valued gradients.
        
        PyTorch's clip_grad_norm_ doesn't support complex gradients yet,
        so we need a custom implementation that handles both cases.
        """
        if self.grad_clip is None:
            return
        
        # Collect all gradients (both real and complex)
        # Initialize as tensor to ensure consistent dtype/device
        total_norm_squared = torch.tensor(0.0, device=self.device, dtype=torch.float32)
        
        for param in self.model.parameters():
            if param.grad is not None:
                grad = param.grad
                if torch.is_complex(grad):
                    # For complex gradients, compute squared norm (|a+bi|^2 = a^2 + b^2)
                    # Use view_as_real to get real and imaginary parts
                    grad_real = torch.view_as_real(grad)
                    # grad_real has shape [..., 2] where last dim is [real, imag]
                    total_norm_squared = total_norm_squared + grad_real.pow(2).sum()
                else:
                    # For real gradients, compute squared norm
                    total_norm_squared = total_norm_squared + grad.pow(2).sum()
        
        # Compute total norm
        total_norm = torch.sqrt(total_norm_squared + 1e-6)
        
        # Clip if norm exceeds threshold
        if total_norm.item() > self.grad_clip:
            clip_coef = torch.tensor(self.grad_clip / total_norm.item(), device=self.device, dtype=torch.float32)
            for param in self.model.parameters():
                if param.grad is not None:
                    # Both real and complex gradients can be multiplied by scalar
                    param.grad.mul_(clip_coef)

    def train(
        self,
        train_loader,
        test_loaders,
        optimizer,
        scheduler,
        regularizer=None,
        training_loss=None,
        eval_losses=None,
        eval_modes=None,
        save_every: int = None,
        save_best: int = None,
        save_dir: Union[str, Path] = "./ckpt",
        resume_from_dir: Union[str, Path] = None,
        max_autoregressive_steps: int = None,
    ):
        """Trains the given model on the given dataset.

        If a device is provided, the model and data processor are loaded to device here.

        Parameters
        -----------
        train_loader: torch.utils.data.DataLoader
            training dataloader
        test_loaders: dict[torch.utils.data.DataLoader]
            testing dataloaders
        optimizer: torch.optim.Optimizer
            optimizer to use during training
        scheduler: torch.optim.lr_scheduler
            learning rate scheduler to use during training
        training_loss: training.losses function
            cost function to minimize
        eval_losses: dict[Loss]
            dict of losses to use in self.eval()
        eval_modes: dict[str], optional
            optional mapping from the name of each loader to its evaluation mode.

            * if 'single_step', predicts one input-output pair and evaluates loss.

            * if 'autoregressive', autoregressively predicts output using last step's
            output as input for a number of steps defined by the temporal dimension of the batch.
            This requires specially batched data with a data processor whose ``.preprocess`` and
            ``.postprocess`` both take ``idx`` as an argument.
        save_every: int, optional, default is None
            if provided, interval at which to save checkpoints
        save_best: str, optional, default is None
            if provided, key of metric f"{loader_name}_{loss_name}"
            to monitor and save model with best eval result
            Overrides save_every and saves on eval_interval
        save_dir: str | Path, default "./ckpt"
            directory at which to save training states if
            save_every and/or save_best is provided
        resume_from_dir: str | Path, default None
            if provided, resumes training state (model, optimizer, regularizer, scheduler) 
            from state saved in `resume_from_dir`
        max_autoregressive_steps : int, default None
            if provided, and a dataloader is to be evaluated in autoregressive mode,
            limits the number of autoregressive in each rollout to be performed.

        Returns
        -------
        all_metrics: dict
            dictionary keyed f"{loader_name}_{loss_name}"
            of metric results for last validation epoch across
            all test_loaders

        """
        self.optimizer = optimizer
        self.scheduler = scheduler
        if regularizer:
            self.regularizer = regularizer
        else:
            self.regularizer = None

        if training_loss is None:
            training_loss = LpLoss(d=2)

        # Warn the user if training loss is reducing across the batch
        if hasattr(training_loss, "reduction"):
            if training_loss.reduction == "mean":
                warnings.warn(
                    f"{training_loss.reduction=}. This means that the loss is "
                    "initialized to average across the batch dim. The Trainer "
                    "expects losses to sum across the batch dim."
                )

        if eval_losses is None:  # By default just evaluate on the training loss
            eval_losses = dict(l2=training_loss)

        # accumulated wandb metrics
        self.wandb_epoch_metrics = None

        # create default eval modes
        if eval_modes is None:
            eval_modes = {}

        # attributes for checkpointing
        self.save_every = save_every
        self.save_best = save_best
        if resume_from_dir is not None:
            self.resume_state_from_dir(resume_from_dir)

        # Load model and data_processor to device
        self.model = self.model.to(self.device)

        if self.use_distributed and dist.is_initialized():
            device_id = dist.get_rank()
            self.model = DDP(self.model, device_ids=[device_id], output_device=device_id)

        if self.data_processor is not None:
            self.data_processor = self.data_processor.to(self.device)

        # ensure save_best is a metric we collect
        if self.save_best is not None:
            metrics = []
            for name in test_loaders.keys():
                for metric in eval_losses.keys():
                    metrics.append(f"{name}_{metric}")
            assert (
                self.save_best in metrics
            ), f"Error: expected a metric of the form <loader_name>_<metric>, got {save_best}"
            best_metric_value = float("inf")
            # either monitor metric or save on interval, exclusive for simplicity
            self.save_every = None

        if self.verbose:
            print(f"Training on {len(train_loader.dataset)} samples")
            print(f"Testing on {[len(loader.dataset) for loader in test_loaders.values()]} samples"
                f"         on resolutions {[name for name in test_loaders]}.")
            sys.stdout.flush()

        for epoch in range(self.start_epoch, self.n_epochs):
            (
                train_err,
                avg_loss,
                avg_lasso_loss,
                epoch_train_time,
            ) = self.train_one_epoch(epoch, train_loader, training_loss)
            epoch_metrics = dict(
                train_err=train_err,
                avg_loss=avg_loss,
                avg_lasso_loss=avg_lasso_loss,
                epoch_train_time=epoch_train_time,
            )

            if epoch % self.eval_interval == 0:
                # evaluate and gather metrics across each loader in test_loaders
                eval_metrics = self.evaluate_all(
                    epoch=epoch,
                    eval_losses=eval_losses,
                    test_loaders=test_loaders,
                    eval_modes=eval_modes,
                    max_autoregressive_steps=max_autoregressive_steps,
                )
                epoch_metrics.update(**eval_metrics)
                # save checkpoint if conditions are met
                if save_best is not None:
                    if eval_metrics[save_best] < best_metric_value:
                        best_metric_value = eval_metrics[save_best]
                        self.checkpoint(save_dir)

            # save checkpoint if save_every and save_best is not set
            if self.save_every is not None:
                if epoch % self.save_every == 0:
                    self.checkpoint(save_dir)

        return epoch_metrics

    def train_one_epoch(self, epoch, train_loader, training_loss):
        """train_one_epoch trains self.model on train_loader
        for one epoch and returns training metrics

        Parameters
        ----------
        epoch : int
            epoch number
        train_loader : torch.utils.data.DataLoader
            data loader of train examples
        test_loaders : dict
            dict of test torch.utils.data.DataLoader objects

        Returns
        -------
        all_errors
            dict of all eval metrics for the last epoch
        """
        self.on_epoch_start(epoch)
        avg_loss = 0
        avg_lasso_loss = 0
        self.model.train()
        if self.data_processor:
            self.data_processor.train()
        t1 = default_timer()
        train_err = 0.0

        # track number of training examples in batch
        self.n_samples = 0

        for idx, sample in enumerate(train_loader):
            loss = self.train_one_batch(idx, sample, training_loss)
            loss.backward()
            
            # Simple progress indicator every 100 batches
            if idx > 0 and idx % 100 == 0:
                print(f"  [Epoch {self.epoch}] Processed {idx} batches, current loss: {loss.item():.6f}")
            
            # Check gradients before clipping (only for first batch to avoid slowdown)
            if self.epoch == 0 and idx == 0:
                has_nan_grad = False
                max_grad_norm = 0.0
                for name, param in self.model.named_parameters():
                    if param.grad is not None:
                        grad = param.grad
                        if torch.isnan(grad).any() or torch.isinf(grad).any():
                            print(f"  ⚠️  NaN/Inf gradient in {name}!")
                            has_nan_grad = True
                        # Compute norm handling both real and complex gradients
                        if torch.is_complex(grad):
                            grad_real = torch.view_as_real(grad)
                            grad_norm = torch.sqrt(grad_real.pow(2).sum()).item()
                        else:
                            grad_norm = torch.sqrt(grad.pow(2).sum()).item()
                        max_grad_norm = max(max_grad_norm, grad_norm)
                if has_nan_grad:
                    print(f"  ⚠️  NaN/Inf gradients detected before clipping!")
                else:
                    print(f"  ✓ Gradients OK before clipping (max norm: {max_grad_norm:.6f})")
            
            # Gradient clipping to prevent explosion
            # Handle complex gradients which PyTorch's clip_grad_norm_ doesn't support
            if self.grad_clip is not None:
                self._clip_gradients()
            
            self.optimizer.step()
            
            # Check weights after optimizer step (only for first batch to avoid slowdown)
            if self.epoch == 0 and idx == 0:
                has_nan_param = False
                for name, param in self.model.named_parameters():
                    if torch.isnan(param.data).any() or torch.isinf(param.data).any():
                        print(f"  ⚠️  NaN/Inf in parameter {name} after optimizer step!")
                        has_nan_param = True
                if has_nan_param:
                    print(f"  ⚠️  Model weights corrupted after optimizer step!")
                else:
                    print(f"  ✓ Model weights OK after optimizer step")
            
            self.optimizer.zero_grad()

            train_err += loss.item()
            with torch.no_grad():
                avg_loss += loss.item()
                if self.regularizer:
                    avg_lasso_loss += self.regularizer.loss

        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step(train_err)
        else:
            self.scheduler.step()

        epoch_train_time = default_timer() - t1

        train_err /= len(train_loader)
        avg_loss /= self.n_samples
        if self.regularizer:
            avg_lasso_loss /= self.n_samples
        else:
            avg_lasso_loss = None

        lr = None
        for pg in self.optimizer.param_groups:
            lr = pg["lr"]
        if self.verbose and epoch % self.eval_interval == 0:
            self.log_training(
                epoch=epoch,
                time=epoch_train_time,
                avg_loss=avg_loss,
                train_err=train_err,
                avg_lasso_loss=avg_lasso_loss,
                lr=lr,
            )

        return train_err, avg_loss, avg_lasso_loss, epoch_train_time

    def evaluate_all(
        self,
        epoch,
        eval_losses,
        test_loaders,
        eval_modes,
        max_autoregressive_steps=None,
    ):
        """evaluate_all iterates through the entire dict of test_loaders
        to perform evaluation on the whole dataset stored in each one.

        Parameters
        ----------
        epoch : int
            current training epoch
        eval_losses : dict[Loss]
            keyed ``loss_name: loss_obj`` for each pair. Full set of
            losses to use in evaluation for each test loader.
        test_loaders : dict[DataLoader]
            keyed ``loader_name: loader`` for each test loader.
        eval_modes : dict[str], optional
            keyed ``loader_name: eval_mode`` for each test loader.
            * If ``eval_modes.get(loader_name)`` does not return a value,
            the evaluation is automatically performed in ``single_step`` mode.
        max_autoregressive_steps : ``int``, optional
            if provided, and one of the test loaders has ``eval_mode == "autoregressive"``,
            limits the number of autoregressive steps performed per rollout.

        Returns
        -------
        all_metrics: dict
            collected eval metrics for each loader.
        """
        # evaluate and gather metrics across each loader in test_loaders
        all_metrics = {}
        for loader_name, loader in test_loaders.items():
            loader_eval_mode = eval_modes.get(loader_name, "single_step")
            loader_metrics = self.evaluate(
                eval_losses,
                loader,
                log_prefix=loader_name,
                mode=loader_eval_mode,
                max_steps=max_autoregressive_steps,
            )
            all_metrics.update(**loader_metrics)
        if self.verbose:
            self.log_eval(epoch=epoch, eval_metrics=all_metrics)
        return all_metrics

    def evaluate(
        self,
        loss_dict,
        data_loader,
        log_prefix="",
        epoch=None,
        mode="single_step",
        max_steps=None,
    ):
        """Evaluates the model on a dictionary of losses

        Parameters
        ----------
        loss_dict : dict of functions
          each function takes as input a tuple (prediction, ground_truth)
          and returns the corresponding loss
        data_loader : data_loader to evaluate on
        log_prefix : str, default is ''
            if not '', used as prefix in output dictionary
        epoch : int | None
            current epoch. Used when logging both train and eval
            default None
        mode : Literal {'single_step', 'autoregression'}
            if 'single_step', performs standard evaluation
            if 'autoregression' loops through `max_steps` steps
        max_steps : int, optional
            max number of steps for autoregressive rollout.
            If None, runs the full rollout.
        Returns
        -------
        errors : dict
            dict[f'{log_prefix}_{loss_name}] = loss for loss in loss_dict
        """
        # Ensure model and data processor are loaded to the proper device

        self.model = self.model.to(self.device)
        if self.data_processor is not None and self.data_processor.device != self.device:
            self.data_processor = self.data_processor.to(self.device)

        self.model.eval()
        if self.data_processor:
            self.data_processor.eval()

        errors = {f"{log_prefix}_{loss_name}": 0 for loss_name in loss_dict.keys()}

        # Warn the user if any of the eval losses is reducing across the batch
        for _, eval_loss in loss_dict.items():
            if hasattr(eval_loss, "reduction"):
                if eval_loss.reduction == "mean":
                    warnings.warn(
                        f"{eval_loss.reduction=}. This means that the loss is "
                        "initialized to average across the batch dim. The Trainer "
                        "expects losses to sum across the batch dim."
                    )

        self.n_samples = 0
        with torch.no_grad():
            for idx, sample in enumerate(data_loader):
                return_output = False
                if idx == len(data_loader) - 1:
                    return_output = True
                if mode == "single_step":
                    eval_step_losses, outs = self.eval_one_batch(
                        sample, loss_dict, return_output=return_output
                    )
                elif mode == "autoregression":
                    eval_step_losses, outs = self.eval_one_batch_autoreg(
                        sample,
                        loss_dict,
                        return_output=return_output,
                        max_steps=max_steps,
                    )

                for loss_name, val_loss in eval_step_losses.items():
                    errors[f"{log_prefix}_{loss_name}"] += val_loss

        for key in errors.keys():
            errors[key] /= self.n_samples

        # on last batch, log model outputs
        if self.log_output and self.wandb_log:
            errors[f"{log_prefix}_outputs"] = wandb.Image(outs)

        return errors

    def on_epoch_start(self, epoch):
        """on_epoch_start runs at the beginning
        of each training epoch. This method is a stub
        that can be overwritten in more complex cases.

        Parameters
        ----------
        epoch : int
            index of epoch

        Returns
        -------
        None
        """
        self.epoch = epoch
        return None

    def train_one_batch(self, idx, sample, training_loss):
        """Run one batch of input through model
           and return training loss on outputs

        Parameters
        ----------
        idx : int
            index of batch within train_loader
        sample : dict
            data dictionary holding one batch

        Returns
        -------
        loss: float | Tensor
            float value of training loss
        """

        self.optimizer.zero_grad(set_to_none=True)
        if self.regularizer:
            self.regularizer.reset()
        if self.data_processor is not None:
            sample = self.data_processor.preprocess(sample)
        else:
            # load data to device if no preprocessor exists
            sample = {k: v.to(self.device) for k, v in sample.items() if torch.is_tensor(v)}

        # Debug: Check for NaN in input data and normalization stats
        if self.epoch == 0 and idx == 0:
            x = sample.get("x", None)
            y = sample.get("y", None)
            if x is not None and torch.is_tensor(x):
                has_nan = torch.isnan(x).any()
                has_inf = torch.isinf(x).any()
                if has_nan or has_inf:
                    print(f"⚠️  WARNING: Input x contains NaN/Inf! NaN: {has_nan}, Inf: {has_inf}")
                else:
                    x_min, x_max, x_mean = x.min().item(), x.max().item(), x.mean().item()
                    print(f"✓ Input x is valid: shape={x.shape}, min={x_min:.6f}, max={x_max:.6f}, mean={x_mean:.6f}")
                    # Check for extreme values that might cause numerical issues
                    if abs(x_max) > 100 or abs(x_min) > 100:
                        print(f"   ⚠️  WARNING: Input x has extreme values (>100), might cause numerical instability")
            if y is not None and torch.is_tensor(y):
                has_nan = torch.isnan(y).any()
                has_inf = torch.isinf(y).any()
                if has_nan or has_inf:
                    print(f"⚠️  WARNING: Target y contains NaN/Inf! NaN: {has_nan}, Inf: {has_inf}")
                else:
                    y_min, y_max, y_mean = y.min().item(), y.max().item(), y.mean().item()
                    print(f"✓ Target y is valid: shape={y.shape}, min={y_min:.6f}, max={y_max:.6f}, mean={y_mean:.6f}")
            
            # Check normalization statistics if data processor exists
            if self.data_processor is not None:
                if hasattr(self.data_processor, 'in_normalizer') and self.data_processor.in_normalizer is not None:
                    in_mean = self.data_processor.in_normalizer.mean
                    in_std = self.data_processor.in_normalizer.std
                    print(f"✓ Input normalizer: mean shape={in_mean.shape if hasattr(in_mean, 'shape') else 'scalar'}, "
                          f"std min={in_std.min().item() if hasattr(in_std, 'min') else in_std:.6f}, "
                          f"std max={in_std.max().item() if hasattr(in_std, 'max') else in_std:.6f}")
                    if hasattr(in_std, 'min') and in_std.min().item() < 1e-6:
                        print(f"   ⚠️  WARNING: Input std is very small (<1e-6), normalization might produce extreme values")
                if hasattr(self.data_processor, 'out_normalizer') and self.data_processor.out_normalizer is not None:
                    out_mean = self.data_processor.out_normalizer.mean
                    out_std = self.data_processor.out_normalizer.std
                    print(f"✓ Output normalizer: mean shape={out_mean.shape if hasattr(out_mean, 'shape') else 'scalar'}, "
                          f"std min={out_std.min().item() if hasattr(out_std, 'min') else out_std:.6f}, "
                          f"std max={out_std.max().item() if hasattr(out_std, 'max') else out_std:.6f}")
                    if hasattr(out_std, 'min') and out_std.min().item() < 1e-6:
                        print(f"   ⚠️  WARNING: Output std is very small (<1e-6), normalization might produce extreme values")

        if isinstance(sample["y"], torch.Tensor):
            self.n_samples += sample["y"].shape[0]
        else:
            self.n_samples += 1

        # Extract 'x' from sample - model only expects input, not target 'y'
        x = sample.get("x", None)
        if x is None:
            raise ValueError("Sample dictionary must contain 'x' key for model input")
        
        # Debug: Check model weights for NaN before forward pass and test with dummy data
        if self.epoch == 0 and idx == 0:
            has_nan_params = False
            for name, param in self.model.named_parameters():
                if param.requires_grad and torch.isnan(param).any():
                    print(f"⚠️  WARNING: Parameter {name} contains NaN!")
                    has_nan_params = True
            if not has_nan_params:
                print("✓ All model parameters are valid (no NaN)")
            
            # Test model with dummy data to see if it's a data issue or model issue
            print("\n🔍 Testing model with dummy data...")
            dummy_x = torch.randn_like(x)
            self.model.eval()
            with torch.no_grad():
                dummy_out = self.model(dummy_x)
                if torch.isnan(dummy_out).any():
                    print(f"⚠️  Model produces NaN even with random dummy data!")
                    print(f"   This suggests a model architecture issue, not a data issue.")
                else:
                    print(f"✓ Model works fine with dummy data (output shape: {dummy_out.shape})")
            self.model.train()
            
            # Add forward hooks to trace where NaN appears - more comprehensive
            nan_detected_in = []
            
            def check_nan_hook(name):
                def hook(module, input, output):
                    if isinstance(output, torch.Tensor):
                        if torch.isnan(output).any() or torch.isinf(output).any():
                            if name not in nan_detected_in:
                                nan_detected_in.append(name)
                                print(f"\n⚠️  NaN/Inf FIRST detected in {name}!")
                                print(f"   Output shape: {output.shape}")
                                print(f"   Has NaN: {torch.isnan(output).any().item()}, Has Inf: {torch.isinf(output).any().item()}")
                                if isinstance(input, tuple) and len(input) > 0 and isinstance(input[0], torch.Tensor):
                                    inp = input[0]
                                    print(f"   Input stats: min={inp.min().item():.6f}, max={inp.max().item():.6f}, "
                                          f"mean={inp.mean().item():.6f}, has_nan={torch.isnan(inp).any().item()}")
                return hook
            
            # Register hooks on key modules
            hooks = []
            if hasattr(self.model, 'positional_embedding') and self.model.positional_embedding is not None:
                hooks.append(self.model.positional_embedding.register_forward_hook(check_nan_hook('positional_embedding')))
            if hasattr(self.model, 'lifting'):
                hooks.append(self.model.lifting.register_forward_hook(check_nan_hook('lifting')))
            if hasattr(self.model, 'fno_blocks'):
                hooks.append(self.model.fno_blocks.register_forward_hook(check_nan_hook('fno_blocks')))
            if hasattr(self.model, 'projection'):
                hooks.append(self.model.projection.register_forward_hook(check_nan_hook('projection')))
            
            # Also check intermediate outputs manually by wrapping forward
            print(f"\n[DEBUG] Testing model forward pass step-by-step...")
            print(f"  Input x shape: {x.shape}, dtype: {x.dtype}")
            print(f"  Input x stats: min={x.min().item():.6f}, max={x.max().item():.6f}, mean={x.mean().item():.6f}")
            
            # Test each component manually through the entire forward pass
            test_x = x.clone()
            nan_found = False
            
            if hasattr(self.model, 'positional_embedding') and self.model.positional_embedding is not None:
                test_x = self.model.positional_embedding(test_x)
                if torch.isnan(test_x).any():
                    print(f"  ⚠️  NaN after positional_embedding!")
                    nan_found = True
                else:
                    print(f"  ✓ positional_embedding OK: shape={test_x.shape}")
            
            if not nan_found and hasattr(self.model, 'lifting'):
                test_x = self.model.lifting(test_x)
                if torch.isnan(test_x).any():
                    print(f"  ⚠️  NaN after lifting!")
                    nan_found = True
                else:
                    print(f"  ✓ lifting OK: shape={test_x.shape}, stats: min={test_x.min().item():.6f}, max={test_x.max().item():.6f}")
            
            # Test domain padding if it exists
            if not nan_found and hasattr(self.model, 'domain_padding') and self.model.domain_padding is not None:
                test_x = self.model.domain_padding.pad(test_x)
                if torch.isnan(test_x).any():
                    print(f"  ⚠️  NaN after domain_padding.pad!")
                    nan_found = True
                else:
                    print(f"  ✓ domain_padding.pad OK: shape={test_x.shape}")
            
            # Test FNO blocks layer by layer
            if not nan_found and hasattr(self.model, 'fno_blocks') and hasattr(self.model.fno_blocks, 'n_layers'):
                for layer_idx in range(self.model.fno_blocks.n_layers):
                    test_x_before = test_x.clone()
                    test_x = self.model.fno_blocks(test_x, layer_idx)
                    if torch.isnan(test_x).any():
                        print(f"  ⚠️  NaN after fno_blocks layer {layer_idx}!")
                        print(f"     Input stats before layer: min={test_x_before.min().item():.6f}, max={test_x_before.max().item():.6f}")
                        nan_found = True
                        break
                    else:
                        print(f"  ✓ fno_blocks layer {layer_idx} OK: shape={test_x.shape}, stats: min={test_x.min().item():.6f}, max={test_x.max().item():.6f}")
            
            # Test domain unpadding if it exists
            if not nan_found and hasattr(self.model, 'domain_padding') and self.model.domain_padding is not None:
                test_x = self.model.domain_padding.unpad(test_x)
                if torch.isnan(test_x).any():
                    print(f"  ⚠️  NaN after domain_padding.unpad!")
                    nan_found = True
                else:
                    print(f"  ✓ domain_padding.unpad OK: shape={test_x.shape}")
            
            # Test projection
            if not nan_found and hasattr(self.model, 'projection'):
                test_x_before = test_x.clone()
                test_x = self.model.projection(test_x)
                if torch.isnan(test_x).any():
                    print(f"  ⚠️  NaN after projection!")
                    print(f"     Input stats before projection: min={test_x_before.min().item():.6f}, max={test_x_before.max().item():.6f}")
                    nan_found = True
                else:
                    print(f"  ✓ projection OK: shape={test_x.shape}, stats: min={test_x.min().item():.6f}, max={test_x.max().item():.6f}")
            
            if not nan_found:
                print(f"  ✓ Full forward pass completed successfully!")
            
            # Reset test_x for actual forward pass
            test_x = None
        else:
            hooks = []
        
        # Check input before forward pass (only for first batch to avoid slowdown)
        if torch.is_tensor(x) and (self.epoch == 0 and idx == 0):
            x_has_nan = torch.isnan(x).any().item()
            x_has_inf = torch.isinf(x).any().item()
            x_max_abs = x.abs().max().item()
            x_min, x_max, x_mean = x.min().item(), x.max().item(), x.mean().item()
            print(f"\n[DEBUG Epoch {self.epoch}, Batch {idx}] Input x check:")
            print(f"  Shape: {x.shape}, dtype: {x.dtype}")
            print(f"  Stats: min={x_min:.6f}, max={x_max:.6f}, mean={x_mean:.6f}, max_abs={x_max_abs:.6f}")
            print(f"  Has NaN: {x_has_nan}, Has Inf: {x_has_inf}")
            if x_has_nan or x_has_inf or x_max_abs > 1e6:
                print(f"  ⚠️  WARNING: Input x has issues!")
                if x_max_abs > 1e6:
                    print(f"  ⚠️  Extreme values detected! This will likely cause NaN in model.")
            print()
        
        try:
            if self.mixed_precision:
                with torch.autocast(device_type=self.autocast_device_type):
                    out = self.model(x)
            else:
                out = self.model(x)
        finally:
            # Remove hooks after forward pass
            if self.epoch == 0 and idx == 0:
                for hook in hooks:
                    hook.remove()
        
        # Debug: Check for NaN in model output immediately after forward pass
        if isinstance(out, torch.Tensor):
            if torch.isnan(out).any() or torch.isinf(out).any():
                # Only print detailed warning for first few batches to avoid clutter
                if self.epoch == 0 and idx < 3:
                    print(f"\n{'='*60}")
                    print(f"⚠️  CRITICAL: Model output contains NaN/Inf!")
                    print(f"   Epoch: {self.epoch}, Batch: {idx}")
                    print(f"   Output shape: {out.shape}")
                    print(f"   Has NaN: {torch.isnan(out).any().item()}, Has Inf: {torch.isinf(out).any().item()}")
                    # Check input to model
                    if torch.is_tensor(x):
                        print(f"   Input x stats: min={x.min().item():.6f}, max={x.max().item():.6f}, mean={x.mean().item():.6f}")
                        print(f"   Input x has NaN: {torch.isnan(x).any().item()}, Inf: {torch.isinf(x).any().item()}")
                        # Check for extreme values
                        if x.abs().max() > 100:
                            print(f"   ⚠️  Input has extreme values (max abs: {x.abs().max().item():.2f})")
                    print(f"{'='*60}\n")

        if self.data_processor is not None:
            out, sample = self.data_processor.postprocess(out, sample)

        loss = 0.0

        # Fix: Pass y explicitly as positional argument to match loss function signature
        y = sample.get("y", None)
        if y is None:
            raise ValueError("Sample dictionary must contain 'y' key for loss computation")
        
        if self.mixed_precision:
            with torch.autocast(device_type=self.autocast_device_type):
                loss_val = training_loss(out, y)
        else:
            loss_val = training_loss(out, y)
        
        # Debug: Check for NaN in loss
        if torch.is_tensor(loss_val):
            if torch.isnan(loss_val) or torch.isinf(loss_val):
                # Only print detailed diagnostics once per epoch to avoid spam
                if idx == 0:
                    print(f"\n⚠️  WARNING: Loss is NaN/Inf! Loss value: {loss_val}")
                    print(f"   Epoch: {self.epoch}, Batch: {idx}")
                    if torch.is_tensor(out):
                        print(f"   Output stats: min={out.min().item():.6f}, max={out.max().item():.6f}, mean={out.mean().item():.6f}")
                        print(f"   Output has NaN: {torch.isnan(out).any().item()}, Inf: {torch.isinf(out).any().item()}")
                    if torch.is_tensor(y):
                        print(f"   Target stats: min={y.min().item():.6f}, max={y.max().item():.6f}, mean={y.mean().item():.6f}")
                        print(f"   Target has NaN: {torch.isnan(y).any().item()}, Inf: {torch.isinf(y).any().item()}")
                    # Check if shapes match
                    if torch.is_tensor(out) and torch.is_tensor(y):
                        if out.shape != y.shape:
                            print(f"   ⚠️  Shape mismatch! Output: {out.shape}, Target: {y.shape}")
        
        loss += loss_val

        if self.regularizer:
            loss += self.regularizer.loss

        return loss

    def eval_one_batch(
        self, sample: dict, eval_losses: dict, return_output: bool = False
    ):
        """eval_one_batch runs inference on one batch
        and returns eval_losses for that batch.

        Parameters
        ----------
        sample : dict
            data batch dictionary
        eval_losses : dict
            dictionary of named eval metrics
        return_outputs : bool
            whether to return model outputs for plotting
            by default False
        Returns
        -------
        eval_step_losses : dict
            keyed "loss_name": step_loss_value for each loss name
        outputs: torch.Tensor | None
            optionally returns batch outputs
        """
        if self.data_processor is not None:
            sample = self.data_processor.preprocess(sample)
        else:
            # load data to device if no preprocessor exists
            sample = {k: v.to(self.device) for k, v in sample.items() if torch.is_tensor(v)}

        self.n_samples += sample["y"].size(0)

        # Extract 'x' from sample - model only expects input, not target 'y'
        x = sample.get("x", None)
        if x is None:
            raise ValueError("Sample dictionary must contain 'x' key for model input")
        out = self.model(x)

        if self.data_processor is not None:
            out, sample = self.data_processor.postprocess(out, sample)

        eval_step_losses = {}

        # Fix: Pass y explicitly as positional argument
        y = sample.get("y", None)
        if y is None:
            raise ValueError("Sample dictionary must contain 'y' key for loss computation")
        
        for loss_name, loss in eval_losses.items():
            val_loss = loss(out, y)
            eval_step_losses[loss_name] = val_loss

        if return_output:
            return eval_step_losses, out
        else:
            return eval_step_losses, None

    def eval_one_batch_autoreg(
        self,
        sample: dict,
        eval_losses: dict,
        return_output: bool = False,
        max_steps: int = None,
    ):
        """eval_one_batch runs inference on one batch
        and returns eval_losses for that batch.

        Parameters
        ----------
        sample : dict
            data batch dictionary
        eval_losses : dict
            dictionary of named eval metrics
        return_outputs : bool
            whether to return model outputs for plotting
            by default False
        max_steps: int
            number of timesteps to roll out
            typically the full trajectory length
            If max_steps is none, runs until the full length

            .. note::
                If a value for ``max_steps`` is not provided, a data_processor
                must be provided to handle rollout logic.
        Returns
        -------
        eval_step_losses : dict
            keyed "loss_name": step_loss_value for each loss name
        outputs: torch.Tensor | None
            optionally returns batch outputs


        """
        eval_step_losses = {loss_name: 0.0 for loss_name in eval_losses.keys()}
        # eval_rollout_losses = {loss_name: 0. for loss_name in eval_losses.keys()}

        t = 0
        if max_steps is None:
            max_steps = float("inf")

        # only increment the sample count once
        sample_count_incr = False

        while sample is not None and t < max_steps:
            if self.data_processor is not None:
                sample = self.data_processor.preprocess(sample, step=t)
            else:
                # load data to device if no preprocessor exists
                sample = {
                    k: v.to(self.device)
                    for k, v in sample.items()
                    if torch.is_tensor(v)
                }

            if sample is None:
                break

            # only increment the sample count once
            if not sample_count_incr:
                self.n_samples += sample["y"].shape[0]
                sample_count_incr = True

            # Extract 'x' from sample - model only expects input, not target 'y'
            x = sample.get("x", None)
            if x is None:
                raise ValueError("Sample dictionary must contain 'x' key for model input")
            out = self.model(x)

            if self.data_processor is not None:
                out, sample = self.data_processor.postprocess(out, sample, step=t)

            # Fix: Pass y explicitly as positional argument
            y = sample.get("y", None)
            if y is None:
                raise ValueError("Sample dictionary must contain 'y' key for loss computation")
            
            for loss_name, loss in eval_losses.items():
                step_loss = loss(out, y)
                eval_step_losses[loss_name] += step_loss

            t += 1
        # average over all steps of the final rollout
        for loss_name in eval_step_losses.keys():
            eval_step_losses[loss_name] /= t

        if return_output:
            return eval_step_losses, out
        else:
            return eval_step_losses, None

    def log_training(
        self,
        epoch: int,
        time: float,
        avg_loss: float,
        train_err: float,
        avg_lasso_loss: float = None,
        lr: float = None,
    ):
        """Basic method to log results
        from a single training epoch.


        Parameters
        ----------
        epoch: int
        time: float
            training time of epoch
        avg_loss: float
            average train_err per individual sample
        train_err: float
            train error for entire epoch
        avg_lasso_loss: float
            average lasso loss from regularizer, optional
        lr: float
            learning rate at current epoch
        """
        # accumulate info to log to wandb
        if self.wandb_log:
            values_to_log = dict(
                train_err=train_err,
                time=time,
                avg_loss=avg_loss,
                avg_lasso_loss=avg_lasso_loss,
                lr=lr,
            )

        msg = f"[{epoch}] time={time:.2f}, "
        msg += f"avg_loss={avg_loss:.4f}, "
        msg += f"train_err={train_err:.4f}"
        if avg_lasso_loss is not None:
            msg += f", avg_lasso={avg_lasso_loss:.4f}"

        print(msg)
        sys.stdout.flush()

        if self.wandb_log:
            wandb.log(data=values_to_log, step=epoch + 1, commit=False)

    def log_eval(self, epoch: int, eval_metrics: dict):
        """log_eval logs outputs from evaluation
        on all test loaders to stdout and wandb

        Parameters
        ----------
        epoch : int
            current training epoch
        eval_metrics : dict
            metrics collected during evaluation
            keyed f"{test_loader_name}_{metric}" for each test_loader

        """
        values_to_log = {}
        msg = ""
        for metric, value in eval_metrics.items():
            if isinstance(value, float) or isinstance(value, torch.Tensor):
                msg += f"{metric}={value:.4f}, "
            if self.wandb_log:
                values_to_log[metric] = value

        msg = f"Eval: " + msg[:-2]  # cut off last comma+space
        print(msg)
        sys.stdout.flush()

        if self.wandb_log:
            wandb.log(data=values_to_log, step=epoch + 1, commit=True)

    def resume_state_from_dir(self, save_dir):
        """
        Resume training from save_dir created by `neuralop.training.save_training_state`

        Params
        ------
        save_dir: Union[str, Path]
            directory in which training state is saved
            (see neuralop.training.training_state)
        """
        if isinstance(save_dir, str):
            save_dir = Path(save_dir)

        # check for save model exists
        if (save_dir / "best_model_state_dict.pt").exists():
            save_name = "best_model"
        elif (save_dir / "model_state_dict.pt").exists():
            save_name = "model"
        else:
            raise FileNotFoundError(
                "Error: resume_from_dir expects a model\
                                        state dict named model.pt or best_model.pt."
            )
        # returns model, loads other modules if provided
        
        (
            self.model,
            self.optimizer,
            self.scheduler,
            self.regularizer,
            resume_epoch,
        ) = load_training_state(
            save_dir=save_dir,
            save_name=save_name,
            model=self.model,
            optimizer=self.optimizer,
            regularizer=self.regularizer,
            scheduler=self.scheduler,
        )

        if resume_epoch is not None:
            if resume_epoch > self.start_epoch:
                self.start_epoch = resume_epoch
                if self.verbose:
                    print(f"Trainer resuming from epoch {resume_epoch}")

    def checkpoint(self, save_dir):
        """checkpoint saves current training state
        to a directory for resuming later. Only saves
        training state on the first GPU.
        See neuralop.training.training_state

        Parameters
        ----------
        save_dir : str | Path
            directory in which to save training state
        """
        if comm.get_local_rank() == 0:
            if self.save_best is not None:
                save_name = "best_model"
            else:
                save_name = "model"
            save_training_state(
                save_dir=save_dir,
                save_name=save_name,
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                regularizer=self.regularizer,
                epoch=self.epoch,
            )
            if self.verbose:
                print(f"[Rank 0]: saved training state to {save_dir}")
