from timeit import default_timer
from pathlib import Path
from typing import Union
import sys
import warnings
import signal

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
        grad_noise_scale: float = None,
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
        self.grad_noise_scale = grad_noise_scale  # Gradient noise scale (None to disable) - helps escape sharp minima
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
        
        # Signal handling for graceful shutdown
        self._save_dir = None
        self._termination_saved = False
        self._original_sigint_handler = None
        self._original_sigterm_handler = None

    def _compute_gradient_norm(self):
        """Compute total gradient norm, handling both real and complex-valued gradients.
        
        Returns
        -------
        total_norm : torch.Tensor
            Total gradient norm
        """
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
        return torch.sqrt(total_norm_squared + 1e-6)
    
    def _clip_gradients(self, total_norm=None):
        """Clip gradients, handling both real and complex-valued gradients.
        
        PyTorch's clip_grad_norm_ doesn't support complex gradients yet,
        so we need a custom implementation that handles both cases.
        
        Parameters
        ----------
        total_norm : torch.Tensor, optional
            Pre-computed gradient norm. If None, will be computed.
        
        Returns
        -------
        was_clipped : bool
            True if gradients were clipped, False otherwise
        norm_value : float
            The gradient norm value
        """
        if self.grad_clip is None:
            if total_norm is not None:
                return False, total_norm.item()
            return False, 0.0
        
        # Compute norm if not provided
        if total_norm is None:
            total_norm = self._compute_gradient_norm()
        
        norm_value = total_norm.item()
        
        # Clip if norm exceeds threshold
        if norm_value > self.grad_clip:
            clip_coef = torch.tensor(self.grad_clip / norm_value, device=self.device, dtype=torch.float32)
            for param in self.model.parameters():
                if param.grad is not None:
                    # Both real and complex gradients can be multiplied by scalar
                    param.grad.mul_(clip_coef)
            return True, norm_value  # Return True if clipping occurred
        return False, norm_value  # Return False if no clipping

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
        self._save_dir = save_dir  # Store save_dir for signal handler
        if resume_from_dir is not None:
            self.resume_state_from_dir(resume_from_dir)
        
        # Register signal handlers for graceful shutdown
        self._register_signal_handlers()

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
                
                # Step scheduler with validation loss (for ReduceLROnPlateau)
                # Use the primary evaluation metric (first one in eval_losses, typically l2)
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    # Use validation L2 loss if available, otherwise use first eval metric
                    val_metric = None
                    for loader_name in test_loaders.keys():
                        metric_key = f"{loader_name}_l2"
                        if metric_key in eval_metrics:
                            val_metric = eval_metrics[metric_key]
                            break
                    if val_metric is None:
                        # Fallback to first available eval metric
                        val_metric = list(eval_metrics.values())[0]
                    
                    # Get LR before stepping
                    old_lr = self.optimizer.param_groups[0]['lr']
                    self.scheduler.step(val_metric)
                    new_lr = self.optimizer.param_groups[0]['lr']
                    
                    # Log LR reduction if it occurred
                    if new_lr < old_lr and self.verbose:
                        print(f"  Learning rate reduced: {old_lr:.2e} -> {new_lr:.2e} (validation loss: {val_metric:.4f})")
                
                # save checkpoint if conditions are met
                if save_best is not None:
                    if eval_metrics[save_best] < best_metric_value:
                        best_metric_value = eval_metrics[save_best]
                        self.checkpoint(save_dir)

            # save checkpoint if save_every and save_best is not set
            if self.save_every is not None:
                if epoch % self.save_every == 0:
                    self.checkpoint(save_dir)
            
            # Check if termination was requested
            if self._termination_saved:
                if self.verbose:
                    print("\n[Training interrupted] Checkpoint saved. Exiting gracefully...")
                break

        # Restore original signal handlers
        self._restore_signal_handlers()
        
        # If termination was requested, raise KeyboardInterrupt to exit main() cleanly
        if self._termination_saved:
            raise KeyboardInterrupt("Training interrupted by user")
        
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
            # Check if termination was requested (e.g., Ctrl+C)
            if self._termination_saved:
                if self.verbose:
                    print(f"\n[Training interrupted] Stopping at batch {idx} of epoch {self.epoch}...")
                break
            
            loss = self.train_one_batch(idx, sample, training_loss)
            loss.backward()
            
            # Simple progress indicator every 100 batches
            # Note: "current loss" is the loss for a single batch (summed across batch dimension)
            if idx > 0 and idx % 100 == 0:
                print(f"  [Epoch {self.epoch}] Processed {idx} batches, current batch loss: {loss.item():.6f}")
            
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
                # Compute gradient norm and clip
                total_norm = self._compute_gradient_norm()
                was_clipped, norm_value = self._clip_gradients(total_norm)
                
                # Log gradient norm every 100 batches
                if idx > 0 and idx % 100 == 0:
                    clipped_str = " (CLIPPED)" if was_clipped else ""
                    print(f"  [Epoch {self.epoch}, Batch {idx}] Gradient norm: {norm_value:.6f}{clipped_str}, clip_threshold: {self.grad_clip}")
            
            # Gradient noise injection to escape sharp minima
            # Add small random noise to gradients to help escape sharp minima and improve generalization
            if self.grad_noise_scale is not None and self.grad_noise_scale > 0:
                noise_scale = self.grad_noise_scale * (self.grad_clip if self.grad_clip is not None else 1.0)
                for param in self.model.parameters():
                    if param.grad is not None:
                        # Generate noise matching gradient shape and dtype
                        if torch.is_complex(param.grad):
                            # For complex gradients, add noise to both real and imaginary parts
                            grad_real_view = torch.view_as_real(param.grad)
                            noise_real = torch.randn_like(grad_real_view[..., 0]) * noise_scale
                            noise_imag = torch.randn_like(grad_real_view[..., 1]) * noise_scale
                            noise = torch.complex(noise_real, noise_imag)
                            param.grad.add_(noise)
                        else:
                            noise = torch.randn_like(param.grad) * noise_scale
                            param.grad.add_(noise)
            
            # Gradient distribution monitoring (every 100 batches)
            if idx > 0 and idx % 100 == 0 and self.verbose:
                layer_grads = {}
                for name, param in self.model.named_parameters():
                    if param.grad is not None:
                        # Compute gradient norm for this parameter
                        if torch.is_complex(param.grad):
                            grad_real = torch.view_as_real(param.grad)
                            grad_norm = torch.sqrt(grad_real.pow(2).sum()).item()
                        else:
                            grad_norm = param.grad.norm().item()
                        
                        # Group by top-level module name
                        layer_name = name.split('.')[0] if '.' in name else name
                        if layer_name not in layer_grads:
                            layer_grads[layer_name] = []
                        layer_grads[layer_name].append(grad_norm)
                
                # Print gradient distribution summary
                if layer_grads:
                    print(f"  [Epoch {self.epoch}, Batch {idx}] Gradient distribution by layer:")
                    for layer, norms in sorted(layer_grads.items()):
                        avg_norm = sum(norms) / len(norms)
                        max_norm = max(norms)
                        min_norm = min(norms)
                        print(f"    {layer}: avg={avg_norm:.4f}, min={min_norm:.4f}, max={max_norm:.4f} ({len(norms)} params)")
            
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

        epoch_train_time = default_timer() - t1

        # Normalize metrics before using them
        train_err /= len(train_loader)  # Average loss per batch
        avg_loss /= self.n_samples  # Average loss per sample
        if self.regularizer:
            avg_lasso_loss /= self.n_samples
        else:
            avg_lasso_loss = None

        # Step scheduler (ReduceLROnPlateau is stepped after evaluation in train() method)
        if not isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step()

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
                # Check if termination was requested (e.g., Ctrl+C)
                if self._termination_saved:
                    break
                
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
                    in_mean_val = in_mean.item() if in_mean.numel() == 1 else f"tensor shape {in_mean.shape}"
                    in_std_min = in_std.min().item() if hasattr(in_std, 'min') and in_std.numel() > 1 else in_std.item()
                    in_std_max = in_std.max().item() if hasattr(in_std, 'max') and in_std.numel() > 1 else in_std.item()
                    print(f"✓ Input normalizer: mean={in_mean_val}, "
                          f"std range=[{in_std_min:.6f}, {in_std_max:.6f}]")
                    if in_std_min < 1e-6:
                        print(f"   ⚠️  WARNING: Input std is very small (<1e-6), normalization might produce extreme values")
                    # Check normalized input range
                    x_normalized = self.data_processor.in_normalizer.transform(x)
                    x_norm_min = x_normalized.min().item()
                    x_norm_max = x_normalized.max().item()
                    
                    # Compute mean/std over the SAME dimensions the normalizer used
                    normalizer_dims = self.data_processor.in_normalizer.dim
                    if normalizer_dims is not None:
                        # Compute over the same dimensions (excluding channel dim)
                        # For channel-wise: dim=[0, 2, 3] means reduce over batch, height, width
                        x_norm_mean = x_normalized.mean(dim=normalizer_dims, keepdim=False)
                        x_norm_std = x_normalized.std(dim=normalizer_dims, keepdim=False)
                        # If result is still a tensor (multiple channels), take mean
                        if torch.is_tensor(x_norm_mean) and x_norm_mean.numel() > 1:
                            x_norm_mean = x_norm_mean.mean().item()
                            x_norm_std = x_norm_std.mean().item()
                        else:
                            x_norm_mean = x_norm_mean.item() if torch.is_tensor(x_norm_mean) else x_norm_mean
                            x_norm_std = x_norm_std.item() if torch.is_tensor(x_norm_std) else x_norm_std
                    else:
                        # Fallback to computing over all dims
                        x_norm_mean = x_normalized.mean().item()
                        x_norm_std = x_normalized.std().item()
                    
                    print(f"   Normalized input stats: min={x_norm_min:.3f}, max={x_norm_max:.3f}, mean={x_norm_mean:.3f}, std={x_norm_std:.3f}")
                    # Note: Single batch std won't be exactly 1.0 - it will only be 1.0 when computed over the full training set
                    # Check if mean is close to 0 (more important than std for a single batch)
                    if abs(x_norm_mean) > 0.1:
                        print(f"   ⚠️  WARNING: Normalized input mean is not ≈0 (mean={x_norm_mean:.3f})")
                    # Only warn about std if it's very far from 1.0 (single batches can vary)
                    if abs(x_norm_std - 1.0) > 0.5:
                        print(f"   ⚠️  NOTE: Normalized input std={x_norm_std:.3f} (single batch may vary; full dataset should be ≈1.0)")
                if hasattr(self.data_processor, 'out_normalizer') and self.data_processor.out_normalizer is not None:
                    out_mean = self.data_processor.out_normalizer.mean
                    out_std = self.data_processor.out_normalizer.std
                    out_mean_val = out_mean.item() if out_mean.numel() == 1 else f"tensor shape {out_mean.shape}"
                    out_std_min = out_std.min().item() if hasattr(out_std, 'min') and out_std.numel() > 1 else out_std.item()
                    out_std_max = out_std.max().item() if hasattr(out_std, 'max') and out_std.numel() > 1 else out_std.item()
                    print(f"✓ Output normalizer: mean={out_mean_val}, "
                          f"std range=[{out_std_min:.6f}, {out_std_max:.6f}]")
                    if out_std_min < 1e-6:
                        print(f"   ⚠️  WARNING: Output std is very small (<1e-6), normalization might produce extreme values")
                    # Check normalized output range
                    y_normalized = self.data_processor.out_normalizer.transform(y)
                    y_norm_min = y_normalized.min().item()
                    y_norm_max = y_normalized.max().item()
                    
                    # Compute mean/std over the SAME dimensions the normalizer used
                    normalizer_dims = self.data_processor.out_normalizer.dim
                    if normalizer_dims is not None:
                        # Compute over the same dimensions (excluding channel dim)
                        y_norm_mean = y_normalized.mean(dim=normalizer_dims, keepdim=False)
                        y_norm_std = y_normalized.std(dim=normalizer_dims, keepdim=False)
                        # If result is still a tensor (multiple channels), take mean
                        if torch.is_tensor(y_norm_mean) and y_norm_mean.numel() > 1:
                            y_norm_mean = y_norm_mean.mean().item()
                            y_norm_std = y_norm_std.mean().item()
                        else:
                            y_norm_mean = y_norm_mean.item() if torch.is_tensor(y_norm_mean) else y_norm_mean
                            y_norm_std = y_norm_std.item() if torch.is_tensor(y_norm_std) else y_norm_std
                    else:
                        # Fallback to computing over all dims
                        y_norm_mean = y_normalized.mean().item()
                        y_norm_std = y_normalized.std().item()
                    
                    print(f"   Normalized output stats: min={y_norm_min:.3f}, max={y_norm_max:.3f}, mean={y_norm_mean:.3f}, std={y_norm_std:.3f}")
                    # Note: Single batch std won't be exactly 1.0 - it will only be 1.0 when computed over the full training set
                    # Check if mean is close to 0 (more important than std for a single batch)
                    if abs(y_norm_mean) > 0.1:
                        print(f"   ⚠️  WARNING: Normalized output mean is not ≈0 (mean={y_norm_mean:.3f})")
                    # Only warn about std if it's very far from 1.0 (single batches can vary)
                    if abs(y_norm_std - 1.0) > 0.5:
                        print(f"   ⚠️  NOTE: Normalized output std={y_norm_std:.3f} (single batch may vary; full dataset should be ≈1.0)")

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
            # Check if termination was requested (e.g., Ctrl+C)
            if self._termination_saved:
                break
            
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
        msg += f"avg_loss={avg_loss:.4f}, "  # Average loss per sample
        msg += f"train_err={train_err:.4f}"  # Average loss per batch
        if avg_lasso_loss is not None:
            msg += f", avg_lasso={avg_lasso_loss:.4f}"
        if lr is not None:
            msg += f", lr={lr:.2e}"

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

        # check for save model exists (check in priority order)
        if (save_dir / "best_model_state_dict.pt").exists():
            save_name = "best_model"
        elif (save_dir / "model_state_dict.pt").exists():
            save_name = "model"
        elif (save_dir / "interrupted_model_state_dict.pt").exists():
            save_name = "interrupted_model"
            if self.verbose:
                print(f"Resuming from interrupted checkpoint: {save_dir}/interrupted_model_*")
        else:
            raise FileNotFoundError(
                "Error: resume_from_dir expects a model state dict named "
                "model_state_dict.pt, best_model_state_dict.pt, or interrupted_model_state_dict.pt."
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

        if self.verbose:
            print(f"✓ Loaded model parameters from checkpoint: {save_dir}/{save_name}_state_dict.pt")
            if self.optimizer is not None:
                print(f"✓ Loaded optimizer state from checkpoint")
            if self.scheduler is not None:
                print(f"✓ Loaded scheduler state from checkpoint")

        if resume_epoch is not None:
            if resume_epoch > self.start_epoch:
                self.start_epoch = resume_epoch
                if self.verbose:
                    print(f"Trainer resuming from epoch {resume_epoch}")

    def _handle_termination_signal(self, signum, frame):
        """Handle termination signals (SIGINT, SIGTERM) by saving checkpoint.
        
        This method is called when the user presses Ctrl+C (SIGINT) or when
        the process receives a SIGTERM signal. It saves the current training
        state so training can be resumed later.
        
        Parameters
        ----------
        signum : int
            Signal number (e.g., signal.SIGINT, signal.SIGTERM)
        frame : frame object
            Current stack frame (unused)
        """
        # Only save once, even if signal is received multiple times
        if self._termination_saved:
            # If already saved and user presses Ctrl+C again, exit immediately
            print("\n[Ctrl+C pressed again] Exiting immediately...")
            sys.stdout.flush()
            import os
            os._exit(1)
        
        # Print message immediately so user knows Ctrl+C was received
        print("\n[Ctrl+C received] Saving checkpoint and stopping training...")
        sys.stdout.flush()
        
        # Only save on rank 0 for distributed training
        if comm.get_local_rank() == 0 and self._save_dir is not None:
            try:
                # Save with special name to indicate interruption
                save_training_state(
                    save_dir=self._save_dir,
                    save_name="interrupted_model",
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    regularizer=self.regularizer,
                    epoch=self.epoch,
                )
                if self.verbose:
                    print(f"[Rank 0]: Saved interrupted training state to {self._save_dir}/interrupted_model_*")
                    print(f"         You can resume training using: resume_from_dir='{self._save_dir}'")
            except Exception as e:
                print(f"[ERROR] Failed to save checkpoint on termination: {e}")
        
        # Mark that we've handled the termination
        # The training loops will check this flag and exit gracefully
        self._termination_saved = True
    
    def _register_signal_handlers(self):
        """Register signal handlers for graceful shutdown."""
        # Store original handlers so we can restore them later
        self._original_sigint_handler = signal.signal(signal.SIGINT, self._handle_termination_signal)
        self._original_sigterm_handler = signal.signal(signal.SIGTERM, self._handle_termination_signal)
    
    def _restore_signal_handlers(self):
        """Restore original signal handlers after training completes."""
        if self._original_sigint_handler is not None:
            signal.signal(signal.SIGINT, self._original_sigint_handler)
        if self._original_sigterm_handler is not None:
            signal.signal(signal.SIGTERM, self._original_sigterm_handler)
        
        # Reset state
        self._termination_saved = False
        self._original_sigint_handler = None
        self._original_sigterm_handler = None

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
