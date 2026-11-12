
"""
This file holds the training loop and the training diagnostics that plot it. Pipe (pipeline.py) calls train, then plot_all_tracked on the
returned tracker's history.

Function of Functions
    - train: AdamW + cosine annealing loop over any model that follows the contract of ModelClass (models/abstract.py). per epoch it trains over the
        trainloader, evaluates the validationloader, keeps the best state dict by test loss and stops early after patience epochs without
        improvement, or as soon as a file named STOP exists next to checkpoint_path. every batch it tracks the loss components and the optimizer momentum norm, every grad_tracking_frequence batches
        it also tracks the per-loss gradient norms and their pairwise cosine similarities. returns (model with the best weights loaded,
        ModelStats). the first _tup_unpack_to_losses items of model.forward's tuple go to model.losses. without use_auto_weighting the
        loss list is summed as is, custom weights are the use_* values passed to losses (e.g. use_smooth=1e-3)
    - plot_all_tracked: plots every tracked quantity with simple_line (plots.py), per epoch / per batch / per gradient sample
    - _get_optimizer_mom: global l2 norm of the optimizer's first moment buffers
    - _per_loss_grads: flat gradient and gradient norm of each loss term w.r.t. the trainable params
    - _cos_similarities: [n_losses, n_losses] cosine similarity between the flat per-loss gradients

Globals:
    - debug: turns on autograd anomaly detection

"""

import os
import tqdm
import torch
import numpy as np
import torch.nn as nn
from copy import deepcopy
from typing import Callable
from SFM.utils.utils import *
from SFM.utils.ModelStats import ModelStats
from SFM.utils.plots import simple_line
from transformers import get_cosine_schedule_with_warmup

# --- Developer Debug Flags ---
debug = False
torch.autograd.set_detect_anomaly(debug)

#! Rules:
#! 1) training only ever uses model.forward method. Method should take one argument being the full batch of data.
#! 2) model.forward should only ever return a tuple(tensor, tensor, ... other) : the first two should always be (prediction, target)
#! 3) this is fead directly unpacked into model.losses(*tup[:_tup_unpack_to_losses]) | _tup_unpack_to_losses = 2 (loss_fn replaces model.losses when given)

def train(
    model:                  torch.nn.Module,
    trainloader:            torch.utils.data.DataLoader,
    validationloader:       torch.utils.data.DataLoader,
    learning_rate:          float,
    weight_decay:           float,
    epochs:                 int,
    patience:               int,
    device:                 torch.device | str,
    grad_tracking_frequency:int,
    use_auto_weighting:     bool     = False,
    use_lr_warmup:          bool     = False,
    num_warmup_steps:       int      = 20,
    min_weight:             float    = 10e-3,
    max_weight:             float    = 10e3,
    checkpoint_path:        str      = None,
    stagepoint_root:        str      = None,
    stagepoint_start:       int      = 2,
    stagepoint_frequency:   int      = 20,
    _tup_unpack_to_losses:  int      = 2,
    speedy:                 bool     = True,
    loss_fn:                Callable = None     #? (output, target) -> (loss list, names), defaults to model.losses

    ) -> tuple[torch.nn.Module, ModelStats]:
    if speedy: print("[Training]: Reduced tracking of metrics, speedy is active!")
    if loss_fn is None: loss_fn = model.losses

# ─-- training loop ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    if use_lr_warmup:
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=epochs)
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = epochs, eta_min=1e-6)

    epoch_bar = tqdm.trange(epochs, colour="blue", desc="Training", leave=False)
    params    = [p for p in model.parameters() if p.requires_grad]
    patience_counter = 0
    weights = None
    stop_file = os.path.join(os.path.dirname(checkpoint_path), "STOP")
    print(f"[Training]: to stop at the end of the current epoch | touch {stop_file}")

    if use_auto_weighting:
        print(f"[Training]: Warning use_auto_weighting depends on grad_tracking_frequency = {grad_tracking_frequency} batches per weight update!")
        
# --- params being tracker ---
    tracker = ModelStats()
    for name in ["lr", 
                 "overall_grad_norm", 
                 "overall_momentum_norm",
                 "gradnorm_weights", 
                 "overall_trainloss", 
                 "overall_testloss", 
                 "train_loss_list", 
                 "test_loss_list", 
                 "momentum_norm", 
                 "grad_norm", 
                 "grad_norms_per_loss", 
                 "cosine_similarities"]:
        
        tracker.track(name)

    tracker.update("best_model", deepcopy(model.state_dict()))
    tracker.update("epoch_best_model", 0)
    tracker.update("epoch_earlystop", None)
    tracker.update("trainloader_size", len(trainloader))
    tracker.update("validationloader_size", len(validationloader))

    grad_tracking_counter = 0

    for epoch in epoch_bar:
# --- train ---
        model.train()
        epoch_train_loss    = 0.0
        momentum_norms      = 0.0
        overall_grad_norms  = 0.0

        for batch in trainloader:
            batch, = move_to_device(device, batch)

            optimizer.zero_grad()
            #! ----------------------
            
            tup = model.forward(batch) #! model_out, target
            loss_list, loss_names = loss_fn(*tup[:_tup_unpack_to_losses])

            if grad_tracking_frequency != 0 and grad_tracking_counter % grad_tracking_frequency == 0:
                if not speedy: 
                    grads, norms = _per_loss_grads(params, losses=loss_list, return_stacked_norms=False)

                if use_auto_weighting:
                    norm_sum = sum(norms)
                    weights = [min(max(n/norm_sum, min_weight), max_weight) for n in norms] # (AWD: https://doi.org/10.1007/s41060-026-01234-2)

                if not speedy: 
                    tracker.append("grad_norms_per_loss", torch.stack(norms))
                    tracker.append("cosine_similarities", _cos_similarities(grads))

            grad_tracking_counter += 1

            if use_auto_weighting:
                if weights is None: weights = [1.0] * len(loss_list) #! initialize at 1.0 for all weights
                train_loss = sum(l * w for l, w in zip(loss_list, weights))
            else:
                train_loss = sum(loss_list) #! custom weights are the use_* values in losses

            #! ----------------------
            momentum_norm =  _get_optimizer_mom(optimizer)
            tracker.append("momentum_norm",momentum_norm)
            tracker.append("train_loss_list", loss_list)

            train_loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, error_if_nonfinite=True)
            optimizer.step()

            overall_grad_norms  += grad_norm.item()
            momentum_norms      += momentum_norm
            epoch_train_loss    += train_loss.detach().item()

            epoch_bar.set_postfix_str(f"Train Loss = {train_loss.item():.4f}")
            tracker.append("grad_norm", grad_norm)
        
        if use_auto_weighting: tracker.append("gradnorm_weights", weights)
        tracker.append("overall_momentum_norm", momentum_norms / len(trainloader))
        tracker.append("overall_grad_norm", overall_grad_norms / len(trainloader))
        tracker.append("overall_trainloss", epoch_train_loss / len(trainloader))
        tracker.append("lr", float(optimizer.param_groups[0]["lr"]))
        scheduler.step()

# --- test ---
        model.eval()
        epoch_test_loss = 0.0
        with torch.no_grad():
            for batch in validationloader:
                batch, = move_to_device(device, batch)
                #! ----------------------
                
                tup = model.forward(batch) #! model_out, target
                loss_list, _ = loss_fn(*tup[:_tup_unpack_to_losses])

                test_loss = sum(l * w for l, w in zip(loss_list, weights)) if use_auto_weighting else sum(loss_list)

                #! ----------------------
                tracker.append("test_loss_list", loss_list)
                epoch_test_loss += test_loss.item()

        epoch_test_loss /= len(validationloader)
        tracker.append("overall_testloss", epoch_test_loss)
        epoch_bar.set_postfix_str(f"Val Loss = {epoch_test_loss:.4f}")

# ── early stopping ────────────────────────────────────────────────────
        if epoch % stagepoint_frequency == stagepoint_start:
            # these are save stages to learn about the training dynamics.
            torch.save({"model_state_dict": model.state_dict()}, os.path.join(stagepoint_root, f"stagepoint_{epoch}.pth"))
        
        if epoch_test_loss <= min(tracker.get("overall_testloss")):
            patience_counter = 0
            tracker.update("best_model", deepcopy(model.state_dict()))
            tracker.update("epoch_best_model", epoch)
            
            torch.save({
                        "epoch":                 epoch,
                        "model_state_dict":      model.state_dict(),
                        "optimizer_state_dict":  optimizer.state_dict(),
                        "scheduler_state_dict":  scheduler.state_dict(),
                    }, checkpoint_path)
        else:
            patience_counter += 1

        if patience_counter >= patience or os.path.exists(stop_file):
            print(f"[Training]: early stopping | epoch = {epoch}")
            tracker.update("epoch_earlystop", epoch)
            break

    tracker.update("loss_names", loss_names)
    print(f"[Training]: best model | epoch = {tracker.get("epoch_best_model") + 1}/{epoch + 1}")
    model.load_state_dict(tracker.get("best_model"))

    return model, tracker

def _get_optimizer_mom(optimizer: torch.optim.Optimizer) -> float:
    # global l2 norm over every parameter's first moment buffer 
    norms = []

    for group in optimizer.param_groups:
        for p in group['params']:
            state   = optimizer.state.get(p, {})
            mom_buf = state.get('momentum_buffer', state.get('exp_avg'))
            if mom_buf is not None:
                norms.append(torch.linalg.vector_norm(mom_buf))

    # no buffers yet (before the first optimizer step, or an optimizer without momentum)
    if len(norms) == 0: return 0.0

    return torch.linalg.vector_norm(torch.stack(norms)).item()

def _cos_similarities(grads: list[torch.Tensor]) -> torch.Tensor:
    # [n_losses, n_losses] cosine similarity between the flat per-loss gradients
    g = torch.stack(grads)
    g = g / (torch.linalg.vector_norm(g, dim=1, keepdim=True) + 1e-12)
    return (g @ g.T) #ie [n, n]

def _per_loss_grads(params: list[torch.Tensor], 
                    losses: torch.Tensor | list[torch.Tensor],
                    return_stacked_norms: bool = False
                    ) -> tuple[list[torch.Tensor], torch.Tensor]:

    grads, norms = [], []
        
    for loss in losses:
        # retain_graph so the next loss (and the caller's backward) can reuse the graph
        # create_graph is off since the norms are only observed, never trained on (unlike GradOptim)
        # materialized zeros for params a loss never touched so every flat grad has the same length
        grad = torch.autograd.grad(loss, params, 
                                    retain_graph=True, 
                                    create_graph=False,
                                    allow_unused=True, 
                                    materialize_grads=True)
                    
        flat = torch.cat([g.reshape(-1) for g in grad])
        grads.append(flat)
        norms.append(torch.linalg.vector_norm(flat))
    
    if return_stacked_norms:
        return grads, torch.stack(norms)
    else:
        return grads, norms

def plot_all_tracked(tracking_dict: dict, path: str):
    os.makedirs(path, exist_ok=True)
    best_ep     = tracking_dict["epoch_best_model"]
    t           = dict(tracking_dict)
    names       = t["loss_names"]

    # key, labels, title, x label, y label, logy, linestyle, x_line
    plots = [
        ("gradnorm_weights",        names,              "Loss Weights",             "Epoch",          "Weight",                  False, "-", best_ep),
        ("train_loss_list",         names,              "Train Loss Components",    "Batch step",     "Loss",                    True,  "-", None),
        ("test_loss_list",          names,              "Test Loss Components",     "Batch step",     "Loss",                    True,  "-", None),
        ("grad_norms_per_loss",     names,              "Per-Loss Gradient Norms",  "Tracked step",   "L2 norm",                 True,  "-", None),
        ("lr",                      ["Learning rate"],  "Learning Rate Schedule",   "Epoch",          "Learning rate",           True,  "-", best_ep),
        ("overall_trainloss",       ["Train loss"],     "Mean Train Loss",          "Epoch",          "Loss",                    True,  "-", best_ep),
        ("overall_testloss",        ["Test loss"],      "Mean Test Loss",           "Epoch",          "Loss",                    True,  "-", best_ep),
        ("momentum_norm",           ["Momentum norm"],  "Optimizer Momentum Norm",  "Batch step",     "L2 norm of first moment", True,  "-", None),
        ("overall_momentum_norm",   ["Momentum norm"],  "Optimizer Momentum Norm",  "Epoch",          "L2 norm of first moment", False, "-", best_ep),
        ("grad_norm",               ["Gradient norm"],  "Global Gradient Norm",     "Batch step",     "L2 norm",                 True,  "",  None),
        ("overall_grad_norm",       ["Gradient norm"],  "Global Gradient Norm",     "Epoch",          "L2 norm",                 False, "-", best_ep),
    ]

    if len(names) != 1 and "cosine_similarities" in t:
        i, j = np.triu_indices(len(names), k=1) 
        t["cosine_similarities"] = t["cosine_similarities"][:, i, j]
        plots.append(("cosine_similarities", [f"{names[a]} | {names[b]}" for a, b in zip(i, j)], "Pairwise Gradient Cosine Similarity", "Tracked sample", "Cosine similarity", False, "-", None))

    # one failed plot (missing key, bad shape, ...) is reported and skipped instead of killing the rest
    for key, labels, title, label_x, label_y, logy, ls, x_line in plots:
        try:
            simple_line(t[key], os.path.join(path, f"{key}_{get_time()}.png"), labels, title, label_x, label_y, logy=logy, ls=ls, x_line= x_line)
        except Exception as e:
            print(f"[plot_all_tracked]: skipped {key} | {type(e).__name__}: {e}")
