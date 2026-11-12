
"""
This file defines a utils module where independent generic functions can be pulled and used throughout the rest of the folder.

Function of Functions
    - get_device: returns a torch.device, the available device in order cuda > mps > cpu
    - get_time: returns a formatted str of the current time (America/New_York) ex: y2026_m09_d18_h15_m37_s33
    - move_to_device: moves any tuple | list of tensors to device and keeps original input containers (tensor, [tensor]) -> (tensor, [tensor])
    - generate_test_dataset: calls agreed on model internal model.predict(batch) model should then figure out how to index the tensor. this method
        only hands the model the full data chunk. splitting the data into inpt / target is the job of internals of model.
        returns (predicted, truth), concatenated to [timesteps, x, y, channels] when autocat else as lists of batches
    - quick_train: bare AdamW loop over a loader with a user closure returning the loss
    - mat_to_h5: converts a .mat file to .h5, one dataset per key
    - get_optimizer: optimizer by name (AdamW, Adam, SGD, RMSprop, Adagrad)
    - qr_pts: QR pivoted sensor points (wip)
    - patch_indexer: samples patch / trace token pairs from a data split (wip)
    - build_results_table: walks every Case_* run in an outputs folder and writes one csv row per run (errors, param count, best epoch, losses, ...),
        rows grouped by model then case. returns the list of row dicts

Wrappers:
    - time_function: wraps any function and prints the execution time of that function
    - auto_self_assign: class decorator that stores every __init__ argument (defaults included) on self before __init__ runs

Globals:
    - debug: when True, prints the device, time and movement traffic

"""

import os
import re
import csv
import json
import time
import h5py
import torch
import scipy.io
import functools
import inspect
import numpy as np
from typing import Callable
from datetime import datetime
from zoneinfo import ZoneInfo
import torch.nn.functional as F

# --- Developer Debug Flags ---
debug = False 

def get_device() -> torch.device:
    """Device selection: CUDA (NVIDIA) > MPS (Apple Silicon) > CPU"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    if debug: print(f"[device]: using device: {device}")
    return device

def get_time() -> str:
    """Returns string of current date and time formatted as y{Y}_m{M}_d{D}_h{H}_m{M}_s{S}, ex. y2026_m09_d18_h00_m49_s54"""
    time = datetime.now(ZoneInfo("America/New_York")).strftime("y%Y_m%m_d%d_h%H_m%M_s%S")
    if debug: print(f"[time]: {time}")
    return time

def time_function(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        time_start = time.perf_counter()

        out = fn(*args, **kwargs)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        time_end = time.perf_counter()

        print(f"[time_function]: runtime ({fn.__name__}) = {1000 * (time_end - time_start):.4f} ms")
        return out

    return wrapper

def move_to_device(device: str, *args:  torch.Tensor | list | tuple) -> tuple:

    """Takes in any tensors, or (nested) lists / tuples of tensors -> returns a tuple of the args with every tensor on device, containers keep their type"""
    non_blocking = torch.device(device).type not in ["cpu", "mps"]
    if debug: print(f"[move_to_device]: moving {len(args)} args to {device} | non_blocking = {non_blocking}")

    def _move(arg):

        if isinstance(arg, torch.Tensor):
            if debug: print(f"[move_to_device]: tensor {tuple(arg.shape)} | {arg.device} -> {device}")
            return arg.to(device, non_blocking=non_blocking)

        if isinstance(arg, (list, tuple)):
            return type(arg)(_move(a) for a in arg)
        
        return arg

    return tuple(_move(arg) for arg in args)

def generate_test_dataset(model: torch.nn.Module, test_dataloader, device, autocat: bool = True, detach_and_cpu: bool = True) -> torch.Tensor | list[torch.tensor]:
    """loops over loader and builds (predicted, truth), both [timesteps, x, y, channels] in loader order"""
    model.eval()
    tups = []
    with torch.no_grad():
        for batch in test_dataloader:
            batch, = move_to_device(device, batch)
            if debug: print(f"[generate_test_dataset]: batch shape = {tuple(batch.shape)} | device = {batch.device}")
            
            if hasattr(model, "inference"):
                tup = model.inference(batch)
            else:
                tup = model.forward(batch)

            if debug: print(f"[generate_test_dataset]: pred shape = {tuple(tup[0].shape)} | target shape = {tuple(tup[1].shape)}")
            
            assert(isinstance(tup, tuple))
            assert(isinstance(tup[0], torch.Tensor)) #! first is output
            assert(isinstance(tup[1], torch.Tensor)) #! second is target
            
            tups.append(tuple(t.clone() for t in tup)) # cudagraphs (max-autotune) reuse output buffers across calls

    if debug: print(f"[generate_test_dataset]: collected {len(tups)} batches | autocat = {autocat}")

    if autocat:
        if detach_and_cpu:
            return tuple(torch.cat(col, dim=0).detach().cpu() for col in zip(*tups))
        else:
            return tuple(torch.cat(col, dim=0) for col in zip(*tups))
    else:
        return tuple(map(list, zip(*tups)))

def get_optimizer(params, optimizer: str, lr: float, weight_decay: float):
    if optimizer == "AdamW":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    elif optimizer == "Adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    elif optimizer == "SGD":
        return torch.optim.SGD(params, lr=lr, weight_decay=weight_decay, momentum=0.9)
    elif optimizer == "RMSprop":
        return torch.optim.RMSprop(params, lr=lr, weight_decay=weight_decay)
    elif optimizer == "Adagrad":
        return torch.optim.Adagrad(params, lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(
            f"Unsupported optimizer {optimizer!r}. Choose from: "
            "'Adam', 'AdamW', 'SGD', 'RMSprop', or 'Adagrad'."
        )

def auto_self_assign(cls):
    init = cls.__init__
    signature = inspect.signature(init)
        
    @functools.wraps(init)
    def new_init(self, *args, **kwargs):
        bound = signature.bind(self, *args, **kwargs)
        bound.apply_defaults()
        for k, v in bound.arguments.items():
            if k != 'self':
                setattr(self, k, v)
        init(self, *args, **kwargs)
            
    cls.__init__ = new_init
    return cls

def to_numpy(x: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float64)

if __name__ == "__main__":
    pass