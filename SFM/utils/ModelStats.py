
"""
ModelStats defines a class that tracks named scalars, lists, dicts, tuples and tensors in one dictionary.

Function of functions
    - .track() : registers a name to be tracked, as an empty list
    - .append() : adds a value to the list of a tracked name by its string name, tensors are detached and moved to the cpu (single element tensors become floats)
    - .update() : lets you completely rewrite a name's value, tracked or not, if you don't want to append to it
    - .get() : returns the value stored under a name
    - .save() : saves all tracked parameters except "best_model" to model_tracking.json inside the given root_path
    - .auto_clean() : removes any empty lists, lists of tensors become lists of np arrays, any other list becomes a one item list holding one np array

Properties
    - .history is the dictionary (self.tracking) that tracks all the items.

Globals
    - debug: when True, prints the tracking traffic

"""

import os
import json
import torch
import numpy as np
from dataclasses import dataclass, field

# --- debug flag ---
debug: bool = False

@dataclass()
class ModelStats():
    # --- types of trackable data ---

    # pre-clean()
    #static: {name: item} item type of any (str, int, float, bool)
    #temporal-resolved: list of type of any (list: int|float, tensor, np.array)

    # post-clean()
    #static: {name: item} item type of any (str, int, float, bool)
    #temporal-resolved: {name: value} value of type np.array [time, number_of_items]

    tracking: dict = field(default_factory=dict)

    def track(self, name: str):
        if debug: print(f"[ModelStats]: began tracking {name}")
        self.tracking[name] = []

    def append(self, name: str, value):
        if debug: print(f"[ModelStats]: appending value to {name}")
        self.tracking[name].append(ModelStats._detach(value))

    def update(self, name: str, value):
        if debug: print(f"[ModelStats]: updating {name}")
        self.tracking[name] = value

    def get(self, name):
        if debug: print(f"[ModelStats]: returning value of {name}")
        return self.tracking[name]

    @staticmethod
    def _detach(value):
        # scalars become floats, tensors leave the graph / device, containers recurse
        if isinstance(value, torch.Tensor):
            return value.item() if value.numel() == 1 else value.detach().cpu()
        
        if isinstance(value, (list, tuple)):
            return [ModelStats._detach(v) for v in value]
        
        return value

    def auto_clean(self):
        if debug: print(f"[ModelStats]: auto cleaning.")
        for key in list(self.tracking):
            value = self.tracking[key]

            if not isinstance(value, list): 
                continue

            else:
                if len(value) == 0:
                    self.tracking.pop(key)
                    if debug: print(f"[ModelStats]: found nothing stored in tracking[{key}], cleaning.")

                elif isinstance(value[0], str): 
                    continue

                elif isinstance(value[0], torch.Tensor):
                    self.tracking[key] = torch.stack(value, dim=0).numpy()

                else:
                    self.tracking[key] = np.array(value)
                
    @property
    def history(self):
        self.auto_clean()
        return self.tracking

    def save(self, root_path):
        # the state dict is written by model.save, everything else is json-able once tensors are listed
        self.auto_clean()
        path = os.path.join(root_path, "model_tracking.json")
        with open(path, "w") as f:
            json.dump({k: v for k, v in self.tracking.items() if k != "best_model"}, f, indent=2, default=lambda o: o.tolist())

        if debug: print(f"[ModelStats]: saved history to {path}")