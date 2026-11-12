
"""
This file defines Dataset, a torch Dataset wrapped around any .h5 file. It has two modes.

Mode 1 is a plain load and inspect of the file, every other attribute is forwarded to the open h5py.File.
Mode 2 is entered by calling __advanced_init__ which slices a chosen set of subdatasets, stacks them along a channel dim
and defines the time axis, after which the object indexes / builds loaders over a chunk of data:
    "chunk of data" : [batch, sequence, x, y, channels]

Function of Functions:

Mode 1:
    - keys: property, returns the keys of the .h5 file which are the "subdatasets" inside the file
    - info: prints each key, shape and dtype
    - getvar(key): returns a h5 pointer to the specific subdataset
    - save(file, data_dict): staticmethod, writes every tensor / np.ndarray in data_dict to a new .h5 file, skips anything else

Mode 2:
    - __advanced_init__(vars_of_interest, collapse_dims, collapse_idxs, timestep_dim): reads and slices each var (collapse_idxs are applied
        along collapse_dims), stacks them into self.data shaped [time, ..., channels] as float32, where time is moved from timestep_dim to dim 0
    - __len__ / __getitem__: number of timesteps / indexes self.data along time
    - build_dataloaders(data_split_ratios, batch_size, sequence_length): splits time into [train][val][test] in order, unfolds each split into
        overlapping windows and returns one DataLoader per ratio. each batch is one contiguous chunk, only the train loader shuffles
        (batch-to-batch, never inside a batch). only the sequential split is implemented

Globals:
    - debug: when True, prints the file, slicing and loader traffic

"""

import torch
import h5py
import traceback
import numpy as np
from torch.utils.data import DataLoader, Dataset as TorchDataset

# --- Developer Debug Flags ---
debug: bool = False

class Dataset(TorchDataset):
    def __init__(self, 
                 path : str, 
                ):

        self.path = path

        assert(h5py.is_hdf5(self.path)), f"[Dataset]: only h5 files are supported, could not open {self.path}"
        self.file = h5py.File(self.path, 'r')

        self.data = None
        self.advanced_available = False
        
        if self.file is None:
            raise AttributeError("[Dataset]: __init__ was unable to build self.file and returned None")

        if debug: print(f"[Dataset]: opened {self.path} | {len(self.file.keys())} keys")

    def __advanced_init__(self,
                          vars_of_interest  : tuple[str],
                          collapse_dims     : tuple[int] = (), 
                          collapse_idxs     : tuple[int | slice] = (),
                          timestep_dim      : int = 0,
                          ):

        if debug: print(f"[Dataset]: collapse_dims = {collapse_dims} | collapse_idxs = {collapse_idxs} | timestep_dim = {timestep_dim}")
        if debug: print(f"[Dataset]: vars_of_interest = {vars_of_interest}")
        if debug: print(f"[Dataset]: keys in file = {list(self.file.keys())}")

        idx = Dataset._convert_collapse(collapse_dims, collapse_idxs)
        tentative_shape  = None
        subdatasets     = []

        if debug: print(f"[Dataset]: slice index = {idx}")

        for var in vars_of_interest:
            try:
                subdataset = self.file[var][idx]
                shape = subdataset.shape
                subdatasets.append(torch.as_tensor(subdataset, dtype=torch.float32))

                if debug: print(f"[Dataset]: read var = {var} | sliced shape = {shape} | tentative_shape = {tentative_shape}")

                if tentative_shape is None:
                    tentative_shape = shape
                else:
                    assert(tentative_shape == shape), f"[Dataset]: Error, shape does not match tentative shape: var = {var} | tentative_shape = {tentative_shape} | shape = {shape}"

            except Exception as e:
                print(f"[Dataset]: Error, in getting var: {var} | {type(e).__name__}: {e}")
                if debug: traceback.print_exc()
                raise

        assert(timestep_dim not in collapse_dims), f"[Dataset]: timestep_dim = {timestep_dim} cannot be collapsed | collapse_dims = {collapse_dims}"

        self.advanced_available = True
        self.timesteps = tentative_shape[timestep_dim]
        self.data = torch.stack(subdatasets, dim=-1).movedim(timestep_dim, 0) # [time, ..., channels]

        if debug: print(f"[Dataset]: done | timesteps = {self.timesteps} | data shape = {tuple(self.data.shape)} | dtype = {self.data.dtype}")
        
    def __getattr__(self, name):
        if name in self.file.keys():
            return self.file[name]
        
        # only reached when normal lookup fails, forwards to the h5 file    
        return getattr(self.file, name)

    @property
    def keys(self):
        if debug: print(f"[Dataset]: returning keys of {self.path}")
        return self.file.keys()

    def info(self):
        if debug: print(f"[Dataset]: printing info for {len(self.file.keys())} keys")
        for key in self.file.keys():
            node = self.file[key]
            if isinstance(node, h5py.Dataset):
                print(f"[Dataset]: key = {key} | shape = {node.shape} | dtype = {node.dtype}")
            else:
                print(f"[Dataset]: key = {key} | Type = {type(node)} (Not a shape-bearing dataset)")

    def getvar(self, key: str):
        if debug: print(f"[Dataset]: returning h5 pointer to {key}")
        return self.file[key]

    @staticmethod
    def _convert_collapse(collapse_dims: tuple[int], collapse_idxs: tuple[int | slice]) -> tuple:
        assert(len(collapse_dims) == len(collapse_idxs)), f"[Dataset]: collapse_dims = {collapse_dims} and collapse_idxs = {collapse_idxs} must pair up"
        if len(collapse_dims) == 0:
            if debug: print("[Dataset]: no collapse requested, using full slice")
            return (slice(None),)

        dims = [slice(None)] * (max(collapse_dims) + 1)

        for d, idx in zip(collapse_dims, collapse_idxs):
            dims[d] = idx
            if debug: print(f"[Dataset]: collapsing dim {d} with {idx}")

        return tuple(dims)

    @staticmethod
    def save(file: str, data_dict: dict):
        with h5py.File(file, mode = "w") as f:
            for key in data_dict.keys():
                if isinstance(data_dict[key], (np.ndarray, torch.Tensor)):
                    data = data_dict[key].detach().cpu().numpy() if isinstance(data_dict[key], torch.Tensor) else data_dict[key]
                    f.create_dataset(key, shape=data.shape, dtype=data.dtype, data=data)
                    if debug: print(f"[Dataset]: saving {key} to file {file} | shape = {data.shape} | dtype = {data.dtype}")
                else:
                    if debug: print(f"[Dataset]: failed to save {key} to file {file}, key was not tensor or np.array: {type(data_dict[key])}")
                    
    def __len__(self):
        assert(self.advanced_available), "[Dataset]: __len__ requires advanced init!"
        if debug: print(f"[Dataset]: returning length = {self.timesteps}")
        return self.timesteps

    def __getitem__(self, index: tuple):
        assert(self.advanced_available), "[Dataset]: __getitem__ requires advanced init!"
        if debug: print(f"[Dataset]: getting index = {index} from data shape = {tuple(self.data.shape)}")
        return self.data.__getitem__(index)

    def build_dataloaders(self, data_split_ratios: tuple, batch_size: int, sequence_length: int):
        assert(self.advanced_available),"[Dataset]: build_dataloaders requires advanced init!"
        assert(sum(data_split_ratios) == 1.0), f"[Dataset]: sum of data_split_ratios must equal 1.0, equals {sum(data_split_ratios)}!"

        if debug: print(f"[Dataset]: building dataloaders | ratios = {data_split_ratios} | batch_size = {batch_size} | sequence_length = {sequence_length} | sequential_split = {sequential_split}")

        if True: # Placeholder for other split regiemes if needed.
            dataloaders = []
            curr_idx = 0

            for i, ratio in enumerate(data_split_ratios):
                training_idx = 0 # this suffles batch-to-batch so that batches are not sequential. intra-batch is always sequential

                if i == len(data_split_ratios) - 1:
                    length = (self.timesteps - curr_idx) # prevents rounding errors causing the last timesteps to be skipped
                else: 
                    length = round(ratio * self.timesteps)
                
                assert(length >= sequence_length), f"[Dataset]: split {i} holds {length} timesteps, fewer than sequence_length = {sequence_length}"

                # This builds [time, sequence_length, x, y, channels] by unfolding each section [data] = [train][val][test] for each loader.
                
                unfolded_data = self[curr_idx:curr_idx + length].unfold(0, sequence_length, 1).movedim(-1, 1)
        
                # Then this splits the time dim into batches [train] -> [batch1][batch2]...[batch-1]. Each batch is now one datapoint to maintain one batch is continuous
                batches = unfolded_data.split(batch_size, dim=0)

                if debug: print(f"[Dataset]: Dataloader: ratio = {ratio} | length = {length} | range = [{curr_idx}:{curr_idx + length}] | unfolded shape = {tuple(unfolded_data.shape)} | batches = {len(batches)} | shuffle = {i == training_idx}")

                dataloaders.append(DataLoader(
                    batches, batch_size=1, shuffle=(i == training_idx), collate_fn=lambda b: b[0],
                ))

                curr_idx += length

            return dataloaders    
        