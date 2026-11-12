
"""
ModelClass: the abstract base every model inherits, holding the shared save / loss / reshape / normalisation / sensor sampling plumbing.

Globals:
    - debug: when True, prints save / indexing traffic and turns on autograd anomaly detection

Authors: text = Claude Opus 5 | code = Gabriel Bray
"""

import os
import json
import torch
import numpy as np
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import Any, Literal
from SFM.utils.utils import get_time
from SFM.losses.losses import losses

# --- Developer Debug Flags ---
debug = False
torch.autograd.set_detect_anomaly(debug)

class ModelClass(ABC, torch.nn.Module):
    """
    Abstract base of every model (an nn.Module). A subclass implements forward, everything else is shared and losses can be overridden.
    """

    def __init__(self):
        """Sets up the cache holding the sensor layout that index reuses between calls."""
        super().__init__() 
        # all purpose cache. 
        self.cached = {"x_idx": None, "y_idx": None, "number_of_sensors": None, "sensor_indexing": None}
        
    def __repr__(self) -> str:
        """Output: nn.Module's layer listing followed by the model's serialisable hyperparameters, as a str."""
        return f"{super().__repr__()} \n{self._get_self_params()}"

    def _get_self_params(self) -> dict[str, Any]:
        """
        Collects the json-serialisable attributes of the model (str / float / int / bool, containers of them and the sensor cache) so a run can be reproduced.

        Output:
            {attribute name: value}
        """
        parameters: dict[str, Any] = {}

        def _recurse_filter(value):
            # acceptable
            if isinstance(value, (str, float, int, bool)):
                return True

            # keep looping 
            elif isinstance(value, (list, tuple)):
                for v in value:
                    if not _recurse_filter(v):
                        return False
                
                return True
            
            # keep looping 
            elif isinstance(value, dict):
                return _recurse_filter(list(value.values()))

            # kick out
            else:
                return False

        for name, value in vars(self).items():
            if name == "cached":
                parameters[name] = {k: v.tolist() if isinstance(v, torch.Tensor) else v for k, v in value.items()}

            if isinstance(value, (str, float, int, bool)):
                parameters[name] = value
                
            elif isinstance(value, (list, tuple, dict)):
                if _recurse_filter(value):
                    parameters[name] = value

        if debug: print(f"[get_self_params]: collected = {list(parameters)}")
        return parameters

    def save(self, root_path: str) -> None:
        """
        Writes the state dict and hyperparameters as a matching model_{stamp}.pth / params_{stamp}.json pair.

        Input:
            root_path: folder to write into, created if missing
        Output:
            None
        """
        merged_params: dict[str, Any] = self._get_self_params()
        stamp: str = get_time()

        os.makedirs(root_path, exist_ok=True)
        model_pth: str = os.path.join(root_path, f"model_{stamp}.pth")
        param_pth: str = os.path.join(root_path, f"params_{stamp}.json")

        with open(param_pth, "w") as f:
            json.dump(merged_params, f, indent=4)
        if debug: print(f"[save]: params = {param_pth}")

        torch.save(self.state_dict(), model_pth)
        if debug: print(f"[save]: model = {model_pth}")

    @abstractmethod
    def forward(self, batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Each model's forward pass, the only method the training loop calls, so it never needs a model's internals.

        Input:
            batch: [batch, seq_len, x, y, channels]
        Output:
            a tuple whose first two items are (output, target), both [batch, x, y, out_channels]
        """
        pass
    
    def losses(self, output: torch.Tensor, target: torch.Tensor, extralosses: dict = {}, **kwargs) -> tuple[list[torch.Tensor], list[str]]:
        """
        The default training objective, handed off to losses/losses.py. A model only defines losses when it needs something else.

        Input:
            output / target: both [batch, x, y, out_channels], as returned by forward
            extralosses: {name: scalar tensor}, appended as they are
            kwargs: use_* overrides of losses()
        Output:
            (list of scalar loss tensors, list of matching names)
        """
        return losses(output, target, extra_losses=extralosses, **kwargs)

    @staticmethod
    def standard_output(out: torch.Tensor, output_size: tuple, out_channels: int):
        """
        Reshapes a prediction into the field layout every model returns.

        Input:
            out: any shape holding x * y * out_channels values per batch element
            output_size: (x, y)
            out_channels: number of channels
        Output:
            [batch, x, y, out_channels]
        """
        expected: int = output_size[0] * output_size[1] * out_channels
        assert(out[0].numel() == expected), f"[standard_output]: expected {expected} values per batch element got {out[0].numel()}"
        return out.reshape(-1, output_size[0], output_size[1], out_channels)

    @staticmethod
    def standard_input(inpt: torch.Tensor, output_size: tuple, sequence_length: int, out_channels: int):
        """
        Asserts a raw batch is exactly the window layout every model consumes.

        Input:
            inpt: [batch, sequence_length, x, y, out_channels]
            output_size: (x, y)
            sequence_length / out_channels: the sizes inpt must have
        Output:
            inpt, unchanged
        """
        expected: tuple = (sequence_length, output_size[0], output_size[1], out_channels)
        assert(inpt.ndim == 5), "[standard_input]: standard_input shaped [batch, sequence_length, x, y, channels]"
        assert(tuple(inpt.shape[1:]) == expected), f"[standard_input]: expected [batch, {', '.join(map(str, expected))}] got {list(inpt.shape)}"
        return inpt

    def update_running_stats(self, input: torch.Tensor, in_channels: int, out_channels: int, momentum: float):
        """
        EMA of the per channel sensor mean / mean square. The buffers are created on the first call (momentum is 1.0 then).
        The output stats copy the first min(in_channels, out_channels) input channels, the rest stay identity (mean 0, std 1).

        Input:
            input: [batch, seq, num_sensors, in_channels]
            in_channels / out_channels: number of sensor / target channels
            momentum: weight of the new batch
        Output:
            None, the input_running_* / output_running_* buffers are updated in place
        """
        assert(input.ndim == 4) # [batch, sequence, num_sensors, channels]
        assert(input.shape[-1] == in_channels)

        # create buffers if not exist already.
        if not {"input_running_mean", "input_running_square_mean", "output_running_mean", "output_running_square_mean", "seen_batches"} <= self._buffers.keys():
            self.register_buffer("input_running_mean", torch.zeros(in_channels, device=self.params_device))
            self.register_buffer("input_running_square_mean", torch.ones(in_channels, device=self.params_device))
            self.register_buffer("output_running_mean", torch.zeros(out_channels, device=self.params_device))
            self.register_buffer("output_running_square_mean", torch.ones(out_channels, device=self.params_device))
            self.register_buffer("seen_batches", torch.zeros((), dtype=torch.long, device=self.params_device))
            momentum = 1.0 # this forces the starting avarage value to be of the first sample, to prevent blow up.

        self.input_running_mean.lerp_(torch.mean(input, dim=(0, 1, 2)), momentum)
        self.input_running_square_mean.lerp_(torch.mean(input**2, dim=(0, 1, 2)), momentum)

        # output channels without a sensor keep mean 0 / std 1 (identity)
        k = min(in_channels, out_channels)
        self.output_running_mean[:k].copy_(self.input_running_mean[:k])
        self.output_running_square_mean[:k].copy_(self.input_running_square_mean[:k])

    @staticmethod
    def std(running_mean, running_square_mean, eps=1e-5) -> torch.Tensor:
        """
        Input:
            running_mean / running_square_mean: [channels]
            eps: floor of the variance
        Output:
            sqrt(mean square - mean^2) with the variance clamped at eps, [channels]
        """
        return (running_square_mean - running_mean ** 2).clamp_min(eps).sqrt()

    @property
    def input_mean(self) -> torch.Tensor:
        """Output: the running per channel mean of the sensors, [in_channels]."""
        return self.input_running_mean

    @property
    def input_std(self) -> torch.Tensor:
        """Output: the running per channel std of the sensors, [in_channels]."""
        return self.std(self.input_running_mean, self.input_running_square_mean)

    @property
    def output_mean(self) -> torch.Tensor:
        """Output: the running per channel mean of the target, [out_channels]."""
        return self.output_running_mean

    @property
    def output_std(self) -> torch.Tensor:
        """Output: the running per channel std of the target, [out_channels]."""
        return self.std(self.output_running_mean, self.output_running_square_mean)

    @staticmethod
    def normalise(x: torch.Tensor, mean: float | torch.Tensor, std: float | torch.Tensor, dim: int = -1) -> torch.Tensor:
        """
        (x - mean) / std along the last dim.

        Input:
            x: [..., channels]
            mean / std: [channels] tensors, or two floats
            dim: unused
        Output:
            x's shape. With float mean / std it returns (normalised, x.mean(), x.std())
        """
        if isinstance(mean, float) and isinstance(std, float): 
            return (x - mean) / std, x.mean(), x.std()
        else:
            assert(x.shape[-1] == mean.shape[0] == std.shape[0])
            assert(1 == mean.ndim == std.ndim)
            normalized = (x - mean.view(*tuple([1]*(x.ndim-1)), mean.shape[0])) / std.view(*tuple([1]*(x.ndim-1)), mean.shape[0])
            return normalized

    @staticmethod
    def denormalise(x: torch.Tensor, mean: float | torch.Tensor, std: float | torch.Tensor, dim: int = -1) -> torch.Tensor:
        """
        x * std + mean along the last dim, the inverse of normalise.

        Input:
            x: [..., channels]
            mean / std: [channels] tensors, or two floats
            dim: unused
        Output:
            x's shape
        """
        if isinstance(mean, float) and isinstance(std, float): 
            return x * std + mean
        else:
            assert(x.shape[-1] == mean.shape[0] == std.shape[0])
            assert(1 == mean.ndim == std.ndim)
            normalized = (x * std.view(*tuple([1]*(x.ndim-1)), mean.shape[0])) + mean.view(*tuple([1]*(x.ndim-1)), mean.shape[0])
            return normalized

    def set_sensor_points(self, points: torch.Tensor | np.ndarray | list, grid: tuple[int, int] | None = None) -> None:
        """
        Custom sensor layout. Checks the points and caches them as the "manual" layout index then uses on every call.

        Input:
            points: [num_sensors, 2] as (x, y) grid indices (tensor, array or list)
            grid: (x, y) to bounds check against, defaults to self.output_size
        Output:
            None
        """
        # custom sensor layout, cached as "manual" so index reuses it on every call
        points = torch.as_tensor(points)
        grid   = grid if grid is not None else getattr(self, "output_size", None)

        assert(points.ndim == 2 and points.shape[1] == 2), f"[set_sensor_points]: points must be [num_sensors, 2] as (x, y) grid indices got {tuple(points.shape)}"
        assert(not points.is_floating_point() or (points == points.round()).all()), "[set_sensor_points]: points are grid indices, they must be whole numbers"
        points = points.long().cpu()

        assert(len(torch.unique(points, dim=0)) == len(points)), "[set_sensor_points]: the same point is listed twice"
        assert(points.min() >= 0), "[set_sensor_points]: points must be >= 0"
        if grid is not None:
            assert(points[:, 0].max() < grid[0] and points[:, 1].max() < grid[1]), f"[set_sensor_points]: points must sit inside the (x, y) grid {tuple(grid)} got max {points.max(dim=0).values.tolist()}"
        if hasattr(self, "num_sensors"):
            assert(points.shape[0] == self.num_sensors), f"[set_sensor_points]: got {points.shape[0]} points, model was built with num_sensors = {self.num_sensors}"

        self.cached["x_idx"] = points[:, 0]
        self.cached["y_idx"] = points[:, 1]
        self.cached["number_of_sensors"] = points.shape[0]
        self.cached["sensor_indexing"] = "manual"
        if hasattr(self, "indexing_method"): self.indexing_method = "manual"

        if debug: print(f"[set_sensor_points]: {points.shape[0]} manual points = {points.tolist()}")

    def index(self,
              batch: torch.Tensor, 
              number_of_sensors: int, 
              sensor_indexing: Literal["midline", "top", "bot", "rand", "rand_const", "manual"], 
              return_locations: bool = True, 
              channel_slice: slice = slice(None), 
              manual_pts: torch.Tensor = None,
              seed: int | None = None):
        """
        Samples sensor readings from a batch. "rand" draws new points for every batch element on every call, every other layout is cached and reused.

        Input:
            batch: [batch, seq_len, x, y, channels]
            number_of_sensors: how many sensors to sample
            sensor_indexing: "midline" / "top" / "bot" (evenly spaced along x at y = y // 2 / 1 / y - 1), "rand", "rand_const" (random, then fixed) or "manual"
            return_locations: adds "sensor_locs" to the output
            channel_slice: the channels the sensors read
            manual_pts: [num_sensors, 2] points for "manual", not needed after set_sensor_points
            seed: seeds torch and the "rand_const" draw
        Output:
            {"sensors_vals": [batch, seq_len, num_sensors, ch], "target": [batch, x, y, channels] (the last timestep), "sensor_locs": [batch, num_sensors, 2]}
        """
        if seed is not None:
            torch.manual_seed(seed)
            rng = np.random.default_rng(seed=seed)
        else:
            rng = np.random


        instep = 1
        return_dict = {}
        options = ["midline", "top", "bot", "rand", "rand_const", "manual"]
        assert(sensor_indexing in options), f"[index_input]: selected sensor mode must be in {options}"

        if debug: print(f"[index_input]: instep = {instep}")

        shape = batch.shape
        b, x, y = shape[0], shape[2], shape[3]

        if debug: print(f"[index_input]: input batch = {shape}")
        if debug: print(f"[index_input]: sensor mode = {sensor_indexing}")

        if sensor_indexing == "rand":
            flat_idx = torch.argsort(torch.randn((b, x * y)), dim = 1)[:, :number_of_sensors] # distinct grid points, works when x or y < number_of_sensors
            x_idx, y_idx = flat_idx // y, flat_idx % y
            b_idx = torch.arange(b)[:, None].expand(-1, number_of_sensors)

            batch_perm = batch.permute(0, 2, 3, 1, 4) # permutes to [batch, x, y, seqlen, channels]
            sensors = batch_perm[b_idx, x_idx, y_idx][..., channel_slice].transpose(1, 2) # advanced indexing replaced dim0-2 with [batch, num_sensors] then becomes fully [batch, num_sensors, seq_len, channels] - permute -> [batch, seq_len, num_sensors, channels]
            locs = torch.stack([x_idx, y_idx], dim=-1)
        
        else:
            if None not in [self.cached["x_idx"], self.cached["y_idx"], self.cached["number_of_sensors"], self.cached["sensor_indexing"]] and number_of_sensors == self.cached["number_of_sensors"] and sensor_indexing == self.cached["sensor_indexing"]:
                x_idx = self.cached["x_idx"]
                y_idx = self.cached["y_idx"]
            else:
                match sensor_indexing:
                    case "midline":
                        x_idx = torch.linspace(0 + instep, x - instep, number_of_sensors).long()
                        y_idx = torch.tensor([y//2]*number_of_sensors)
                    case "top":
                        x_idx = torch.linspace(0 + instep, x - instep, number_of_sensors).long()
                        y_idx = torch.tensor([0 + instep]*number_of_sensors)
                    case "bot":
                        x_idx = torch.linspace(0 + instep, x - instep, number_of_sensors).long()
                        y_idx = torch.tensor([y - instep]*number_of_sensors)
                    case "rand_const":
                        flat_idx = torch.from_numpy(rng.choice(x * y, number_of_sensors, replace=False))
                        x_idx, y_idx = flat_idx // y, flat_idx % y
                    case "manual":
                        assert(manual_pts is not None), "[index_input]: manual needs points, call set_sensor_points(points) first or pass manual_pts"
                        self.set_sensor_points(manual_pts, grid=(x, y))
                        x_idx = self.cached["x_idx"]
                        y_idx = self.cached["y_idx"]
                        assert(x_idx.shape[0] == number_of_sensors), f"[index_input]: got {x_idx.shape[0]} manual points for number_of_sensors = {number_of_sensors}"

                self.cached["x_idx"] = x_idx
                self.cached["y_idx"] = y_idx
                self.cached["number_of_sensors"] = number_of_sensors
                self.cached["sensor_indexing"] = sensor_indexing

            sensors = batch[:, :, x_idx, y_idx, channel_slice]
            locs = torch.stack([x_idx, y_idx], dim=-1).unsqueeze(0).expand(b, number_of_sensors, 2)

        return_dict["sensors_vals"] = sensors
        return_dict["target"] = batch[:, -1]

        if debug:
            print(f"[index_input]: sensors = {sensors.shape}")
            print(f"[index_input]: target = {return_dict["target"].shape}")

        if return_locations: 
            return_dict["sensor_locs"] = locs.to(batch.device)

            if debug:
                print(f"[index_input]: sensor locations = {locs.shape}")
                print(f"[index_input]: locs = {locs}")

        return return_dict
        
    @property
    def params_device(self):
        """Output: the torch.device the model's parameters live on."""
        return next(self.parameters()).device

if __name__ == "__main__":
    pass