"""
VoronoiCNN: the Voronoi tessellation-assisted CNN of Fukami et al. (2021), lifting sparse sensors onto the full grid and reconstructing with a flat same-padded conv stack.

Globals:
    - debug: when True, prints the running stats and turns on autograd anomaly detection

Authors: text = Claude Opus 5 | code = Gabriel Bray
"""

import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from SFM.utils.utils import get_time
from SFM.utils.plots import PLOT_STYLE
from SFM.models.abstract import ModelClass

# --- Developer Debug Flags ---
debug = False
torch.autograd.set_detect_anomaly(debug)

@dataclass(eq=False, repr=False) #! eq=False keeps nn.Module's hash, repr=False keeps ModelClass's repr
class VoronoiCNN(ModelClass):
    """
    Voronoi tessellation-assisted CNN (Fukami et al. 2021).
    A dataclass whose fields are the arguments stored on self, the shared api is inherited from ModelClass (abstract.py).

    Fields:
        num_sensors: number of sensors
        batch_size: batch size, stored only
        sequence_length: timesteps in the sensor window
        output_size: (x, y) of one snapshot
        indexing_method: sensor layout handed to index (abstract.py)
        use_sensor_locations: stored only
        in_channels: channels the sensors read
        out_channels: channels of the field the model outputs
        dropout: dropout of the mid blocks
        mid_blocks: conv / relu / dropout blocks between the first and the last conv
        mid_channel_size: channels of the mid blocks
        kernel_size: kernel of every conv
        normalize_outputs: normalises the sensors and the target with the running stats
        momentum: momentum of the running stats
    """
    num_sensors:          int
    batch_size:           int
    sequence_length:      int
    output_size:          tuple[int, int]
    indexing_method:      str
    use_sensor_locations: bool = True
    in_channels:          int = 2
    out_channels:         int = 2
    dropout:              float = 0.0
    mid_blocks:           int = 7      # https://github.com/kfukami/Voronoi-CNN/blob/main/Voronoi-CNN-cy.py uses 7 layers
    mid_channel_size:     int = 48     # uses 48
    kernel_size:          int = 7
    normalize_outputs:    bool = True
    momentum:             float = 0.05

    def __post_init__(self) -> None:
        """Builds the conv2d / relu (/ dropout) stack taking sequence_length * in_channels tessellated maps plus the sensor mask."""
        super().__init__()

        # one tessellated map per timestep and channel plus the sensor mask
        self.cnn_in_channels      = self.sequence_length * self.in_channels + 1 # pointwise mask to show model exact sensor location.
        self.activation           = nn.ReLU

        self.cnn = nn.ModuleList()
        self.cnn.append(nn.Conv2d(self.cnn_in_channels, self.mid_channel_size, kernel_size=self.kernel_size, padding='same'))
        self.cnn.append(self.activation())

        for _ in range(self.mid_blocks):
            self.cnn.append(nn.Conv2d(self.mid_channel_size, self.mid_channel_size, kernel_size=self.kernel_size, padding='same'))
            self.cnn.append(self.activation())
            self.cnn.append(nn.Dropout(self.dropout))

        self.cnn.append(nn.Conv2d(self.mid_channel_size, self.out_channels, kernel_size=self.kernel_size, padding='same'))
        self.cnn = nn.Sequential(*self.cnn)

        self.cache = {"tessellated_grid_idx": None, "indices_grid_idx": None, "locations" : None}

    def forward(self, batch): 
        """
        Samples the sensors and, when normalize_outputs, normalises them with the running stats (updated in training only).

        Input:
            batch: [batch, sequence_length, x, y, out_channels]
        Output:
            (output, target): both [batch, x, y, out_channels], normalised in training, physical in eval
        """
        batch        = self.standard_input(batch, self.output_size, self.sequence_length, self.out_channels)
        return_dict  = self.index(batch, self.num_sensors, self.indexing_method, return_locations=True, channel_slice=slice(0, self.in_channels, None))
        sensors_vals = return_dict["sensors_vals"]   # [batch, seq, numsens, channels]
        sensor_locs  = return_dict["sensor_locs"]    # [batch, numsens, 2]
        target       = return_dict["target"]         # [batch, x, y, channels]

        if not self.normalize_outputs:
            out = self.predict(sensors_vals, sensor_locs)
            return out, target

        # stats only move in training, but the sensors are normalised in both modes so eval sees the same input scale
        if self.training:
            self.update_running_stats(sensors_vals, self.in_channels, self.out_channels, self.momentum)

            if debug:
                print(f"[{type(self).__name__}]: momentum = {self.momentum}")
                print(f"[VoronoiCNN]: input mean = {self.input_mean}, output mean = {self.output_mean}")
                print(f"[VoronoiCNN]: input std = {self.input_std}, output std = {self.output_std}")

        sensors_vals = self.normalise(sensors_vals, self.input_mean, self.input_std, dim=-1)
        out = self.predict(sensors_vals, sensor_locs)

        if self.training: 
            return out, self.normalise(target, self.output_mean, self.output_std, dim=-1) # loss in normalised space
        else:             
            return self.denormalise(out, self.output_mean, self.output_std, dim=-1), target # physical units

    @staticmethod
    def _voronoi_tessellate_2d(locations: torch.Tensor, output_size: tuple[int, int]) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Takes in batched sensor locations and returns the nearest-sensor index at every grid point plus a 1.0 mask on the sensor cells.

        Assuming:
            - same number of sensors in every sequence in every batch; not necessarily same location
            - one sequence has same number of sensors in the same configurations ie. sensors do not move within sequence

        Input:
            locations: [batch, number_of_sensors, 2] as (x, y) grid indices
            output_size: fixed tuple of ints like (64, 64)
        Output:
            (tessellation, mask): both [batch, x, y], the nearest sensor of every grid point (0 to num_sensors - 1) and 1.0 where there is a sensor, 0.0 elsewhere
        """

        batch = locations.shape[0]
        x_shape, y_shape = output_size
        x = torch.arange(0, x_shape, device=locations.device)
        y = torch.arange(0, y_shape, device=locations.device)

        grid_x, grid_y = torch.meshgrid(x, y, indexing="ij")

        dx = (grid_x[None, None] - locations[:, :, 0, None, None])**2
        dy = (grid_y[None, None] - locations[:, :, 1, None, None])**2

        out = (dx + dy).argmin(dim=1) # [batch, x, y] nearest sensor per grid point

        b_idx = torch.arange(batch, device=locations.device)[:, None]
        idx   = torch.zeros((batch, x_shape, y_shape), device=locations.device)
        idx[b_idx, locations[:, :, 0], locations[:, :, 1]] = 1.0

        assert(out.shape == idx.shape)

        return out, idx

    @staticmethod
    def _build_input(sensors: torch.Tensor, 
                     tessellated_grid_idx: torch.Tensor,
                     indices_grid_idx: torch.Tensor,
                    ):
        """
        Uses the tessellated grid indices to gather every grid point's nearest sensor window and appends the mask as the last channel.

        Input:
            sensors: [batch, sequence_length, num_sensors, channels]
            tessellated_grid_idx: [batch, x, y] each element has an integer number containing the sensor that belongs there 0-num_sensors-1
            indices_grid_idx: [batch, x, y] 1.0 where there is a sensor 0.0 elsewhere
        Output:
            [batch, sequence_length * channels + 1, x, y]
        """

        assert(tessellated_grid_idx.shape == indices_grid_idx.shape)

        b, seq, _, ch = sensors.shape
        _, x, y       = tessellated_grid_idx.shape

        flat  = tessellated_grid_idx.flatten(1).long()                  # [batch, x*y]
        b_idx = torch.arange(b, device=sensors.device)[:, None]         # [batch, 1] broadcasts against flat

        inpt = sensors.transpose(1, 2)[b_idx, flat]                     # [batch, x*y, sequence_length, channels] every grid point's nearest sensor window
        inpt = inpt.reshape(b, x, y, seq * ch).permute(0, 3, 1, 2)      # [batch, sequence_length * channels, x, y]

        return torch.cat([inpt, indices_grid_idx.unsqueeze(1)], dim=1)      # mask appended as the last channel

    def predict(self, sensor_vals, sensor_locs):
        """
        Tessellates the sensors onto the grid and runs the conv stack in bf16.

        Input:
            sensor_vals: [batch, seq, num_sensors, in_channels]
            sensor_locs: [batch, num_sensors, 2] as (x, y) grid indices
        Output:
            [batch, x, y, out_channels]
        """
        tessellated_grid_idx, indices_grid_idx = VoronoiCNN._voronoi_tessellate_2d(locations=sensor_locs, output_size=self.output_size)
        inpt = VoronoiCNN._build_input(sensor_vals, tessellated_grid_idx, indices_grid_idx)

        with torch.autocast(sensor_vals.device.type, dtype=torch.bfloat16): 
            out = self.cnn(inpt).permute(0, 2, 3, 1) 
            # [batch, out_channels, x, y] -> [batch, x, y, out_channels]
        
        out = out.float() # back to fp32 so the losses compare against the fp32 target

        out = self.standard_output(out, self.output_size, self.out_channels)

        return out
