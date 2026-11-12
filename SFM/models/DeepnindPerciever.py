"""
PercieverIO: DeepMind's Perceiver IO (huggingface) for field reconstruction, cross attending sensor tokens into a latent array and decoding with one fourier query per grid point.

Globals:
    - debug: when True, prints the running stats and turns on autograd anomaly detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PerceiverConfig, PerceiverModel
from SFM.models.abstract import ModelClass
from dataclasses import dataclass
from SFM.models.fourierEncoder import fourierEncoder
from transformers.models.perceiver.modeling_perceiver import PerceiverBasicDecoder

# --- Developer Debug Flags ---
debug = False
torch.autograd.set_detect_anomaly(debug)

@dataclass(eq=False, repr=False) #! eq=False keeps nn.Module's hash, repr=False keeps ModelClass's repr
class PercieverIO(ModelClass):
    """
    DeepMind's Perceiver IO (huggingface) for field reconstruction.
    A dataclass whose fields are the arguments stored on self, the shared api is inherited from ModelClass (abstract.py).

    Fields:
        output_size: (x, y) of one snapshot
        seq_len: timesteps in the sensor window
        num_sensors: number of sensors
        in_channels: channels the sensors read
        out_channels: channels of the field the model outputs
        indexing_method: sensor layout handed to index (abstract.py)
        device: the device the model runs on
        dropout: stored only, not handed to the Perceiver config
        d_model: width of the tokens and of the latents
        num_latents: size of the latent array
        num_blocks: self attention blocks
        num_self_attends_per_block: self attention layers per block
        num_cross_attention_heads / num_self_attention_heads: attention heads
        normalize_outputs: normalises the sensors and the target with the running stats
        momentum: momentum of the running stats
    """
    output_size:               tuple[int, int]
    seq_len:                   int
    num_sensors:               int
    in_channels:               int
    out_channels:              int
    indexing_method:           str
    device:                    torch.device | str
    dropout:                   float = 0.1
    d_model:                   int = 128
    num_latents:               int = 128
    num_blocks:                int = 8
    num_self_attends_per_block:int = 2
    num_cross_attention_heads: int = 4
    num_self_attention_heads:  int = 4
    normalize_outputs:         bool = True
    momentum:                  float = 0.05

    def __post_init__(self):
        """Builds the sensor tokenizer, the fourier location embedding, the PerceiverModel and the PerceiverBasicDecoder."""
        super().__init__()

        config = PerceiverConfig(
            d_model=self.d_model,                                        # Latent feature dimension
            d_latents=self.d_model,
            num_latents=self.num_latents,                                # Size of the latent bottleneck
            num_blocks=self.num_blocks,                                  # Number of self-attention blocks (outer loop)
            num_self_attends_per_block=self.num_self_attends_per_block,  # Self-attention steps per block
            num_cross_attention_heads=self.num_cross_attention_heads,    # Heads in input cross-attention
            num_self_attention_heads=self.num_self_attention_heads,      # Heads in latent self-attention
        )

        self.tokenizer  = nn.Linear(self.in_channels*self.seq_len, self.d_model)
        self.embed      = fourierEncoder(2, self.d_model, norm_to_max=max(self.output_size))
        self.model      = PerceiverModel(config)
        self.decoder    = PerceiverBasicDecoder(
            config,
            output_num_channels=self.out_channels,
            position_encoding_type="none",
            num_channels=self.d_model,
            num_heads=self.num_cross_attention_heads,
            use_query_residual=True,
        )

    def forward(self, batch): 
        """
        Samples the sensors and, when normalize_outputs, normalises them with the running stats (updated in training only).

        Input:
            batch: [batch, seq_len, x, y, out_channels]
        Output:
            (output, target): both [batch, x, y, out_channels], normalised in training, physical in eval
        """
        batch = self.standard_input(batch, self.output_size, self.seq_len, self.out_channels)
        return_dict  = self.index(batch, self.num_sensors, self.indexing_method, True, channel_slice=slice(0, self.in_channels, None))
        sensors_vals = return_dict["sensors_vals"]  # [batch, seq, numsens, channels]
        sensors_locs = return_dict["sensor_locs"]   # [batch, numsens, 2]
        target       = return_dict["target"]        # [batch, x, y, channels]

        if not self.normalize_outputs:
            out = self.predict(sensors_vals, sensors_locs)
            return out, target

        # stats only move in training, but the sensors are normalised in both modes so eval sees the same input scale
        if self.training:
            self.update_running_stats(sensors_vals, self.in_channels, self.out_channels, self.momentum)

            if debug:
                print(f"[{type(self).__name__}]: momentum = {self.momentum}")
                print(f"[PercieverIO]: input mean = {self.input_mean}, output mean = {self.output_mean}")
                print(f"[PercieverIO]: input std = {self.input_std}, output std = {self.output_std}")

        sensors = self.normalise(sensors_vals, self.input_mean, self.input_std, dim=-1)
        out     = self.predict(sensors, sensors_locs)

        if self.training: 
            return out, self.normalise(target, self.output_mean, self.output_std, dim=-1) # loss in normalised space
        else:             
            return self.denormalise(out, self.output_mean, self.output_std, dim=-1), target # physical units
    
    def predict(self, sensors_vals, sensors_locs):
        """
        tokens = embed(location) + tokenizer(sensor window), encoded to latents and decoded at every grid point query.

        Input:
            sensors_vals: [batch, seq, num_sensors, in_channels]
            sensors_locs: [batch, num_sensors, 2] as (x, y) grid indices
        Output:
            [batch, x, y, out_channels]
        """
        B = sensors_vals.shape[0]

        intokens = self.embed(sensors_locs) + self.tokenizer(sensors_vals.permute(0, 2, 1, 3).flatten(-2))
        intokens = intokens.flatten(1, -2)
        queries  = self.queries.flatten(1, -2)

        latents = self.model(inputs=intokens).last_hidden_state             # [B, num_latents, d_latents]
        queries = self.queries.flatten(0, 1).unsqueeze(0).expand(B, -1, -1)
        out     = self.decoder(queries, z=latents).logits                   # [B, X*Y, out_channels]
        
        return out.view(B, *self.output_size, self.out_channels)

    @property
    def output_grid_locs(self):
        """Output: every (x, y) grid index, [x, y, 2]."""
        x = torch.arange(0, self.output_size[0])
        y = torch.arange(0, self.output_size[1])
        grid_x, grid_y = torch.meshgrid(x, y, indexing="ij")
        grid = torch.stack((grid_x, grid_y), dim=-1)
        return grid.to(self.device)

    @property
    def queries(self):
        """Output: the fourier embedding of every grid point, [x, y, d_model]."""
        return self.embed(self.output_grid_locs)

