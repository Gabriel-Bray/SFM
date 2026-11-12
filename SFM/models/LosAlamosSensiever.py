"""
Senseiver: the Los Alamos Senseiver (Santos et al. 2023), a perceiver style encoder over sparse sensor tokens decoded by cross attention at every grid point.
PositionalEncoder: the fixed sin / cos table of grid positions it reads the sensor and query positions from.

Globals:
    - debug: when True, prints the running stats and turns on autograd anomaly detection
"""

import math
import torch
import torch.nn as nn
from einops import rearrange, repeat
import torch.nn.functional as F
from SFM.models.abstract import ModelClass
from dataclasses import dataclass
from SFM.models.modules import CrossAttentionLayer, SelfAttentionLayer

# --- Developer Debug Flags ---
debug = False
torch.autograd.set_detect_anomaly(debug)

@dataclass(eq=False, repr=False) #! eq=False keeps nn.Module's hash, repr=False keeps ModelClass's repr
class Senseiver(ModelClass):
    """
    The Los Alamos Senseiver (Santos et al. 2023).
    A dataclass whose fields are the arguments stored on self, the shared api is inherited from ModelClass (abstract.py).

    Fields:
        output_size: (x, y) of one snapshot
        seq_len: timesteps in the sensor window
        num_sensors: number of sensors
        in_channels: channels the sensors read
        out_channels: channels of the field the model outputs
        indexing_method: sensor layout handed to index (abstract.py)
        device: the device the model runs on
        num_freq_bands: frequency bands per spatial dim of the positional encoding
        latent_channels: width of the latents
        enc_preproc_channels: width the sensor tokens are projected to
        enc_num_latents: size of the latent array
        enc_num_blocks: encoder blocks, each one cross attention + its self attention layers
        enc_num_cross_attention_heads / enc_num_self_attention_heads: attention heads of the encoder
        enc_num_self_attention_layers_per_block: self attention layers per block
        dec_query_channels: width of the learned query
        dec_num_cross_attention_heads: attention heads of the decoder
        dec_preproc_channels: optional width the queries are projected to
        dec_num_query_tokens: learned query tokens, repeated for every grid point
        dropout: dropout of the attention layers
        normalize_outputs: normalises the sensors and the target with the running stats
        momentum: momentum of the running stats
    """
    output_size:                                tuple[int, ...]
    seq_len:                                    int
    num_sensors:                                int
    in_channels:                                int
    out_channels:                               int
    indexing_method:                            str
    device:                                     torch.device | str
    num_freq_bands:                             int = 8
    latent_channels:                            int = 128
    enc_preproc_channels:                       int = 128
    enc_num_latents:                            int = 128
    enc_num_blocks:                             int = 4
    enc_num_cross_attention_heads:              int = 4
    enc_num_self_attention_heads:               int = 4
    enc_num_self_attention_layers_per_block:    int = 2
    dec_query_channels:                         int = 128
    dec_num_cross_attention_heads:              int = 4
    dec_preproc_channels:                       int | None = None
    dec_num_query_tokens:                       int = 1
    dropout:                                    float = 0.1
    normalize_outputs:                          bool = True
    momentum:                                   float = 0.05

    def __post_init__(self):
        """Builds the fixed positional table, the encoder (first block + one weight shared block reused enc_num_blocks - 1 times) and the query decoder."""
        super().__init__()

        def create_block():
            return nn.ModuleList([
                CrossAttentionLayer(
                    num_q_channels=self.latent_channels,
                    num_kv_channels=self.enc_preproc_channels,
                    num_heads=self.enc_num_cross_attention_heads,
                    dropout=self.dropout,
                ),
                nn.Sequential(*[
                    SelfAttentionLayer(
                        num_q_channels=self.latent_channels,
                        num_kv_channels=self.latent_channels,
                        num_heads=self.enc_num_self_attention_heads,
                        dropout=self.dropout,
                    )
                    for _ in range(self.enc_num_self_attention_layers_per_block)
                ]),
            ])

        # -- Encoder ---------------------------------------------------------
        pos_channels = self.num_freq_bands*len(self.output_size)*2

        # fixed encodings for every grid point, row = flattened (x, y) index
        self.register_buffer("pos_table", PositionalEncoder((*self.output_size, 1), self.num_freq_bands), persistent=False)

        self.enc_preproc        = nn.Linear(self.in_channels*self.seq_len + pos_channels, self.enc_preproc_channels)
        self.enc_first_block    = create_block()

        # enc_shared_block weights are shared across the remaining enc_num_blocks - 1 passes
        if self.enc_num_blocks > 1: self.enc_shared_block = create_block()

        self.enc_latents = nn.Parameter(torch.empty(self.enc_num_latents, self.latent_channels))

        # -- Decoder ---------------------------------------------------------
        dec_q_channels = pos_channels + self.dec_query_channels

        if self.dec_preproc_channels: dec_attn_channels = self.dec_preproc_channels
        else:                         dec_attn_channels = dec_q_channels

        if self.dec_preproc_channels: self.dec_preproc = nn.Linear(dec_q_channels, self.dec_preproc_channels)
        else:                         self.dec_preproc = None

        self.dec_cross_attention = CrossAttentionLayer(
                                num_q_channels=dec_attn_channels,
                                num_kv_channels=self.latent_channels,
                                num_heads=self.dec_num_cross_attention_heads,
                                dropout=self.dropout,
                            )

        self.dec_postproc = nn.Linear(dec_attn_channels, self.out_channels)
        self.dec_query = nn.Parameter(torch.empty(self.dec_num_query_tokens, self.dec_query_channels))

        # -- Params Init ---------------------------------------------------------
        with torch.no_grad():
            self.enc_latents.normal_(0.0, 0.02).clamp_(-2.0, 2.0)
            self.dec_query.normal_(0.0, 0.02).clamp_(-2.0, 2.0)

    @staticmethod
    def _apply_block(block, latents, tokens, pad_mask):
        """
        One encoder block, the latents cross attend into the tokens then self attend.

        Input:
            block: (cross attention, self attention)
            latents: [batch, enc_num_latents, latent_channels]
            tokens: [batch, num_sensors, enc_preproc_channels]
            pad_mask: optional padding mask over the tokens
        Output:
            latents, same shape
        """
        cross_attention, self_attention = block
        return self_attention(cross_attention(latents, tokens, pad_mask))

    def encode(self, sensor_tokens, pad_mask=None):
        """
        Cross attends the latents into the sensor tokens then self attends, once per block.

        Input:
            sensor_tokens: [batch, num_sensors, seq * in_channels + pos_channels]
            pad_mask: optional padding mask over the tokens
        Output:
            latents: [batch, enc_num_latents, latent_channels]
        """
        b       = sensor_tokens.shape[0]
        tokens  = self.enc_preproc(sensor_tokens)
        latents = self.enc_latents.expand(b, -1, -1)
        latents = self._apply_block(self.enc_first_block, latents, tokens, pad_mask)
        
        for _ in range(self.enc_num_blocks - 1):
            latents = self._apply_block(self.enc_shared_block, latents, tokens, pad_mask)

        return latents

    def decode(self, latents, query_pos):
        """
        queries = positional encoding + learned query, cross attended into the latents and projected to out_channels.

        Input:
            latents: [batch, enc_num_latents, latent_channels]
            query_pos: [batch, x * y, pos_channels]
        Output:
            [batch, x * y, out_channels]
        """
        b       = latents.shape[0]
        queries = self.dec_query.expand(b, -1, -1)
        queries = torch.repeat_interleave(queries, query_pos.shape[1], dim=1)
        queries = torch.cat([query_pos, queries], dim=-1)

        if self.dec_preproc is not None:
            queries = self.dec_preproc(queries)

        out = self.dec_cross_attention(queries, latents)
        
        return self.dec_postproc(out)

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
                print(f"[Senseiver]: input mean = {self.input_mean}, output mean = {self.output_mean}")
                print(f"[Senseiver]: input std = {self.input_std}, output std = {self.output_std}")

        sensors = self.normalise(sensors_vals, self.input_mean, self.input_std, dim=-1)
        out     = self.predict(sensors, sensors_locs)

        if self.training: 
            return out, self.normalise(target, self.output_mean, self.output_std, dim=-1) # loss in normalised space
        else:             
            return self.denormalise(out, self.output_mean, self.output_std, dim=-1), target # physical units
    
    def predict(self, sensor_vals, sensor_locs):
        """
        Builds sensor tokens from the window + positional table, encodes and decodes every grid point.

        Input:
            sensor_vals: [batch, seq, num_sensors, in_channels]
            sensor_locs: [batch, num_sensors, 2] as (x, y) grid indices
        Output:
            [batch, x, y, out_channels]
        """
        # sensor_vals [batch, seq, num_sensors, in_channels], sensor_locs [batch, num_sensors, 2]
        
        B         = sensor_vals.shape[0]
        vals      = sensor_vals.permute(0, 2, 1, 3).flatten(-2) # [batch, num_sensors, seq * in_channels]
        
        sensor_idx = sensor_locs[..., 0].long() * self.output_size[1] + sensor_locs[..., 1].long() # [batch, num_sensors]
        tokens    = torch.cat([vals, self.pos_table[sensor_idx]], dim=-1)
        query_pos = self.pos_table.unsqueeze(0).expand(B, -1, -1)
        
        latents   = self.encode(tokens)
        out       = self.decode(latents, query_pos) # [batch, X*Y, out_channels]
                                      
        return out.view(B, *self.output_size, self.out_channels)
  
    @property
    def output_grid_locs(self):
        """Output: every (x, y) grid index, [x, y, 2]."""
        x = torch.arange(0, self.output_size[0])
        y = torch.arange(0, self.output_size[1])
        grid_x, grid_y = torch.meshgrid(x, y, indexing="ij")
        grid = torch.stack((grid_x, grid_y), dim=-1)
        return grid.to(self.device)

def PositionalEncoder(image_shape,num_frequency_bands,max_frequencies=None):
    """
    Sin / cos fourier features of the [-1, 1] grid coordinates.

    Input:
        image_shape: (*spatial, channels)
        num_frequency_bands: frequency bands per spatial dim
        max_frequencies: highest frequency per spatial dim, defaults to the spatial shape
    Output:
        [prod(spatial), 2 * len(spatial) * num_frequency_bands]
    """
    
    *spatial_shape, _ = image_shape
   
    coords = [ torch.linspace(-1, 1, steps=s) for s in spatial_shape ]
    pos = torch.stack(torch.meshgrid(*coords, indexing="ij"), dim=len(spatial_shape)) 
    
    encodings = []
    if max_frequencies is None:
        max_frequencies = pos.shape[:-1]

    frequencies = [ torch.linspace(1.0, max_freq / 2.0, num_frequency_bands)
                                              for max_freq in max_frequencies ]
    
    frequency_grids = []
    for i, frequencies_i in enumerate(frequencies):
        frequency_grids.append(pos[..., i:i+1] * frequencies_i[None, ...])

    encodings.extend([torch.sin(math.pi * frequency_grid) for frequency_grid in frequency_grids])
    encodings.extend([torch.cos(math.pi * frequency_grid) for frequency_grid in frequency_grids])
    enc = torch.cat(encodings, dim=-1)
    enc = rearrange(enc, "... c -> (...) c")

    return enc