"""
PatchTransformer: shared base for the transformer models, sensor tokens + patch queries decoded through a pixel shuffle conv head.
EncoderDecoderTransformer: memory = encoder(tokens), latent = decoder(queries, memory).
EncoderTransformer: latent = encoder(cat(tokens, queries)) at the query positions.
Tokenizer / PatchConvHead / PatchConvHeadTranspose: the sensor / query tokens and the patch <-> pixel heads, also used by the diffusion models.

Globals:
    - debug: when True, prints the running stats and turns on autograd anomaly detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from SFM.losses.losses import losses
from SFM.models.abstract import ModelClass
from typing import ClassVar
from dataclasses import dataclass
from SFM.models.fourierEncoder import fourierEncoder

# --- Developer Debug Flags ---
debug = False
torch.autograd.set_detect_anomaly(debug)

@dataclass(eq=False, repr=False) #! eq=False keeps nn.Module's hash, repr=False keeps ModelClass's repr
class PatchTransformer(ModelClass):
    """
    Shared base of the transformer models. A subclass sets use_decoder and implements predict.
    A dataclass whose fields are the arguments stored on self, the shared api is inherited from ModelClass (abstract.py).

    Fields:
        output_size: (x, y) of one snapshot
        seq_len: timesteps in the sensor window
        num_sensors: number of sensors
        in_channels: channels the sensors read
        out_channels: channels of the field the model outputs
        indexing_method: sensor layout handed to index (abstract.py)
        d_model: width of the tokens
        nhead: attention heads
        num_layers: layers of the encoder (and of the decoder)
        tokenize_mode: "sequence" (seq_len * in_channels -> d_model, one token per sensor) or "point" (in_channels -> d_model, one token per sensor per timestep)
        patch_size: pixels per side of one query patch
        dim_feedforward: width of the feedforward layers
        dropout: dropout of the transformer layers
        normalize_outputs: normalises the sensors and the target with the running stats
        momentum: momentum of the running stats
    """
    use_decoder: ClassVar[bool] # set by the subclass, ClassVar keeps it out of the dataclass fields

    output_size:          tuple[int, int]
    seq_len:              int
    num_sensors:          int
    in_channels:          int
    out_channels:         int
    indexing_method:      str
    d_model:              int = 128
    nhead:                int = 4
    num_layers:           int = 8
    tokenize_mode:        str = "sequence"     #? ["sequence", "point"]
    patch_size:           int = 6
    dim_feedforward:      int = 256
    dropout:              float = 0.1
    normalize_outputs:    bool = True
    momentum:             float = 0.05

    def __post_init__(self):
        """Builds the transformer encoder (+ decoder when use_decoder), the Tokenizer and the PatchConvHead."""
        super().__init__()
        assert(self.tokenize_mode in ["sequence", "point"]), f"[PatchTransformer]: tokenize_mode should be [sequence, point] got, {self.tokenize_mode}!"

        self.latent_size = tuple(-(-s // self.patch_size) for s in self.output_size) # ceil, the extra pixels are chopped after the head

        config = dict(
            d_model         = self.d_model,
            nhead           = self.nhead,
            dim_feedforward = self.dim_feedforward,
            dropout         = self.dropout,
            activation      = "gelu",
            batch_first     = True,
            norm_first      = True,
        )

        self.TransformerEncoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(**config), num_layers=self.num_layers, norm=nn.LayerNorm(self.d_model), enable_nested_tensor=False
        )
        if self.use_decoder:
            self.TransformerDecoder = nn.TransformerDecoder(
                nn.TransformerDecoderLayer(**config), num_layers=self.num_layers, norm=nn.LayerNorm(self.d_model)
            )

        self.tokenizer = Tokenizer(input_ch=self.seq_len*self.in_channels if self.tokenize_mode == "sequence" else self.in_channels,
                             d_model=self.d_model,
                             norm_to_max=max(self.latent_size)*self.patch_size,
                             patch_size=self.patch_size,
                             output_size=self.output_size,
                             latent_size=self.latent_size,
                             )

        self.convHead = PatchConvHead(self.d_model, self.out_channels, self.patch_size, self.latent_size, self.output_size)
    
    def forward(self, batch):
        """
        Samples the sensors and, when normalize_outputs, normalises them with the running stats (updated in training only).

        Input:
            batch: [batch, seq_len, x, y, out_channels]
        Output:
            (output, target): both [batch, x, y, out_channels], normalised in training, physical in eval
        """
        batch        = self.standard_input(batch, self.output_size, self.seq_len, self.out_channels)
        return_dict  = self.index(batch, self.num_sensors, self.indexing_method, True, channel_slice=slice(0, self.in_channels, None))
        sensors_vals = return_dict["sensors_vals"]   # [batch, seq, numsens, channels]
        sensors_locs = return_dict["sensor_locs"]    # [batch, numsens, 2]
        target       = return_dict["target"]         # [batch, x, y, channels]

        if not self.normalize_outputs:
            out = self.predict(sensors_vals, sensors_locs)
            return out, target

        # stats only move in training, but the sensors are normalised in both modes so eval sees the same input scale
        if self.training:
            self.update_running_stats(sensors_vals, self.in_channels, self.out_channels, self.momentum)

            if debug:
                print(f"[{type(self).__name__}]: momentum = {self.momentum}")
                print(f"[{type(self).__name__}]: input mean = {self.input_mean}, output mean = {self.output_mean}")
                print(f"[{type(self).__name__}]: input std = {self.input_std}, output std = {self.output_std}")

        sensors_vals = self.normalise(sensors_vals, self.input_mean, self.input_std, dim=-1)
        out = self.predict(sensors_vals, sensors_locs)

        if self.training:
            return out, self.normalise(target, self.output_mean, self.output_std, dim=-1) # loss in normalised space
        else:
            return self.denormalise(out, self.output_mean, self.output_std, dim=-1), target # physical units

    def losses(self, output: torch.Tensor, target: torch.Tensor, extralosses: dict = {}, **kwargs):
        """
        Hands off to losses/losses.py with a small smoothness weight (use_smooth = 1e-3) against patch seams.

        Input:
            output / target: both [batch, x, y, out_channels]
            extralosses: {name: scalar tensor}, appended as they are
            kwargs: use_* overrides of losses()
        Output:
            (list of scalar loss tensors, list of matching names)
        """
        return losses(output, target, extra_losses=extralosses, use_smooth=1e-3, **kwargs) #? small smoothness weight against patch seams

class EncoderDecoderTransformer(PatchTransformer):
    """memory = encoder(sensor tokens), latent = decoder(patch queries, memory), field = conv head(latent)."""
    use_decoder = True

    def predict(self, sensors_vals: torch.Tensor, sensors_locs: torch.Tensor):
        """
        Input:
            sensors_vals: [batch, seq, num_sensors, in_channels]
            sensors_locs: [batch, num_sensors, 2] as (x, y) grid indices
        Output:
            [batch, x, y, out_channels]
        """
        B = sensors_vals.shape[0]
        memory = self.TransformerEncoder(self.tokenizer.sensor(sensors_vals, sensors_locs, self.tokenize_mode))
        latent = self.TransformerDecoder(self.tokenizer.queries(B), memory)
        out = self.convHead(latent)
        return self.standard_output(out, self.output_size, self.out_channels)

class EncoderTransformer(PatchTransformer):
    """latent = encoder(cat(sensor tokens, patch queries)) read at the query positions, field = conv head(latent)."""
    use_decoder = False

    #? You may consider if interesting, for EncoderTransformer to force a AE-style decoder on the sensor tokens to yeild correct positions and values.
    #?     - Gabriel Bray -> Future Gabriel Bray 

    def predict(self, sensors_vals: torch.Tensor, sensors_locs: torch.Tensor):
        """
        Input:
            sensors_vals: [batch, seq, num_sensors, in_channels]
            sensors_locs: [batch, num_sensors, 2] as (x, y) grid indices
        Output:
            [batch, x, y, out_channels]
        """
        queries = self.tokenizer.queries(sensors_vals.shape[0])
        latent  = self.TransformerEncoder(torch.cat([self.tokenizer.sensor(sensors_vals, sensors_locs, self.tokenize_mode), queries], dim=1))
        out = self.convHead(latent[:, -queries.shape[1]:])
        return self.standard_output(out, self.output_size, self.out_channels)

class Tokenizer(nn.Module):
    """Sensor tokens and patch queries: a linear layer on the sensor values plus a fourier embedding of the locations."""
    def __init__(self, input_ch: int, 
                 d_model:           int, 
                 norm_to_max:       int   = None, 
                 patch_size:        int = None, 
                 output_size:       tuple = None, 
                 latent_size:       tuple = None, 
                 share_embedder:    bool  = True,
                 features:          int = 2):
        """
        Builds the fourier location embedding(s) and the linear tokenizer.

        Input:
            input_ch: values per token, seq_len * in_channels for "sequence" tokens or in_channels for "point" tokens
            d_model: width of the tokens
            norm_to_max: value the locations are divided by before the fourier features
            patch_size: pixels per side of one query patch
            output_size: (x, y) of the field
            latent_size: (patches_x, patches_y), defaults to ceil(output_size / patch_size)
            share_embedder: the queries reuse the sensors' fourier embedding
            features: dims of one location
        """

        super().__init__()

        self.patch_size     = patch_size
        self.input_ch       = input_ch
        self.d_model        = d_model
        self.output_size    = output_size
        self.latent_size    = latent_size

        self.fourierEmbedding = fourierEncoder(features, d_model, 8, norm_to_max=norm_to_max)
        
        if share_embedder:
            self.queriesfourierEmbedding = self.fourierEmbedding
        else:
            self.queriesfourierEmbedding = fourierEncoder(features, d_model, num_frequencies=8, norm_to_max=norm_to_max)

        self.tokenizer = nn.Linear(input_ch, d_model)

    def sensor(self, sensors_vals: torch.Tensor, sensors_locs: torch.Tensor, mode = "sequence"):
        """
        tokenizer(sensor values) + embed(sensor location).

        Input:
            sensors_vals: [batch, seq, num_sensors, in_channels]
            sensors_locs: [batch, num_sensors, 2] as (x, y) grid indices
            mode: "sequence" (one token per sensor) or "point" (one token per sensor per timestep)
        Output:
            [batch, num_sensors, d_model] for "sequence", [batch, seq * num_sensors, d_model] for "point"
        """
        assert(sensors_vals.shape[0] == sensors_locs.shape[0])
        assert(sensors_vals.shape[2] == sensors_locs.shape[1])
        assert(mode in ["sequence", "point"]), f"[Tokenizer]: tokenize_mode should be [sequence, point] got, {mode}!"
        
        embeded_sensors = self.fourierEmbedding(sensors_locs.float())                                    

        if mode == "sequence":
            return (self.tokenizer(sensors_vals.permute(0, 2, 1, 3).flatten(2)) + embeded_sensors)                    
        elif mode == "point":
            return (self.tokenizer(sensors_vals) + embeded_sensors[:, None]).flatten(1, 2)                    

    def queries(self, B: int):
        """
        The fourier embedding of the centre pixel of every patch, as in the Perceiver IO decoder.

        Input:
            B: batch size
        Output:
            [B, patches_x * patches_y, d_model]
        """
        return self.queriesfourierEmbedding(self.latent_locs).flatten(0, 1).unsqueeze(0).expand(B, -1, -1) # [B, Px*Py, d_model]

    @property
    def output_locs(self):
        """Output: every (x, y) grid index as floats, [x, y, 2]."""
        x = torch.arange(self.output_size[0])
        y = torch.arange(self.output_size[1])
        grid_x, grid_y = torch.meshgrid(x, y, indexing="ij")
        return torch.stack((grid_x, grid_y), dim=-1).float().to(self.params_device)                 

    @property
    def latent_locs(self):
        """Output: the centre pixel (x, y) of every patch_size patch on the (ceil padded) grid, [patches_x, patches_y, 2]."""
        if self.latent_size is not None:
            x = torch.arange(self.latent_size[0]) * self.patch_size + self.patch_size // 2
            y = torch.arange(self.latent_size[1]) * self.patch_size + self.patch_size // 2
        else:
            x = torch.arange(-(-self.output_size[0]//self.patch_size)) * self.patch_size + self.patch_size // 2
            y = torch.arange(-(-self.output_size[1]//self.patch_size)) * self.patch_size + self.patch_size // 2

        grid_x, grid_y = torch.meshgrid(x, y, indexing="ij")
        return torch.stack((grid_x, grid_y), dim=-1).float().to(self.params_device)                 

    @property
    def params_device(self):
        """Output: the torch.device the parameters live on."""
        return next(self.parameters()).device

class PatchConvHead(nn.Module):
    """
    Patch tokens -> field. Every token is projected to one hidden vector per pixel of its patch (pixel shuffle), followed by a two layer
    resnet style conv head.
    """
    def __init__(self, in_channels: int, out_channels: int, patch_size: int, latent_size: tuple, output_size: tuple):
        """
        Input:
            in_channels: width of the tokens
            out_channels: channels of the field
            patch_size: pixels per side of one patch
            latent_size: (patches_x, patches_y)
            output_size: (x, y) of the field, the padded border past it is chopped
        """
        super().__init__()
        self.in_channels    = in_channels
        self.out_channels   = out_channels
        self.output_size    = output_size
        self.latent_size    = latent_size
        self.patch_size     = patch_size
        self.hidden         = 32 # channels per pixel after the shuffle, small so the full res convs stay cheap
        self.proj           = nn.Linear(in_channels, patch_size**2 * self.hidden) # one hidden vector per pixel in the patch instead of a nearest copy
        self.head_in        = nn.Conv2d(in_channels=self.hidden, out_channels=self.hidden, kernel_size=3, padding="same")
        self.head_out       = nn.Conv2d(in_channels=self.hidden, out_channels=out_channels, kernel_size=3, padding="same")

    def forward(self, latent: torch.Tensor):
        """
        Input:
            latent: [batch, patches_x * patches_y, in_channels]
        Output:
            [batch, out_channels, x, y]
        """
        B = latent.shape[0]
        assert(latent.shape[2] == self.in_channels)

        grid = self.proj(latent).transpose(1, 2).reshape(B, self.hidden * self.patch_size**2, *self.latent_size) # [B, h*p*p, Px, Py]
        grid = F.pixel_shuffle(grid, self.patch_size)                                                             # [B, h, Px*p, Py*p]
        grid = grid + F.gelu(self.head_in(grid))
        out  = self.head_out(grid)[:, :, :self.output_size[0], :self.output_size[1]]    
        
        return out

class PatchConvHeadTranspose(nn.Module):
    """Field -> patch tokens. Unfolds the field into patch_size patches and runs a two layer resnet style linear head on each."""
    def __init__(self, in_channels: int, out_channels: int, patch_size: int, output_size: tuple):
        """
        Input:
            in_channels: channels of the field
            out_channels: width of the tokens
            patch_size: pixels per side of one patch
            output_size: (x, y) of the field
        """
        super().__init__()
        self.in_channels    = in_channels
        self.out_channels   = out_channels
        self.output_size    = output_size
        self.patch_size     = patch_size
        self.latent_size    = tuple(-(-s // patch_size) for s in output_size) # ceil, patch grid [Px, Py]
        self.head_in        = nn.Linear(in_channels * (patch_size**2), in_channels * (patch_size**2))
        self.head_out       = nn.Linear(in_channels * (patch_size**2), out_channels)
    
    def forward(self, inpt: torch.Tensor):
        """
        Input:
            inpt: [batch, in_channels, x, y], reflect padded up to a multiple of patch_size
        Output:
            [batch, patches_x, patches_y, out_channels]
        """
        B, C, X, Y = inpt.shape
        assert(inpt.shape[1] == self.in_channels)
        assert(inpt.ndim == 4)

        if X%self.patch_size != 0 or Y%self.patch_size:
            inpt = F.pad(inpt, (0, -Y%self.patch_size, 0, -X%self.patch_size), mode = "reflect") # F.pad goes last dim first: (Y, X)

        grid = inpt.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size) # [B, in_channels, Lx, Ly, patch_size, patch_size]
        grid = grid.permute(0, 2, 3, 1, 4, 5).flatten(3)
        
        grid = grid + F.gelu(self.head_in(grid))
        out  = self.head_out(grid)

        return out