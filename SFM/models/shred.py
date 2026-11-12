
"""
!SHRED: simplified SHRED (Kutz et al.), a stacked lstm (or mamba2) over the sensor window decoded by linear / relu / dropout blocks into the full field.

debug: when True, prints the shape traffic and turns on autograd anomaly detection.

!SHRED(ModelClass, nn.Module), with auto_self_assign storing every __init__ argument on self and the shared api inherited from abstract.py:

    !__init__: builds the encoder (encoder_mode "lstm" or "mamba2") and the decoder (decoder_mode "mlp" widening the final hidden state out to a flattened field,
        or "perceiver" where fourier encoded pixel queries cross-attend (Perceiver IO decoder) to the encoder's per-timestep outputs).
        io: takes output_size, seq_len, num_sensors, hidden_size, hidden_layers, dropout, device, decoder_layer_sizes, in_channels, out_channels, indexing_method,
            use_layernorm, normalize_outputs, momentum, locations_retrofit, lstm_input_proj, encoder_mode and decoder_mode and returns nothing.

    !forward: samples the sensors and, when normalize_outputs, normalises them with the running stats (updated in training only).
        io: takes batch [batch, seq_len, x, y, out_channels] and returns (output, target) both [batch, x, y, out_channels], normalised in training, physical in eval.

    !predict: encodes the sensor window and decodes the hidden state after the final timestep into the field.
        io: takes sensors_vals [batch, seq, num_sensors, in_channels] and sensors_locs (unused for now) and returns [batch, x, y, out_channels].

!KutzSHRED(ModelClass, nn.Module): the reference pyshred SHRED (lstm + mlp) wrapped in the same forward / normalisation / losses api.

    Authors: text = Claude Opus 5 | code = Gabriel Bray
"""

import torch
import pyshred as ps
import torch.nn as nn
from typing import Any
from SFM.utils.utils import auto_self_assign
from SFM.models.abstract import ModelClass
from SFM.losses.losses import losses
from transformers import Mamba2Config, Mamba2Model, PerceiverConfig
from transformers.models.perceiver.modeling_perceiver import PerceiverBasicDecoder
from SFM.models.fourierEncoder import fourierEncoder

# --- Developer Debug Flags ---
debug = False
torch.autograd.set_detect_anomaly(debug)

@auto_self_assign
class SHRED(ModelClass):
    def __init__(
        self,
        output_size:          tuple[int, int],
        seq_len:              int,
        num_sensors:          int,
        in_channels:          int,
        out_channels:         int,
        indexing_method:      str,
        device:               torch.device | str,
        hidden_size:          int = 256,
        hidden_layers:        int = 1,
        dropout:              float = 0.1,
        decoder_layer_sizes:  list[int] = [512, 2048],
        use_layernorm:        bool = True,
        normalize_outputs:    bool = True,
        momentum:             float = 0.05,
        locations_retrofit:   bool = True,
        lstm_input_proj:      bool = False,
        encoder_mode:         str = "lstm",
        decoder_mode:         str = "mlp",
        prediction_mode:      str = "last", # ["last", "full"]
    ):

        super().__init__()
        assert(encoder_mode in ("lstm", "mamba2")),                          f"[SHRED]: encoder_mode should be [lstm, mamba2] got {encoder_mode}"
        assert(decoder_mode == "mlp"),                                       f"[SHRED]: decoder_mode should be [mlp] got {decoder_mode}"
        assert(prediction_mode in ("last", "full")),                         f"[SHRED]: prediction_mode should be [last, full] got {prediction_mode}"
        assert(not locations_retrofit or seq_len % 2 == 0),                  f"[SHRED]: locations_retrofit embeds each location to a seq_len series, fourierEncoder needs it even, got {seq_len}"
        assert(in_channels <= out_channels),                                 f"[SHRED]: sensors are sliced from the target channels, in_channels {in_channels} > out_channels {out_channels}"
        assert(len(decoder_layer_sizes) > 0),                                 "[SHRED]: decoder_layer_sizes must have at least one layer"
        if encoder_mode == "mamba2": assert((2 * hidden_size) % 16 == 0),    f"[SHRED]: mamba2 needs expand*hidden_size divisible by head_dim (16), got hidden_size {hidden_size}"

        self.flattened_output_size: int = self.output_size[0] * self.output_size[1] * self.out_channels
        self.activation = nn.ReLU

        # each location's sinusoid is a [seq_len] series fed in as an extra "sensor", so retrofit doubles the sensor count
        self.num_tokens: int = self.num_sensors * (2 if locations_retrofit else 1)
        if locations_retrofit:
            self.positionalEmbeding = fourierEncoder(2, embed_dim=self.seq_len, norm_to_max=max(*self.output_size))

        if self.encoder_mode == "lstm":
            if lstm_input_proj:
                self.input_proj = nn.Linear(self.num_tokens * self.in_channels, self.hidden_size)

            _kwargs: dict[str, Any] = dict(
                input_size  = self.hidden_size if lstm_input_proj else self.num_tokens * self.in_channels,
                hidden_size = self.hidden_size,
                num_layers  = self.hidden_layers,
                batch_first = True,
            )
            
            self.encoder = nn.LSTM(**_kwargs)
        
        elif self.encoder_mode == "mamba2":
            head_dim = 16                                   # any divisor of expand*hidden_size
            expand   = 2
            config = Mamba2Config(
                vocab_size         = 1,                     # unused, we feed inputs_embeds; keep tiny so no 32k embedding table is built
                hidden_size        = self.hidden_size,
                num_hidden_layers  = self.hidden_layers,
                expand             = expand,
                head_dim           = head_dim,
                num_heads          = expand * self.hidden_size // head_dim,
                state_size         = 256,
                n_groups           = 1,
                conv_kernel        = 4,
                chunk_size         = self.seq_len,          # one chunk covers the whole sensor window
                use_cache          = False,
                use_bias           = False,
                use_conv_bias      = True,
                residual_in_fp32   = True,
                rescale_prenorm_residual = False,
                tie_word_embeddings = False,
            )
            self.input_proj = nn.Linear(self.num_tokens * self.in_channels, self.hidden_size)
            self.encoder = Mamba2Model(config)
        
        else:
            raise ValueError(f"Encoder Mode should be [lstm, mamba2] got, {encoder_mode}!")

        if use_layernorm: self.layer_norm = nn.LayerNorm(self.hidden_size)

        if self.decoder_mode == "mlp":
            
            decoder = nn.ModuleList()

            previous_layer_size = self.hidden_size
            for size in self.decoder_layer_sizes:
                decoder.append(nn.Linear(previous_layer_size, size))
                decoder.append(self.activation())
                decoder.append(nn.Dropout(self.dropout))
                previous_layer_size = size

            decoder.append(nn.Linear(self.decoder_layer_sizes[-1], self.flattened_output_size))
            self.decoder: nn.Sequential = nn.Sequential(*decoder)

        else:
            raise ValueError(f"Decoder Mode should be [mlp, perceiver] got, {decoder_mode}!")

    @property
    def output_grid_locs(self):
        x = torch.arange(0, self.output_size[0])
        y = torch.arange(0, self.output_size[1])
        grid_x, grid_y = torch.meshgrid(x, y, indexing="ij")
        grid = torch.stack((grid_x, grid_y), dim=-1)
        return grid.to(self.device)

    def forward(self, batch): 
        batch = self.standard_input(batch, self.output_size, self.seq_len, self.out_channels)
        return_dict = self.index(batch, self.num_sensors, self.indexing_method, True, channel_slice=slice(0, self.in_channels, None), seed = 0)

        #! ---------------------------------------------------
        #! print(f"[SHRED]: WARNING SEED = 0 NOT Random Sensors")
    
        sensors_vals = return_dict["sensors_vals"]   # [batch, seq, numsens, channels]
        sensors_locs = return_dict["sensor_locs"]    # [batch, numsens, 2]
        target       = return_dict["target"]         # [batch, x, y, channels]

        if self.prediction_mode == "full":
            target = batch                           # [batch, seq, x, y, channels], every frame is supervised

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
            if self.prediction_mode == "full":
                return self.denormalise(out, self.output_mean, self.output_std, dim=-1)[:, -1], target[:, -1] # physical units
            elif self.prediction_mode == "last":
                return self.denormalise(out, self.output_mean, self.output_std, dim=-1), target # physical units

    def predict(self, sensors_vals: torch.Tensor, sensors_locs: torch.Tensor | None = None) -> torch.Tensor:

        # concat after normalisation so the sinusoid "sensors" stay out of the running stats
        if self.locations_retrofit:
            pos = self.positionalEmbeding(sensors_locs.float()).transpose(1, 2)                                # [B, numsens, seq] -> [B, seq, numsens]
            sensors_vals = torch.cat([sensors_vals, pos[..., None].expand_as(sensors_vals)], dim=2)    # [B, seq, 2*numsens, channels]

        B, S, N, C  = sensors_vals.shape

        # --- classic path ---
        if self.encoder_mode == "lstm":

            h_0 = torch.zeros(self.hidden_layers, B, self.hidden_size, device=self.device)
            c_0 = torch.zeros(self.hidden_layers, B, self.hidden_size, device=self.device)

            inpt = sensors_vals.reshape(B, S, N*C)
            
            if self.lstm_input_proj:
                inpt = self.input_proj(inpt)
   
            seq_out, (h_out, _) = self.encoder(inpt, (h_0, c_0)) # seq_out [B, S, hidden], h_out [B, hidden]
            
            if self.prediction_mode == "last":   hidden_vec = h_out[-1] # [B, hidden]
            elif self.prediction_mode == "full": hidden_vec = seq_out   # [B, S, hidden]

        # --- mamba path ---
        elif self.encoder_mode == "mamba2":

            inpt = self.input_proj(sensors_vals.reshape(B, S, N*C))

            seq_out = self.encoder(inputs_embeds=inpt, use_cache=False).last_hidden_state  
            if self.prediction_mode == "last":   hidden_vec = seq_out[:, -1] # [B, hidden]
            elif self.prediction_mode == "full": hidden_vec = seq_out     # [B, S, hidden]


        # --- decoder --- 
        if self.use_layernorm:
            hidden_vec = self.layer_norm(hidden_vec)

        if self.decoder_mode == "mlp":
            out = self.decoder(hidden_vec)

        if self.prediction_mode == "last":
            return self.standard_output(out, self.output_size, self.out_channels)   # [B, x, y, C]
        elif self.prediction_mode == "full":
            return out.view(B, S, *self.output_size, self.out_channels)             # [B, S, x, y, C]

    def losses(self, output: torch.Tensor, target: torch.Tensor, extralosses: dict = {}, **kwargs):
        return losses(output, target, extra_losses=extralosses, **kwargs)

@auto_self_assign
class KutzSHRED(ModelClass):
    def __init__(
        self,
        output_size:          tuple[int, int],
        seq_len:              int,
        num_sensors:          int,
        in_channels:          int,
        out_channels:         int,
        indexing_method:      str,
        device:               torch.device | str,
        hidden_size:          int = 256,
        hidden_layers:        int = 1,
        dropout:              float = 0.1,
        decoder_layer_sizes:  list[int] = [512, 1024],
        normalize_outputs:    bool = True,
        momentum:             float = 0.05
    ):

        super().__init__()

        self.lstm   = ps.LSTM(hidden_size, hidden_layers, True)
        self.mlp    = ps.MLP(decoder_layer_sizes, dropout)
        self.model  = ps.SHRED(self.lstm, self.mlp)
        self.model.sequence.initialize(input_size=num_sensors*in_channels, lags=seq_len, decoder_type=type(self.model.decoder).__name__)
        self.model.decoder.initialize(input_size=hidden_size, output_size=output_size[0]*output_size[1]*out_channels)
        self.model.to(device)

    def forward(self, batch): 
        batch        = self.standard_input(batch, self.output_size, self.seq_len, self.out_channels)
        return_dict  = self.index(batch, self.num_sensors, self.indexing_method, False, channel_slice=slice(0, self.in_channels, None))
        sensors_vals = return_dict["sensors_vals"]   # [batch, seq, numsens, channels]
        target       = return_dict["target"]         # [batch, x, y, channels]

        if not self.normalize_outputs:
            out = self.predict(sensors_vals.flatten(2))
            return out, target

        # stats only move in training, but the sensors are normalised in both modes so eval sees the same input scale
        if self.training:
            self.update_running_stats(sensors_vals, self.in_channels, self.out_channels, self.momentum)

            if debug:
                print(f"[{type(self).__name__}]: momentum = {self.momentum}")
                print(f"[KutzSHRED]: input mean = {self.input_mean}, output mean = {self.output_mean}")
                print(f"[KutzSHRED]: input std = {self.input_std}, output std = {self.output_std}")

        sensors_vals = self.normalise(sensors_vals, self.input_mean, self.input_std, dim=-1)
        out = self.predict(sensors_vals.flatten(2))

        if self.training: 
            return out, self.normalise(target, self.output_mean, self.output_std, dim=-1) # loss in normalised space
        else:             
            return self.denormalise(out, self.output_mean, self.output_std, dim=-1), target # physical units

    def predict(self, sensors_vals: torch.Tensor, sensors_locs: torch.Tensor | None = None) -> torch.Tensor:
    
        out = self.model.forward(sensors_vals)
        
        out = self.standard_output(out, self.output_size, self.out_channels)
        
        return out

    def losses(self, output: torch.Tensor, target: torch.Tensor, extralosses: dict = {}, **kwargs):
        return losses(output, target, extra_losses=extralosses, **kwargs)


if __name__ == "__main__":
    pass

