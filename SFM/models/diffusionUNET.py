"""
DiffusionAbstract: shared v-prediction diffusion api (DDPM train / DDIM inference schedulers, sensor prep, losses), subclasses supply _forward and _inference.
DiffusionUNET: sensor conditioned diffusion with a transformer conditioner and a cross attention diffusers UNet2DConditionModel.

Globals:
    - debug: when True, turns on autograd anomaly detection
"""

import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import UNet2DConditionModel, DDPMScheduler, DDIMScheduler
from SFM.models.transformer import Tokenizer
from SFM.models.abstract import ModelClass
from dataclasses import dataclass

# --- Developer Debug Flags ---
debug = False
torch.autograd.set_detect_anomaly(debug)

""" Original Implementation for Referance
@auto_self_assign
class DiffusionUNET(ModelClass):

    def __init__(
        self,
        row:                    int,
        col:                    int,
        output_size:            tuple[int, int],
        num_sensors:            int,
        seq_len:                int,
        in_channels:            int,
        out_channels:           int,
        indexing_method:        str,
        num_train_timesteps:    int,
        num_inference_timesteps:int,
        dropout:                float,
        device:                 str,
        tokenizer_mode:         str = "sequence",
        momentum:               float = 0.05,
        condition_transformer_d_model: int = 128,
        condition_transformer_n_head:  int = 4,
        condition_transformer_layers:  int = 4,
        condition_transformer_dim_feedforward: int = 256

    ):
        super().__init__()

        self.eps                    = 10e-12
        self.row                    = row
        self.col                    = col

        assert(tokenizer_mode in ["sequence", "point"]), f"[PatchTransformer]: token_mode should be [sequence, point] got, {tokenizer_mode}!"
        assert(row >= output_size[0] and col >= output_size[1]), f"[DiffusionUNET]: unet size ({row}, {col}) must cover output_size {output_size}"
        assert(row % 8 == 0 and col % 8 == 0), f"[DiffusionUNET]: unet size ({row}, {col}) must be divisible by 8"

        self.tokenizer = Tokenizer(self.seq_len * self.in_channels if tokenizer_mode == "sequence" else self.in_channels, condition_transformer_d_model, norm_to_max=max(*output_size))

        self.condition_transformer = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model         = condition_transformer_d_model,
                nhead           = condition_transformer_n_head,
                dim_feedforward = condition_transformer_dim_feedforward,
                batch_first     = True,
            ),
            num_layers = condition_transformer_layers,
        )

        self.unet = UNet2DConditionModel(
            sample_size=(row, col),
            in_channels=out_channels,
            out_channels=out_channels,
            layers_per_block=2,
            block_out_channels=(32, 64, 128, 256),
            down_block_types=(
                "CrossAttnDownBlock2D",
                "CrossAttnDownBlock2D",
                "CrossAttnDownBlock2D",
                "CrossAttnDownBlock2D",
            ),
            up_block_types=(
                "CrossAttnUpBlock2D",
                "CrossAttnUpBlock2D",
                "CrossAttnUpBlock2D",
                "CrossAttnUpBlock2D",
            ),
            cross_attention_dim=condition_transformer_d_model,
            dropout=dropout,
        )

        # --- scheulers for standard inference ---
        sch_kwargs = dict(
            num_train_timesteps=num_train_timesteps,
            beta_schedule="linear",
            clip_sample=False,
        )

        self.ddpm_scheduler = DDPMScheduler(**sch_kwargs, prediction_type="v_prediction")
        self.ddim_scheduler = DDIMScheduler(**sch_kwargs, prediction_type="v_prediction")

    def forward(self, batch):
        condition, locations, normalized_target, _ = self.prepare(batch)
        v_pred, v_target = self._forward(condition, locations, self.to_unet(normalized_target))
        return self.from_unet(v_pred), self.from_unet(v_target) # loss in v space

    @torch.no_grad()
    def inference(self, batch, seed: int = 0, eta: int = 0):
        condition, locations, normalized_target, raw_target = self.prepare(batch)
        x, _ = self.predict(condition, locations, self.to_unet(normalized_target), seed=seed, eta=eta)
        out = self.from_unet(x)
        return self.denormalise(out, self.output_mean, self.output_std, dim=-1), raw_target # physical units

    def prepare(self, batch):
        batch        = self.standard_input(batch, self.output_size, self.seq_len, self.out_channels)
        return_dict  = self.index(batch, self.num_sensors, self.indexing_method, True, channel_slice=slice(0, self.in_channels, None))
        sensors_vals = return_dict["sensors_vals"]   # [batch, seq, numsens, channels]
        sensors_locs = return_dict["sensor_locs"]    # [batch, numsens, 2]
        raw_target   = return_dict["target"]         # [batch, x, y, channels]

        # stats only move in training, but the sensors are normalised in both modes so eval sees the same input scale
        if self.training:
            self.update_running_stats(sensors_vals, self.in_channels, self.out_channels, self.momentum)

        sensors_vals = self.normalise(sensors_vals, self.input_mean, self.input_std, dim=-1)
        return sensors_vals, sensors_locs, self.normalise(raw_target, self.output_mean, self.output_std, dim=-1), raw_target

    def to_unet(self, x: torch.Tensor):
        return F.pad(x.permute(0, 3, 1, 2), (0, self.col - x.shape[2], 0, self.row - x.shape[1]))

    def from_unet(self, x: torch.Tensor):
        return x[:, :, :self.output_size[0], :self.output_size[1]].permute(0, 2, 3, 1)

    def losses(self, output: torch.Tensor, target: torch.Tensor, extralosses: dict = {}, **kwargs):
        return losses(output, target, extra_losses=extralosses, **kwargs)

    def _forward(self,
                condition:  torch.Tensor,
                locations:  torch.Tensor,
                target:     torch.Tensor,
                ):

        batch       = target.shape[0]
        device      = target.device
        noise       = torch.randn_like(target)
        timesteps   = torch.randint(0, self.ddpm_scheduler.config.num_train_timesteps, (batch,), device=device).long()


        ctx         = self.condition_transformer(self.tokenizer.sensor(condition, locations, mode=self.tokenizer_mode))

        noisy       = self.ddpm_scheduler.add_noise(target, noise, timesteps) # x_t = α·x₀ + σ·ε   (model input)

        v_target    = self.ddpm_scheduler.get_velocity(target, noise, timesteps) # v = α·ε − σ·x₀   (training target)

        with torch.autocast(str(self.device), dtype=torch.bfloat16): 
            v_pred = self.unet(noisy, timesteps, encoder_hidden_states=ctx).sample
        
        v_pred = v_pred.float()

        return v_pred, v_target

    @torch.no_grad()
    def predict(
        self,
        condition:     torch.Tensor,
        locations:     torch.Tensor,
        batch_target:  torch.Tensor | None = None,
        seed:          int = 0,
        eta:           int = 0,
    ) -> torch.Tensor:

        history = []
        batch_size = condition.shape[0]
        generator = torch.Generator(device=self.device).manual_seed(seed)
        self.ddim_scheduler.set_timesteps(self.num_inference_timesteps, device=self.device)
        self.eval()

        ctx  = self.condition_transformer(self.tokenizer.sensor(condition, locations, mode=self.tokenizer_mode))

        original_noise = torch.randn(batch_size, self.out_channels, self.row, self.col, device=self.device, generator=generator)
        x = original_noise

        for t in tqdm.tqdm(self.ddim_scheduler.timesteps, colour="green", desc="Sampling", leave=False):

            with torch.autocast(str(self.device), dtype=torch.bfloat16): 
                v_pred = self.unet(x, t, encoder_hidden_states=ctx).sample
            
            v_pred = v_pred.float()

            stepped = self.ddim_scheduler.step(v_pred, t, x, generator=generator, eta=eta)

            if batch_target is not None:
                v_target = self.ddpm_scheduler.get_velocity(batch_target, original_noise, t)

            history.append({
                "noisy": x.cpu().detach(),
                "v_pred" : v_pred.cpu().detach(),
                "v_target" : v_target.cpu().detach() if batch_target is not None else None,
                "prev_sample": stepped.prev_sample.cpu().detach(),
                "pred_original_sample": stepped.pred_original_sample.cpu().detach(),
            })

            x = stepped.prev_sample

        return x, history
"""

@dataclass(eq=False, repr=False) #! eq=False keeps nn.Module's hash, repr=False keeps ModelClass's repr
class DiffusionAbstract(ModelClass):
    """
    Shared v-prediction diffusion api. A subclass appends its own fields and implements _forward (training) and _inference (sampling).
    A dataclass whose fields are the arguments stored on self, the shared api is inherited from ModelClass (abstract.py).

    Fields:
        output_size: (x, y) of one snapshot
        num_sensors: number of sensors
        seq_len: timesteps in the sensor window
        in_channels: channels the sensors read
        out_channels: channels of the field the model outputs
        indexing_method: sensor layout handed to index (abstract.py)
        device: the device the model runs on
        num_train_timesteps: diffusion steps of the training schedule
        num_inference_timesteps: DDIM steps at inference
        dropout: dropout of the subclass's network
        tokenizer_mode: "sequence" or "point" sensor tokens (Tokenizer, transformer.py)
        momentum: momentum of the running stats
    """
    #! subclasses append their own fields after these, so every field past device needs a default
    output_size:             tuple[int, int]
    num_sensors:             int
    seq_len:                 int
    in_channels:             int
    out_channels:            int
    indexing_method:         str
    device:                  str
    num_train_timesteps:     int = 1000
    num_inference_timesteps: int = 50
    dropout:                 float = 0.1
    tokenizer_mode:          str = "sequence"
    momentum:                float = 0.05

    def __post_init__(self):
        """Builds the DDPM (training) and DDIM (inference) schedulers, linear betas and v-prediction."""
        super().__init__()
        assert(self.tokenizer_mode in ["sequence", "point"]), f"[{type(self).__name__}]: tokenizer_mode should be [sequence, point] got, {self.tokenizer_mode}!"

        # --- scheulers for standard inference ---
        sch_kwargs = dict(
            num_train_timesteps=self.num_train_timesteps,
            beta_schedule="linear",
            clip_sample=False,
        )

        self.ddpm_scheduler = DDPMScheduler(**sch_kwargs, prediction_type="v_prediction")
        self.ddim_scheduler = DDIMScheduler(**sch_kwargs, prediction_type="v_prediction")

    def forward(self, batch):
        """
        Samples the sensors and target (always normalised with the running stats, updated in training only) and predicts the velocity at a random timestep.

        Input:
            batch: [batch, seq_len, x, y, out_channels]
        Output:
            (v_pred, v_target): both [batch, x, y, out_channels], the loss lives in v space
        """
        condition, locations, normalized_target, _ = self.prepare(batch)
        v_pred, v_target = self._forward(condition, locations, normalized_target)
        return v_pred, v_target

    @torch.no_grad()
    def inference(self, batch, seed: int | None = None, eta: float = 0.0):
        """
        What generate_test_dataset calls, DDIM samples the field for a batch.

        Input:
            batch: [batch, seq_len, x, y, out_channels]
            seed: seeds the sampling noise, None uses the global rng
            eta: DDIM eta, 0.0 is deterministic
        Output:
            (output, target): both [batch, x, y, out_channels] in physical units
        """
        condition, locations, normalized_target, raw_target = self.prepare(batch)
        x, _ = self._inference(condition, locations, normalized_target, seed=seed, eta=eta)
        return self.denormalise(x, self.output_mean, self.output_std, dim=-1), raw_target # physical units

    def prepare(self, batch):
        """
        Samples the sensors and the target from a batch and normalises both with the running stats (updated in training only).

        Input:
            batch: [batch, seq_len, x, y, out_channels]
        Output:
            (sensors_vals, sensors_locs, target, raw_target): normalised sensors [batch, seq_len, num_sensors, in_channels], locations [batch, num_sensors, 2],
            normalised target and physical target, both [batch, x, y, out_channels]
        """
        batch        = self.standard_input(batch, self.output_size, self.seq_len, self.out_channels)
        
        return_dict  = self.index(batch, 
                                  self.num_sensors, 
                                  self.indexing_method, 
                                  True, 
                                  slice(0, self.in_channels, None))
        
        sensors_vals = return_dict["sensors_vals"]   # [batch, seq, numsens, channels]
        sensors_locs = return_dict["sensor_locs"]    # [batch, numsens, 2]
        raw_target   = return_dict["target"]         # [batch, x, y, channels]

        # stats only move in training, but the sensors are normalised in both modes so eval sees the same input scale
        if self.training:
            self.update_running_stats(sensors_vals, self.in_channels, self.out_channels, self.momentum)

        norm_sensors_vals   = self.normalise(sensors_vals, self.input_mean, self.input_std, dim=-1)
        norm_target         = self.normalise(raw_target, self.output_mean, self.output_std, dim=-1)
        
        return norm_sensors_vals, sensors_locs, norm_target, raw_target

    @staticmethod
    def pad_to_divisable(tensor: torch.Tensor, divisor: int):
        """
        Zero pads the last two dims up to a multiple of divisor.

        Input:
            tensor: [batch, c, x, y]
            divisor: the multiple to pad up to
        Output:
            (padded tensor [batch, c, row, col], (row, col), (padX, padY))
        """
        B, C, X, Y = tensor.shape
        Xnew  = -(-X // divisor) * divisor
        Ynew  = -(-Y // divisor) * divisor
        padX = Xnew - X
        padY = Ynew - Y

        return F.pad(tensor, (0, padY, 0, padX)), (Xnew, Ynew), (padX, padY)

@dataclass(eq=False, repr=False) #! eq=False keeps nn.Module's hash, repr=False keeps ModelClass's repr
class DiffusionUNET(DiffusionAbstract):
    """
    Sensor conditioned diffusion: a transformer over the sensor tokens conditions a cross attention diffusers UNet2DConditionModel.

    Fields:
        DiffusionAbstract's fields come first (output_size ... momentum), dropout is the dropout of the UNet
        condition_transformer_d_model / _n_head / _layers / _dim_ffd: token width, heads, layers and feedforward width of the conditioner
    """
    # DiffusionAbstract's fields come first (output_size ... momentum)
    condition_transformer_d_model:  int = 128
    condition_transformer_n_head:   int = 4
    condition_transformer_layers:   int = 4
    condition_transformer_dim_ffd:  int = 256

    def __post_init__(self):
        """Builds the sensor Tokenizer + transformer conditioner and the cross attention UNet sized to output_size padded up to a multiple of 8."""
        super().__post_init__() # schedulers + ModelClass.__init__

        _, self.padded_size, _ = DiffusionUNET.pad_to_divisable(torch.zeros((1, 1, *self.output_size)), 8)

        self.tokenizer = Tokenizer(input_ch     = self.seq_len * self.in_channels if self.tokenizer_mode == "sequence" else self.in_channels,
                                   d_model      = self.condition_transformer_d_model,
                                   norm_to_max  = max(*self.output_size))

        self.condition_transformer = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model         = self.condition_transformer_d_model,
                nhead           = self.condition_transformer_n_head,
                dim_feedforward = self.condition_transformer_dim_ffd,
                batch_first     = True,
            ),
            num_layers = self.condition_transformer_layers,
        )

        self.unet = UNet2DConditionModel(
            sample_size=self.padded_size,
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            layers_per_block=2,
            block_out_channels=(32, 64, 128, 256),
            down_block_types=(
                "DownBlock2D",
                "CrossAttnDownBlock2D",
                "CrossAttnDownBlock2D",
                "CrossAttnDownBlock2D",
            ),
            up_block_types=(
                "CrossAttnUpBlock2D",
                "CrossAttnUpBlock2D",
                "CrossAttnUpBlock2D",
                "UpBlock2D",
            ),
            cross_attention_dim=self.condition_transformer_d_model,
            dropout=self.dropout,
        )

    def to_unet(self, x: torch.Tensor):
        """[batch, x, y, c] -> zero padded [batch, c, row, col]."""
        return DiffusionUNET.pad_to_divisable(x.permute(0, 3, 1, 2), 8)[0]

    def from_unet(self, x: torch.Tensor):
        """Zero padded [batch, c, row, col] -> [batch, x, y, c]."""
        return x[:, :, :self.output_size[0], :self.output_size[1]].permute(0, 2, 3, 1)

    def _model_pass(self, x, t, condition):
        """
        One UNet call, padding in and chopping out.

        Input:
            x: [batch, x, y, out_channels], the noisy field
            t: the timestep, one or [batch]
            condition: [batch, tokens, condition_transformer_d_model], the conditioner's output
        Output:
            [batch, x, y, out_channels]
        """
        return self.from_unet(self.unet(self.to_unet(x), t, condition).sample)

    def _forward(self,
                condition:  torch.Tensor,
                locations:  torch.Tensor,
                target:     torch.Tensor,
                ):
        """
        Noises the target at a random timestep and predicts its velocity.

        Input:
            condition: [batch, seq_len, num_sensors, in_channels], the normalised sensors
            locations: [batch, num_sensors, 2] as (x, y) grid indices
            target: [batch, x, y, out_channels], normalised
        Output:
            (v_pred, v_target): both [batch, x, y, out_channels]
        """

        assert(target.shape[1:] == (*self.output_size, self.out_channels))

        batch       = target.shape[0]
        device      = target.device
        noise       = torch.randn_like(target)
        timesteps   = torch.randint(0, self.ddpm_scheduler.config.num_train_timesteps, (batch,), device=device).long()

        ctx         = self.condition_transformer(self.tokenizer.sensor(condition, locations, mode=self.tokenizer_mode))

        noisy       = self.ddpm_scheduler.add_noise(target, noise, timesteps) # x_t = α·x₀ + σ·ε   (model input)

        v_target    = self.ddpm_scheduler.get_velocity(target, noise, timesteps) # v = α·ε − σ·x₀   (training target)

        with torch.autocast(str(self.device), dtype=torch.bfloat16): 
            v_pred = self._model_pass(noisy, timesteps, ctx)
        
        v_pred = v_pred.float()

        return v_pred, v_target

    @torch.no_grad()
    def _inference(
        self,
        condition:     torch.Tensor,
        locations:     torch.Tensor,
        target:        torch.Tensor | None = None,
        seed:          int | None = None,
        eta:           float = 0.0,
    ) -> torch.Tensor:
        """
        DDIM samples a field from noise under the condition, keeping every step.

        Input:
            condition: [batch, seq_len, num_sensors, in_channels], the normalised sensors
            locations: [batch, num_sensors, 2] as (x, y) grid indices
            target: optional [batch, x, y, out_channels] normalised target, only used to record v_target in the history
            seed: seeds the sampling noise, None uses the global rng
            eta: DDIM eta, 0.0 is deterministic
        Output:
            (sample [batch, x, y, out_channels], history): history is a list of per step dicts (noisy, v_pred, v_target, prev_sample, pred_original_sample)
        """
        
        self.eval()

        history     = []
        batch_size  = condition.shape[0]
        generator   = torch.Generator(device=self.device).manual_seed(seed) if seed is not None else None # None -> global rng, a fresh unseeded Generator repeats the same noise every call

        self.ddim_scheduler.set_timesteps(self.num_inference_timesteps, device=self.device)

        ctx =self.condition_transformer(self.tokenizer.sensor(condition, locations, mode=self.tokenizer_mode))

        original_noise = torch.randn(batch_size, self.output_size[0], self.output_size[1], self.out_channels, device=self.device, generator=generator)
        x = original_noise

        for t in tqdm.tqdm(self.ddim_scheduler.timesteps, colour="green", desc="Sampling", leave=False):

            with torch.autocast(str(self.device), dtype=torch.bfloat16):
                v_pred = self._model_pass(x, t, ctx)

            v_pred = v_pred.float()

            stepped = self.ddim_scheduler.step(v_pred, t, x, generator=generator, eta=eta)

            if target is not None:
                v_target = self.ddpm_scheduler.get_velocity(target, original_noise, t)

            history.append({
                "noisy": x.cpu().detach(),
                "v_pred" : v_pred.cpu().detach(),
                "v_target" : v_target.cpu().detach() if target is not None else None,
                "prev_sample": stepped.prev_sample.cpu().detach(),
                "pred_original_sample": stepped.pred_original_sample.cpu().detach(),
            })

            x = stepped.prev_sample

        return x, history
    