
"""
DiffusionTransformer: sensor conditioned v-prediction diffusion where one transformer sees a timestep token, the sensor tokens and the noisy field's patch tokens.

Globals:
    - debug: when True, turns on autograd anomaly detection
"""

import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.embeddings import Timesteps, TimestepEmbedding
from SFM.models.diffusionUNET import DiffusionAbstract
from dataclasses import dataclass
from SFM.models.transformer import PatchConvHeadTranspose, PatchConvHead, Tokenizer

# --- Developer Debug Flags ---
debug = False
torch.autograd.set_detect_anomaly(debug)

@dataclass(eq=False, repr=False) #! eq=False keeps nn.Module's hash, repr=False keeps ModelClass's repr
class DiffusionTransformer(DiffusionAbstract):
    """
    Sensor conditioned v-prediction diffusion with a single transformer. forward / inference / prepare come from DiffusionAbstract (diffusionUNET.py).

    Fields:
        DiffusionAbstract's fields come first (output_size ... momentum), dropout is the dropout of the transformer layers
        d_model: width of the tokens
        nhead: attention heads
        num_layers: layers of the transformer
        dim_feedforward: width of the feedforward layers
        patch_size: pixels per side of one patch
    """
    # DiffusionAbstract's fields come first (output_size ... momentum)
    d_model:                int = 64
    nhead:                  int = 4
    num_layers:             int = 4
    dim_feedforward:        int = 256
    patch_size:             int = 3

    def __post_init__(self):
        """Builds the patch encoder, sensor Tokenizer, timestep embedding, transformer and patch decoder (the schedulers come from DiffusionAbstract)."""
        super().__post_init__() # schedulers + ModelClass.__init__

        self.latent_size = tuple(-(-s // self.patch_size) for s in self.output_size) # ceil, patch grid [Px, Py], extra pixels chopped by the decoder

        # noisy field [B, X, Y, out_ch] -> patch tokens [B, Px*Py, d_model]
        self.encoder = PatchConvHeadTranspose(in_channels   = self.out_channels,
                                              out_channels  = self.d_model,
                                              patch_size    = self.patch_size,
                                              output_size   = self.output_size)

        # sensor tokens + fourier position queries for the patch grid
        self.tokenizer = Tokenizer(input_ch     = self.seq_len * self.in_channels if self.tokenizer_mode == "sequence" else self.in_channels,
                                   d_model      = self.d_model,
                                   norm_to_max  = max(self.latent_size) * self.patch_size,
                                   patch_size   = self.patch_size,
                                   output_size  = self.output_size,
                                   latent_size  = self.latent_size)

        # diffusion timestep -> one [B, 1, d_model] token
        self.time_proj      = Timesteps(self.d_model, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.time_embedding = TimestepEmbedding(self.d_model, self.d_model)

        config = dict(
            d_model         = self.d_model,
            nhead           = self.nhead,
            dim_feedforward = self.dim_feedforward,
            dropout         = self.dropout,
            activation      = "gelu",
            batch_first     = True,
            norm_first      = True,
        )

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(**config), num_layers=self.num_layers, norm=nn.LayerNorm(self.d_model), enable_nested_tensor=False
        )

        # patch tokens [B, Px*Py, d_model] -> v field [B, out_ch, X, Y]
        self.decoder = PatchConvHead(in_channels    = self.d_model,
                                     out_channels   = self.out_channels,
                                     patch_size     = self.patch_size,
                                     latent_size    = self.latent_size,
                                     output_size    = self.output_size)

    def _model_pass(self, input_image: torch.Tensor, condition: torch.Tensor, locations: torch.Tensor, timestep: torch.Tensor):
        """
        One transformer call over cat(timestep token, sensor tokens, noisy patch tokens), read at the patch positions and decoded to the field.

        Input:
            input_image: [batch, x, y, out_channels], the noisy field
            condition: [batch, seq_len, num_sensors, in_channels], the normalised sensors
            locations: [batch, num_sensors, 2] as (x, y) grid indices
            timestep: [batch]
        Output:
            [batch, x, y, out_channels]
        """
        B, S, N, C = condition.shape
        # B, X, Y, C = input_image
        # B, N, 2    = locations.shape
        # B,         = timestep.shape
        assert(input_image.shape[0] == condition.shape[0] == locations.shape[0] == timestep.shape[0])
        assert(condition.shape[1:] == (self.seq_len, self.num_sensors, self.in_channels))
        assert(locations.shape[1:] == (self.num_sensors, 2))
        assert(input_image.shape[1:] == (*self.output_size, self.out_channels))

        time_token  = self.time_embedding(self.time_proj(timestep)).unsqueeze(1)
        sensor_toks = self.tokenizer.sensor(condition, locations, self.tokenizer_mode)
        queries     = self.tokenizer.queries(B)
        image_toks  = self.encoder(input_image.permute(0, 3, 1, 2)).flatten(1, 2) + queries
    
        raw_tokens  = torch.cat((time_token, sensor_toks, image_toks), dim = 1)

        tokens      = self.transformer.forward(src = raw_tokens)[:, -image_toks.shape[1]:]

        output_image = self.decoder(tokens).permute(0, 2, 3, 1)

        return output_image

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

        noisy       = self.ddpm_scheduler.add_noise(target, noise, timesteps) # x_t = α·x₀ + σ·ε   (model input)

        v_target    = self.ddpm_scheduler.get_velocity(target, noise, timesteps) # v = α·ε − σ·x₀   (training target)

        with torch.autocast(str(self.device), dtype=torch.bfloat16): 
            v_pred = self._model_pass(noisy, condition, locations,timesteps)
        
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
        original_noise = torch.randn(batch_size, self.output_size[0], self.output_size[1], self.out_channels, device=self.device, generator=generator)
        x = original_noise

        for t in tqdm.tqdm(self.ddim_scheduler.timesteps, colour="red", desc="Sampling", leave=False):

            with torch.autocast(str(self.device), dtype=torch.bfloat16): 
                v_pred = self._model_pass(x, condition, locations,t.repeat(x.shape[0]))
            
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
    

        