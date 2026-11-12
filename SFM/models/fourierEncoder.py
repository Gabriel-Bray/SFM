"""
fourierEncoder: embeds low dimensional features (e.g. sensor coordinates) with sin / cos fourier features followed by an MLP.

Globals:
    - debug: when True, checks the inputs on every encode call (not just the first) and turns on autograd anomaly detection
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Developer Debug Flags ---
debug = False
torch.autograd.set_detect_anomaly(debug)

class fourierEncoder(nn.Module):
    """Fourier feature embedding: features -> (raw, sin, cos at every frequency) -> two layer SiLU MLP."""
    def __init__(self, features: int, embed_dim: int, num_frequencies: int = 8, norm_to_max: float = None, mode: str = "add"):
        """
        Builds the frequency buffer pi * 2^k for k < num_frequencies, the two layer SiLU MLP and the two linear decoders passthrough uses.

        Input:
            features: number of input features, e.g. 2 for (x, y)
            embed_dim: width of the embedding
            num_frequencies: frequencies per feature
            norm_to_max: optional value to divide the inputs by
            mode: "add" or "split", only checked ("add" needs embed_dim divisible by features)
        """
        super().__init__()
        assert(mode in ["split", "add"] and (embed_dim%features == 0 or mode != "add") )

        self.features = features
        self.embed_dim = embed_dim
        self.norm_to_max = norm_to_max
        self.encoded_dim = features + 2 * features * num_frequencies
        self._init_checks()

        self.register_buffer("freqs", math.pi * 2.0 ** torch.arange(num_frequencies, dtype=torch.float32), persistent=False)

        self.feats_decoder = nn.Linear(embed_dim, features)
        self.orig_decoder = nn.Linear(embed_dim, embed_dim)

        self.mlp: nn.Sequential = nn.Sequential(
            nn.Linear(self.encoded_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )

        self.firstpass = True

    def _init_checks(self):
        """Asserts features and embed_dim are non-zero ints and norm_to_max is a non-zero number or None."""
        assert(isinstance(self.features, int) and self.features != 0)
        assert(isinstance(self.embed_dim, int) and self.embed_dim != 0)
        assert(isinstance(self.norm_to_max, (int, float, type(None))) and self.norm_to_max != 0.0)

    def _encode_checks(self, features: torch.Tensor):
        """
        Asserts the inputs are not flat, have the right feature count, are non-negative and sit under norm_to_max.

        Input:
            features: [..., features]
        """
        assert(features.ndim != 1), f"[fourierEncoder]: Please unflatten features! Shape: {features.shape}"
        assert(features.shape[-1] == self.features)
        assert((features >= 0.0).all()), f"Got min: {features.min()}, features shape: {features.shape}"
        if self.norm_to_max:
            assert((features <= self.norm_to_max).all()), f"[fourierEncoder]: specified norm_to_max as {self.norm_to_max}, got value larger in tensor {features.max()}"        

    def encode(self, features: torch.Tensor):
        """
        Concatenates the raw features (divided by norm_to_max when given) with sin / cos of every feature at every frequency.
        The inputs are checked on the first call, and on every call when debug.

        Input:
            features: [..., features]
        Output:
            [..., features + 2 * features * num_frequencies]
        """
        # features : [..., f]
        if debug or self.firstpass: 
            self._encode_checks(features)
            self.firstpass = False

        features_flat = features.flatten(0, -2)

        if self.norm_to_max is not None:
            features_flat = features_flat / self.norm_to_max

        angles = features_flat.unsqueeze(dim=-1) * self.freqs
        encoded = torch.cat([features_flat, torch.sin(angles).flatten(1), torch.cos(angles).flatten(1)], dim=-1)
        
        return encoded.view(*features.shape[:-1], self.encoded_dim)

    @staticmethod
    def _encode(features: torch.Tensor, norm_to_max: float | None = None, freqs: int = 8):
        """
        encode without a module, the same fourier features from explicit arguments.

        Input:
            features: [..., f]
            norm_to_max: optional value to divide the inputs by
            freqs: number of frequencies
        Output:
            [..., f + 2 * f * freqs]
        """

        features_flat = features.flatten(0, -2)

        if norm_to_max is not None:
            features_flat = features_flat / norm_to_max

        freq_bands = math.pi * 2.0 ** torch.arange(freqs, dtype=torch.float32, device=features.device)
        angles = features_flat.unsqueeze(dim=-1) * freq_bands
        
        encoded =  torch.cat([features_flat, torch.sin(angles).flatten(1), torch.cos(angles).flatten(1)], dim=-1)
        
        return encoded.view(*features.shape[:-1], -1)   # [..., f + 2 * f * freqs]

    def forward(self, features: torch.Tensor):
        """
        Encodes the features and passes them through the MLP.

        Input:
            features: [..., features]
        Output:
            [..., embed_dim]
        """

        # sinusoidal_feats : [..., encoded_dim]
        sinusoidal_feats_flat = self.encode(features)
 
        encoded_flat = self.mlp(sinusoidal_feats_flat)

        return encoded_flat.view(*features.shape[:-1], self.embed_dim)

    def passthrough(self, input_tensor: torch.Tensor, features: torch.Tensor):
        """
        Adds the embedding of features to input_tensor and decodes both back out of the sum, with the loss that keeps them recoverable.

        Input:
            input_tensor: [..., embed_dim]
            features: [..., features]
        Output:
            (output_tensor [..., embed_dim], pred_features [..., features], pred_input_tensor [..., embed_dim], loss)
            loss = summed squared error of both reconstructions + mean square of the embedding
        """
        assert(input_tensor.shape == features.shape[:-1] + (self.embed_dim,))
        
        embedding       = self.forward(features)
        output_tensor   = embedding + input_tensor

        pred_features   = self.feats_decoder(output_tensor.flatten(1, -2)).view_as(features)*self.norm_to_max
        features_loss   = ((features.flatten() - pred_features.flatten())**2).sum()

        pred_input_tensor = self.orig_decoder(output_tensor.flatten(1, -2)).view_as(input_tensor)
        orig_loss       = ((input_tensor.flatten() - pred_input_tensor.flatten())**2).sum()

        # Scale loss forces model to not overexpand scale, forces to minimum nessesary. Never allow scale loss to exceed orig_loss or features_loss
        scale_loss      = (embedding**2).mean()
        
        return output_tensor, pred_features, pred_input_tensor, features_loss + orig_loss + scale_loss
