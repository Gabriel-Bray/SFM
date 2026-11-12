
"""
example: the high level api, then the low level api (a blank custom model and an extra loss) swapped into the same run.
set path / variables to a real h5 file, stored time first as (t, x, y) per variable.
"""

import torch
import SFM
from SFM.losses.losses import losses
from SFM.utils.utils import auto_self_assign

# --- High Level ---
Dataset  = SFM.Dataset2D()
Pipeline = SFM.Pipe()
Model    = SFM.Model("SHRED")

Dataset.config(path="data.h5", variables=("u", "v"), crop=((None, None), (None, None)), downsample=1)
Pipeline.config(split=(0.7, 0.2, 0.1), epochs=100, lr=1e-3, weight_decay=1e-4, batch_size=256)
Model.config(sequence=50, sensors=3, indexing="rand_const", in_channels=("u", "v"), out_channels=("u", "v", "p"), use_mse=True, compile=True)

#? the model's own __init__ arguments are edited the same way, print(Model) lists them
# Model.config(hidden_size=128, decoder_layer_sizes=[256, 1024])

#? custom sensor points instead of a count
# Model.config(sensors=[[10, 20], [40, 20], [70, 20]])

# --- Low Level ---
#! blank model: subclass ModelClass, name the shared arguments you need in __init__ and return (output, target) from forward
@auto_self_assign
class BlankModel(SFM.ModelClass):
    def __init__(self,
                 output_size:       tuple[int, int],    # (x, y) of one snapshot
                 seq_len:           int,
                 num_sensors:       int,
                 in_channels:       int,                # channels the sensors read
                 out_channels:      int,                # channels of the field, empty ones included
                 indexing_method:   str,
                 hidden:            int = 256,          # your own arguments need defaults, Model.config(hidden=...) edits them
                 ):

        super().__init__()

        self.net = torch.nn.Sequential(
            torch.nn.Linear(seq_len * num_sensors * in_channels, hidden),
            torch.nn.GELU(),
            torch.nn.Linear(hidden, output_size[0] * output_size[1] * out_channels),
        )

    def forward(self, batch: torch.Tensor):
        # batch [batch, seq_len, x, y, out_channels], index samples the sensors off the leading in_channels
        sampled      = self.index(batch, self.num_sensors, self.indexing_method, True, channel_slice=slice(0, self.in_channels, None))
        sensors_vals = sampled["sensors_vals"]   # [batch, seq_len, num_sensors, in_channels]
        sensors_locs = sampled["sensor_locs"]    # [batch, num_sensors, 2]
        target       = sampled["target"]         # [batch, x, y, out_channels]

        out = self.net(sensors_vals.flatten(1))
        return self.standard_output(out, self.output_size, self.out_channels), target

    def losses(self, output: torch.Tensor, target: torch.Tensor):
        return losses(output, target) # the built in terms switched on in Model.config

#! blank loss: (output, target) -> scalar tensor, both [batch, x, y, channels] with the channels in out_channels order
def divergence(output: torch.Tensor, target: torch.Tensor):
    u, v = output[..., 0], output[..., 1]
    return ((torch.diff(u, dim=1)[:, :, :-1] + torch.diff(v, dim=2)[:, :-1, :])**2).mean()

if __name__ == "__main__":
    # Model.custom(model=BlankModel)
    # Model.config(hidden=512, extra_losses={"divergence": (divergence, 0.1)})

    Pipeline.fit(Model, Dataset)
    pred, truth = Pipeline.generate_test_dataset() # [timesteps, x, y, channels] in out_channels order
    Pipeline.plot_stats()
