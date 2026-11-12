
"""
The defaults behind every .config() call of the high level api (pipeline.py). Every field has a default, .config(name=value) changes one.

Globals:
    - LOSS_TERMS: the use_* loss term names
    - COMPILE_MODES: the torch.compile modes
"""

from dataclasses import dataclass, field

LOSS_TERMS:     tuple = ("use_mse", "use_l1", "use_huber", "use_var", "use_fluc", "use_smooth", "use_spectral")
COMPILE_MODES:  tuple = ("default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs")

@dataclass
class DataConfig:
    """
    Options of Dataset2D.config: which h5 file, which variables and how the file is cut down. The file is stored time first, (t, a, b) per variable.
    crop / downsample / transpose describe the final [time, x, y] field, x = a and y = b unless transpose.

    Fields:
        path: the h5 file
        variables: the datasets to read, one channel each
        crop: ((x0, x1), (y0, y1)) keeps a window, either side can be None
        downsample: keeps every n-th point, an int or (nx, ny)
        cut: (t0, t1) or (t0, t1, step) keeps a range of timesteps
        plane: {dim: index} picks one plane of a file with more dims, e.g. {1: 43} for (t, z, y, x)
        transpose: swaps the two spatial dims, x = b and y = a
    """
    path:               str             | None = None
    variables:          tuple[str, ...] | None = None
    crop:               tuple           | None = None               #? ((x0, x1), (y0, y1))
    downsample:         int             | tuple = 1
    cut:                tuple           | None = None               #? (t0, t1) or (t0, t1, step)
    plane:              dict            | None = None               #? {dim as stored: index}
    transpose:          bool                   = False

@dataclass
class PipeConfig:
    """
    Options of Pipe.config: dataloaders, optimiser, schedule, stopping and outputs.

    Fields:
        split: (train, val, test) fractions of the timesteps, in time order
        batch_size: windows per batch
        epochs: maximum number of epochs, also the length of the cosine schedule
        lr / weight_decay: AdamW learning rate and weight decay
        patience: epochs without improvement before the early stop
        grad_tracking_frequency: batches between per-loss gradient snapshots
        use_auto_weighting: weights the loss terms from their gradients, updated every grad_tracking_frequency batches
        use_lr_warmup / num_warmup_steps: cosine schedule with warm-up and the epochs it warms up for
        speedy: reduced tracking of metrics
        device: None picks cuda > mps > cpu
        output_root: folder the run folder {model}_{time} is written into
    """
    split:                      tuple           = (0.7, 0.2, 0.1)   #? train, val, test in time order
    batch_size:                 int             = 256
    epochs:                     int             = 1000
    lr:                         float           = 1e-3
    weight_decay:               float           = 1e-4
    patience:                   int             = 150
    grad_tracking_frequency:    int             = 50                #? batches between per-loss gradient snapshots
    use_auto_weighting:         bool            = False
    use_lr_warmup:              bool            = False
    num_warmup_steps:           int             = 20
    speedy:                     bool            = True
    device:                     str | None      = None              #? None picks cuda > mps > cpu
    output_root:                str             = "outputs"

@dataclass
class ModelSetup:
    """
    Options of Model.config that every model shares. The model's own hyperparameters are not here, they are the defaults of its __init__
    and Model.config takes them by name.

    Fields:
        model: a MODELS name (pipeline.py), Model.custom swaps in your own
        sequence: timesteps in the sensor window
        sensors: how many, or [[x, y], ...] custom sensor points
        indexing: sensor layout, one of "midline", "top", "bot", "rand", "rand_const"
        in_channels: the variables the sensors read (default: every Dataset2D variable)
        out_channels: what the model outputs (default: in_channels). An out channel that is not a Dataset2D variable is an empty channel
        compile: False, True (max-autotune-no-cudagraphs) or a torch.compile mode from COMPILE_MODES
        use_*: the loss terms, bool or weight (True = 1.0), same meaning as loss_flags in losses/losses.py
        extra_losses: {name: fn} or {name: (fn, weight)} with fn(output, target) -> scalar tensor, both [batch, x, y, channels] in out_channels order
        extra_space: "physical" hands extra losses denormalised fields, "normalised" hands them what the model returns
    """
    model:              str                     = "SHRED"           #? a MODELS name, Model.custom swaps in your own
    sequence:           int                     = 50
    sensors:            int | list              = 3                 #? how many, or [[x, y], ...] custom sensor points
    indexing:           str                     = "rand_const"      #? ["midline", "top", "bot", "rand", "rand_const"]
    in_channels:        tuple[str, ...] | None  = None
    out_channels:       tuple[str, ...] | None  = None
    compile:            bool | str              = True              #? False, True (max-autotune-no-cudagraphs) or one of COMPILE_MODES

    # --- losses ---
    use_mse:            bool | float            = True
    use_l1:             bool | float            = False
    use_huber:          bool | float            = False
    use_var:            bool | float            = False
    use_fluc:           bool | float            = False
    use_smooth:         bool | float            = False
    use_spectral:       bool | float            = False
    extra_losses:       dict                    = field(default_factory=dict)   #? {name: fn(output, target)} or {name: (fn, weight)}
    extra_space:        str                     = "physical"        #? ["physical", "normalised"]

    def flags(self) -> dict:
        """Output: {name: value} of every use_* loss term in LOSS_TERMS."""
        return {name: getattr(self, name) for name in LOSS_TERMS}
