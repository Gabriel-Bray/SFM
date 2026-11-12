
"""
The high level api, three objects that are each set up with .config(name=value, ...) and run with Pipe.fit. 2-D only for now.

    import SFM

    Dataset  = SFM.Dataset2D()
    Pipeline = SFM.Pipe()
    Model    = SFM.Model("SHRED")

    Dataset.config(path="channel_flow.h5", variables=("u", "v", "p"), crop=((0, 128), (0, 128)), downsample=2)
    Pipeline.config(split=(0.7, 0.2, 0.1), epochs=200, lr=1e-3, weight_decay=1e-4, batch_size=256)
    Model.config(sequence=50, sensors=3, indexing="rand_const", in_channels=("u", "v"), out_channels=("u", "v", "p", "rho"), use_mse=True, hidden_size=256)

    Pipeline.fit(Model, Dataset)
    pred, truth = Pipeline.generate_test_dataset()

Every .config() only changes the names it is given, returns the object and can be called again. The options and defaults are in config.py.

Channels: inside the model the order is in_channels first, then the other out_channels that have data, then the empty ones.

Custom model: subclass ModelClass (models/abstract.py), example.py has a blank one. Its __init__ receives any of output_size, seq_len, sequence_length,
    batch_size, num_sensors, in_channels, out_channels, indexing_method and device (SHARED) that it names, its other arguments are set by Model.config.
    forward(batch) -> (output, target). losses(output, target) -> (list of loss tensors, list of names) defaults to losses/losses.py.

Globals:
    - debug: when True, prints the data / model traffic
    - MODELS: model name -> class, the built in models
    - SHARED: the __init__ arguments the Pipe fills in for every model
"""

import os
import json
import time
import torch
import inspect
from dataclasses import asdict, fields, replace
from SFM.config import *
from SFM.utils.utils import get_time, get_device, generate_test_dataset
from SFM.utils.Dataset import Dataset
from SFM.internal import train, plot_all_tracked
from SFM.losses.losses import loss_flags
from SFM.models.abstract import ModelClass
from SFM.models.shred import SHRED, KutzSHRED
from SFM.models.voronoiCnn import VoronoiCNN
from SFM.models.DeepnindPerciever import PercieverIO
from SFM.models.LosAlamosSensiever import Senseiver
from SFM.models.transformer import EncoderDecoderTransformer, EncoderTransformer
from SFM.models.diffusionTransformer import DiffusionTransformer
from SFM.models.diffusionUNET import DiffusionUNET, DiffusionAbstract

# --- Developer Debug Flags ---
debug = False

MODELS: dict = {
    "SHRED":                        SHRED,
    "KutzSHRED":                    KutzSHRED,
    "VoronoiCNN":                   VoronoiCNN,
    "PercieverIO":                  PercieverIO,
    "Senseiver":                    Senseiver,
    "EncoderDecoderTransformer":    EncoderDecoderTransformer,
    "EncoderTransformer":           EncoderTransformer,
    "DiffusionUNET":                DiffusionUNET,
    "DiffusionTransformer":         DiffusionTransformer,
}

SHARED: tuple = ("output_size", "seq_len", "sequence_length", "batch_size", "num_sensors", "in_channels", "out_channels", "indexing_method", "device")

def _update(cfg, kwargs: dict, who: str):
    """
    Replaces the given fields of a config dataclass.

    Input:
        cfg: a config dataclass (config.py)
        kwargs: {field name: new value}, an unknown name asserts
        who: the name printed by the assert
    Output:
        a new config with the fields replaced
    """
    options = [f.name for f in fields(cfg)]
    unknown = [k for k in kwargs if k not in options]
    assert(len(unknown) == 0), f"[{who}]: unknown config {unknown}, options = {options}"
    return replace(cfg, **kwargs)

def _as_slice(window, step: int = 1) -> slice:
    """
    Input:
        window: None, a slice, (start, stop) or (start, stop, step)
        step: multiplies the step already in window
    Output:
        a slice
    """
    # None | slice | (start, stop) | (start, stop, step) -> slice, step multiplies a step already there
    if window is None: window = slice(None)
    if not isinstance(window, slice): window = slice(*window)
    return slice(window.start, window.stop, step * (window.step or 1))

class Dataset2D:
    """An h5 file cut down to [time, x, y, variables]. The options are DataConfig (config.py)."""
    def __init__(self):
        """Starts from the DataConfig defaults, nothing is read until load."""
        self.cfg     = DataConfig()
        self.dataset = None

    def __repr__(self) -> str:
        return f"Dataset2D({self.cfg})"

    def config(self, **kwargs):
        """
        Changes the given options: path, variables, crop, downsample, cut, plane, transpose.

        Input:
            kwargs: name=value for any DataConfig field (config.py)
        Output:
            self
        """
        self.cfg = _update(self.cfg, kwargs, "Dataset2D")
        return self

    def load(self) -> Dataset:
        """
        Reads and slices the file.

        Output:
            a Dataset (utils/Dataset.py) whose .data is [time, x, y, variables] on the cpu, also kept as self.dataset
        """
        c = self.cfg
        assert(c.path is not None and c.variables is not None), "[Dataset2D]: config needs path and variables"
        if isinstance(c.variables, str): c.variables = (c.variables,)

        dataset = Dataset(c.path)
        missing = [v for v in c.variables if v not in dataset.keys]
        assert(len(missing) == 0), f"[Dataset2D]: variables {missing} are not in {c.path}, keys = {list(dataset.keys)}"

        #! the file is stored time first. planes are picked until (t, a, b) is left, then x = a and y = b unless transpose
        shape   = dataset.file[c.variables[0]].shape
        plane   = dict(c.plane) if c.plane is not None else {}
        spatial = [d for d in range(1, len(shape)) if d not in plane]
        assert(len(spatial) == 2), f"[Dataset2D]: 2-D only for now, {c.variables[0]} is stored as {shape}. pick planes until (t, a, b) is left, plane = {{dim: index}}"

        crop_x, crop_y = c.crop if c.crop is not None else (None, None)
        down_x, down_y = (c.downsample, c.downsample) if isinstance(c.downsample, int) else c.downsample
        idx_x, idx_y   = _as_slice(crop_x, down_x), _as_slice(crop_y, down_y)

        dataset.__advanced_init__(vars_of_interest  = tuple(c.variables),
                                  collapse_dims     = tuple(plane) + tuple(spatial),
                                  collapse_idxs     = tuple(plane.values()) + ((idx_y, idx_x) if c.transpose else (idx_x, idx_y)),
                                  timestep_dim      = 0)

        if c.transpose: dataset.data = dataset.data.transpose(1, 2).contiguous()

        if c.cut is not None:
            dataset.data      = dataset.data[_as_slice(c.cut)]
            dataset.timesteps = dataset.data.shape[0]

        if debug: print(f"[Dataset2D]: {c.path} | stored = {shape} | data = {tuple(dataset.data.shape)}")

        self.dataset = dataset
        return dataset

class Model:
    """One class for every model. Pick the type, then .config sets what its __init__ will be run with. The shared options are ModelSetup (config.py)."""
    def __init__(self, model: str = "SHRED"):
        """
        Input:
            model: a MODELS name
        """
        self.setup          = ModelSetup()
        self.hyper          = {}        # the model's own __init__ arguments that Model.config changed, the rest stay at their defaults
        self.custom_model   = None
        self.net            = None
        self.config(model=model)

    def __repr__(self) -> str:
        return f"Model({self.name}, {self.setup}, hyperparameters = {self.hyperparameters})"

    @property
    def name(self) -> str:
        """Output: the name of the model, its MODELS key or the class name of a custom model."""
        return self.model_class.__name__ if self.custom_model is not None else self.setup.model

    @property
    def model_class(self) -> type:
        """Output: the class that will be built, from MODELS or the custom model."""
        if self.custom_model is None: return MODELS[self.setup.model]
        return self.custom_model if isinstance(self.custom_model, type) else type(self.custom_model)

    @property
    def hyperparameters(self) -> dict:
        """Output: {name: value} of the model's own __init__ arguments, the defaults unless Model.config changed them."""
        # the model's own __init__ arguments and their current values, the defaults unless Model.config changed them
        return self.defaults | self.hyper

    @property
    def defaults(self) -> dict:
        """Output: {name: default} of the model's own __init__ arguments (SHARED left out), empty for a custom model that is already built."""
        if isinstance(self.custom_model, ModelClass): return {} # already built by the user
        params = inspect.signature(self.model_class.__init__).parameters
        return {k: p.default for k, p in params.items() if k not in SHARED + ("self",) and p.kind not in (p.VAR_KEYWORD, p.VAR_POSITIONAL)}

    def config(self, **kwargs):
        """
        Changes the given options. model, sequence, sensors, indexing, in_channels, out_channels, compile, the use_* loss terms, extra_losses and
        extra_space are shared by every model (ModelSetup). Anything else is an argument of that model's own __init__, whose defaults hold until
        changed here. Picking another model resets them.

        Input:
            kwargs: name=value for any ModelSetup field (config.py) or argument of the model's __init__
        Output:
            self
        """
        shared = [f.name for f in fields(ModelSetup)]

        if "model" in kwargs:
            assert(kwargs["model"] in MODELS), f"[Model]: model must be in {list(MODELS)} got {kwargs['model']}, Model.custom takes your own"
            if kwargs["model"] != self.setup.model or self.custom_model is not None:
                self.hyper, self.custom_model = {}, None # a new model starts from its own defaults

        self.setup = replace(self.setup, **{k: v for k, v in kwargs.items() if k in shared})
        self.hyper.update({k: v for k, v in kwargs.items() if k not in shared})

        # everything that is not shared must be an argument of the model's __init__
        params  = inspect.signature(self.model_class.__init__).parameters
        open_kw = any(p.kind == p.VAR_KEYWORD for p in params.values()) and not isinstance(self.custom_model, ModelClass)
        unknown = [k for k in self.hyper if not open_kw and k not in self.defaults]
        if len(unknown) > 0:
            for k in unknown: self.hyper.pop(k)
            raise AssertionError(f"[Model]: {self.name} has no config {unknown}, options = {shared + list(self.defaults)}")

        assert(self.setup.compile in (True, False) or self.setup.compile in COMPILE_MODES), f"[Model]: compile should be True, False or one of {COMPILE_MODES} got {self.setup.compile}"
        assert(self.setup.extra_space in ["physical", "normalised"]), f"[Model]: extra_space should be [physical, normalised] got {self.setup.extra_space}"
        return self

    def custom(self, model):
        """
        The low level api, swaps in your own model. .config then edits its __init__ arguments the same way.

        Input:
            model: a ModelClass subclass, or an instance of one
        Output:
            self
        """
        assert(isinstance(model, ModelClass) or (isinstance(model, type) and issubclass(model, ModelClass))), "[Model]: a custom model must subclass ModelClass (models/abstract.py)"
        self.hyper, self.custom_model = {}, model # its own __init__ defaults, Model.config edits them
        return self

    # --- channel order inside the model ---
    def channels(self, variables: tuple):
        """
        Works out the channel order inside the model: in_channels first, then the other out_channels that have data, then the empty ones.

        Input:
            variables: the Dataset2D variables
        Output:
            None, sets in_channels, out_channels, data_vars, empty_vars, channel_order and supervised (how many leading channels have data)
        """
        in_channels  = tuple(self.setup.in_channels) if self.setup.in_channels is not None else tuple(variables)
        out_channels = tuple(self.setup.out_channels) if self.setup.out_channels is not None else in_channels

        missing = [v for v in in_channels if v not in variables]
        assert(len(missing) == 0), f"[Model]: in_channels {missing} are not Dataset2D variables {tuple(variables)}, the sensors need data"
        #! the models slice the sensors off the leading channels of the field, so every in channel is also an out channel for now
        assert(all(v in out_channels for v in in_channels)), f"[Model]: every in channel must also be an out channel, got in_channels = {in_channels} | out_channels = {out_channels}"

        self.in_channels    = in_channels
        self.out_channels   = out_channels
        self.data_vars      = in_channels + tuple(v for v in out_channels if v not in in_channels and v in variables)
        self.empty_vars     = tuple(v for v in out_channels if v not in variables)
        self.channel_order  = self.data_vars + self.empty_vars
        self.supervised     = len(self.data_vars)   # leading channels that have data

    # --- instanciate model ---
    def build(self, grid: tuple, batch_size: int, device: torch.device) -> torch.nn.Module:
        """
        Builds the torch model from the SHARED arguments its __init__ names plus the hyperparameters, sets custom sensor points, moves it to
        the device and compiles it.

        Input:
            grid: (x, y) of one snapshot
            batch_size: handed to the models that take it
            device: the device the model is moved to
        Output:
            the model (compiled unless compile is off or the device is mps), also kept as self.net
        """
        s = self.setup

        points          = None if isinstance(s.sensors, int) else s.sensors
        indexing_method = "manual" if points is not None else s.indexing
        num_sensors     = len(points) if points is not None else s.sensors
        assert(indexing_method != "manual" or points is not None), "[Model]: manual indexing needs the points, sensors = [[x, y], ...]"

        shared = dict(output_size       = grid,
                      seq_len           = s.sequence,
                      sequence_length   = s.sequence,
                      batch_size        = batch_size,
                      num_sensors       = num_sensors,
                      in_channels       = len(self.in_channels),
                      out_channels      = len(self.channel_order),
                      indexing_method   = indexing_method,
                      device            = device)

        if isinstance(self.custom_model, ModelClass):
            net = self.custom_model # already built by the user

        else:
            # each model only takes the shared arguments its __init__ names
            accepted = inspect.signature(self.model_class.__init__).parameters
            if not any(a.kind == a.VAR_KEYWORD for a in accepted.values()):
                shared = {k: v for k, v in shared.items() if k in accepted}

            if debug: print(f"[Model]: {self.name} shared = {shared} | config = {self.hyper}")
            net = self.model_class(**shared, **self.hyper)

        assert(len(self.empty_vars) == 0 or not isinstance(net, DiffusionAbstract)), "[Model]: diffusion models noise the whole target, empty channels are only for regressive models"

        if points is not None: net.set_sensor_points(points, grid=grid)

        net = net.to(device)
        self.param_count = sum(p.numel() for p in net.parameters())

        compile_mode = "max-autotune-no-cudagraphs" if s.compile is True else s.compile
        if compile_mode and device.type != "mps":
            import torch._functorch.config
            torch._functorch.config.donated_buffer = False
            torch.backends.cudnn.benchmark = True
            net = torch.compile(net, mode=compile_mode)

        self.net = net
        print(f"[Model]: {self.name} | grid = {grid} | channels = {self.channel_order} | empty = {self.empty_vars} | sensors = {num_sensors} ({indexing_method}) | compile = {compile_mode} | params = {self.param_count/1e6:.2f}M")
        return net

    def losses(self, output: torch.Tensor, target: torch.Tensor):
        """
        The built in terms on the channels that have data, then the extra losses on every channel. Handed to train as loss_fn.

        Input:
            output / target: both [batch, x, y, channels] in the model's channel order
        Output:
            (list of scalar loss tensors, list of matching names)
        """
        k = self.supervised
        loss_list, names = self.net.losses(output[..., :k], target[..., :k]) if k < output.shape[-1] else self.net.losses(output, target)

        if len(self.setup.extra_losses) > 0:
            # in training the model returns normalised fields, eval is already physical
            if self.setup.extra_space == "physical" and self.net.training and getattr(self.net, "normalize_outputs", False):
                output = self.net.denormalise(output, self.net.output_mean, self.net.output_std, dim=-1)
                target = self.net.denormalise(target, self.net.output_mean, self.net.output_std, dim=-1)

            # extra losses see the channels the way out_channels lists them
            if self.channel_order != self.out_channels:
                order = [self.channel_order.index(v) for v in self.out_channels]
                output, target = output[..., order], target[..., order]

            for name, fn in self.setup.extra_losses.items():
                fn, weight = fn if isinstance(fn, (tuple, list)) else (fn, 1.0)
                loss_list.append(float(weight) * fn(output, target))
                names.append(name)

        return loss_list, names

class Pipe:
    """Loads the data, builds the model, trains, saves and generates the test dataset. The options are PipeConfig (config.py)."""
    def __init__(self):
        """Starts from the PipeConfig defaults, the model, dataset and stats are set by prepare / fit."""
        self.cfg     = PipeConfig()
        self.model   = None
        self.dataset = None
        self.stats   = None

    def __repr__(self) -> str:
        return f"Pipe({self.cfg})"

    def config(self, **kwargs):
        """
        Changes the given options: split, batch_size, epochs, lr, weight_decay, patience, device, output_root, ...

        Input:
            kwargs: name=value for any PipeConfig field (config.py)
        Output:
            self
        """
        self.cfg = _update(self.cfg, kwargs, "Pipe")
        return self

    def prepare(self, model: Model, dataset: Dataset2D):
        """
        Loads the dataset into loaders and builds the model, nothing is trained or written.

        Input:
            model: the Model
            dataset: the Dataset2D
        Output:
            self, with the trainloader / valloader / testloader, net, grid, device and output paths set
        """
        assert(isinstance(model, Model) and isinstance(dataset, Dataset2D)), "[Pipe]: takes (Model, Dataset2D)"
        c = self.cfg
        assert(any(model.setup.flags().values()) or len(model.setup.extra_losses) > 0), "[Pipe]: every loss is off, turn one on in Model.config"

        self.model, self.dataset, self.stats = model, dataset, None

        # --- Enviroment ---
        self.time       = get_time()
        self.device     = torch.device(c.device) if c.device is not None else get_device()
        self.root_path  = os.path.join(c.output_root, f"{model.name}_{self.time}")
        self.paths      = {name: os.path.join(self.root_path, name) for name in ["model", "tracking"]}

        # --- load data -> builds trainloader, valloader, testloader ---
        data = dataset.load()
        model.channels(dataset.cfg.variables)

        data.data = data.data[..., [dataset.cfg.variables.index(v) for v in model.data_vars]]
        if len(model.empty_vars) > 0: # no data behind these, zeros hold the place in the target
            data.data = torch.cat([data.data, torch.zeros(*data.data.shape[:-1], len(model.empty_vars))], dim=-1)

        data.data = data.data.to(self.device)  #* pre-move data to train device to remove transfer bottleneck.
        self.grid = tuple(data.data.shape[1:3]) # (x, y) of one snapshot
        print(f"[Pipe]: data = {dataset.cfg.path} | shape = {tuple(data.data.shape)} | device = {self.device}")

        self.trainloader, self.valloader, self.testloader = data.build_dataloaders(c.split, c.batch_size, model.setup.sequence)

        # --- instanciate model ---
        self.net = model.build(self.grid, c.batch_size, self.device)

        # --- model-specific-loaders ---
        if hasattr(self.net, "custom_loaders"):
            # Dont shoot yourself in the foot. Use traindata.
            train_data = data.data[:int(c.split[0]*data.data.shape[0])]
            val_data   = data.data[int(c.split[0]*data.data.shape[0]):int((c.split[0]+c.split[1])*data.data.shape[0])]
            test_data  = data.data[int((c.split[0]+c.split[1])*data.data.shape[0]):]
            self.trainloader, self.valloader, self.testloader = self.net.custom_loaders(train_data, val_data, test_data)

        return self

    @property
    def sensor_points(self):
        """Output: the cached [num_sensors, 2] sensor points, None when they are resampled every batch or not sampled yet."""
        x_idx, y_idx = self.net.cached["x_idx"], self.net.cached["y_idx"]
        if x_idx is None or y_idx is None: return None
        return torch.stack([x_idx, y_idx], dim=-1)

    @property
    def full_config(self) -> dict:
        """Output: {"dataset", "pipe", "model", "run"}, every option of the run, what save writes to config.json."""
        m = self.model
        return dict(dataset = asdict(self.dataset.cfg),
                    pipe    = asdict(self.cfg),
                    model   = {f.name: getattr(m.setup, f.name) for f in fields(m.setup)} | {"model": m.name, "extra_losses": list(m.setup.extra_losses), "hyperparameters": m.hyperparameters},
                    run     = dict(channels = m.channel_order, empty_vars = m.empty_vars, grid = self.grid, param_count = m.param_count))

    def fit(self, model: Model, dataset: Dataset2D):
        """
        prepare, train (internal.py), save. Outputs land in {output_root}/{model}_{time}/ as model and tracking folders plus config.json.

        Input:
            model: the Model
            dataset: the Dataset2D
        Output:
            self, self.stats holds the ModelStats of the run
        """
        self.prepare(model, dataset)
        for path in self.paths.values(): os.makedirs(path, exist_ok=True)
        c = self.cfg
        loss_flags.update(model.setup.flags())

        # --- train ---
        start_time = time.perf_counter()

        self.net, self.stats = train(self.net,
                                     self.trainloader,
                                     self.valloader,
                                     c.lr,
                                     c.weight_decay,
                                     c.epochs,
                                     c.patience,
                                     self.device,
                                     c.grad_tracking_frequency,
                                     use_auto_weighting=c.use_auto_weighting,
                                     use_lr_warmup=c.use_lr_warmup,
                                     num_warmup_steps=c.num_warmup_steps,
                                     checkpoint_path=os.path.join(self.paths["model"], "checkpoint.pth"),
                                     stagepoint_root=self.paths["model"],
                                     speedy=c.speedy,
                                     loss_fn=model.losses)

        train_time = time.perf_counter() - start_time # (seconds)
        print(f"[Pipe]: training time - {train_time/60:.2f} minutes")

        self.stats.update("param_count", model.param_count)
        self.stats.update("training_time", train_time)
        if self.sensor_points is not None: self.stats.update("sensor_pts", self.sensor_points)

        self.save()
        return self

    def save(self):
        """Writes the model, the stats (once trained) and config.json into the run folder."""
        assert(self.model is not None), "[Pipe]: nothing to save yet, call fit or prepare first"
        for path in self.paths.values(): os.makedirs(path, exist_ok=True)

        self.net.save(self.paths["model"])
        if self.stats is not None: self.stats.save(self.paths["tracking"])

        with open(os.path.join(self.root_path, "config.json"), "w") as f:
            json.dump(self.full_config, f, indent=4, default=str)

        print(f"[Pipe]: saved to {self.root_path}")

    def plot_stats(self):
        """Plots every tracked training quantity into the tracking folder (plot_all_tracked, internal.py)."""
        assert(self.stats is not None), "[Pipe]: nothing tracked yet, call fit first"
        plot_all_tracked(self.stats.history, self.paths["tracking"])

    def generate_test_dataset(self, save: bool = True):
        """
        Reconstructs the test split.

        Input:
            save: writes {variable}_pred / {variable}_true to model/test_data.h5
        Output:
            (pred, truth): both [timesteps, x, y, channels] with the channels in out_channels order, truth is nan on empty channels
        """
        assert(self.model is not None), "[Pipe]: no model yet, call fit or prepare first"
        m = self.model
        pred, truth = generate_test_dataset(self.net, self.testloader, self.device, autocat=True)[:2]

        # model channel order -> out_channels order
        order = [m.channel_order.index(v) for v in m.out_channels]
        pred, truth = pred[..., order], truth[..., order]

        empty = [i for i, v in enumerate(m.out_channels) if v in m.empty_vars]
        truth[..., empty] = float("nan") # no data behind an empty channel

        if save:
            os.makedirs(self.paths["model"], exist_ok=True)
            data_dict = {}
            for c, name in enumerate(m.out_channels):
                data_dict[f"{name}_pred"] = pred[..., c]
                if name not in m.empty_vars: data_dict[f"{name}_true"] = truth[..., c]

            Dataset.save(os.path.join(self.paths["model"], "test_data.h5"), data_dict)

        return pred, truth
