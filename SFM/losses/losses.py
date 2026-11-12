
"""
The shared training objective, the default every model's .losses hands off to (ModelClass, models/abstract.py).

Globals:
    - debug: when True, prints the loss names / values and turns on autograd anomaly detection
    - loss_flags: the default on / off switch for each term, set by Pipe.fit from Model.config (pipeline.py) and read at call time
"""

import torch
import torch.nn.functional as F

# --- Developer Debug Flags ---
debug = False
torch.autograd.set_detect_anomaly(debug)

loss_flags = dict(use_mse             = False,
                  use_l1              = False,
                  use_huber           = False,
                  use_var             = False,
                  use_fluc            = False,
                  use_smooth          = False,
                  use_spectral        = False)

def losses(output: torch.Tensor,
           target: torch.Tensor,
           use_mse:             bool | float | None = None, #? True = 1.0, False / 0.0 skips, a number weights the term
           use_l1:              bool | float | None = None,
           use_huber:           bool | float | None = None,
           use_var:             bool | float | None = None,
           use_fluc:            bool | float | None = None,
           use_smooth:          bool | float | None = None,
           use_spectral:        bool | float | None = None,
           extra_losses: dict = {}, # {"name" : value}
           ) -> tuple[list[torch.Tensor], list[str]]:
    """
    Builds the list of enabled loss terms (per channel MSE, L1, Huber, Var, Fluctuations, Smooth, Spectral) plus any extra_losses.

    Input:
        output / target: both [batch, x, y, channels]
        use_*: per term override, bool or weight (True = 1.0, False / 0.0 skips), None reads loss_flags
        extra_losses: {name: scalar tensor}, appended as they are
    Output:
        (list of weighted scalar loss tensors, list of matching names)
    """

    # read loss_flags at call time
    use_mse             = loss_flags["use_mse"]             if use_mse             is None else use_mse
    use_l1              = loss_flags["use_l1"]              if use_l1              is None else use_l1
    use_huber           = loss_flags["use_huber"]           if use_huber           is None else use_huber
    use_var             = loss_flags["use_var"]             if use_var             is None else use_var
    use_fluc            = loss_flags["use_fluc"]            if use_fluc            is None else use_fluc
    use_smooth          = loss_flags["use_smooth"]          if use_smooth          is None else use_smooth
    use_spectral        = loss_flags["use_spectral"]        if use_spectral        is None else use_spectral

    # [batch, x, y, channels]
    assert(output.shape == target.shape), f"[losses]: target and output must have same shape got target.shape = {target.shape} | output.shape = {output.shape}"

    losses: list[torch.Tensor] = []
    names:  list[str]          = []

    # bool or number, True -> 1.0, anything == 0.0 is skipped
    if use_mse:
        for i, (o, t) in enumerate(zip(torch.unbind(output, dim=-1), torch.unbind(target, dim=-1))):
            losses.append(float(use_mse) * F.mse_loss(o, t))
            names.append(f"MSE_channel_{i}")

    if use_l1:
        losses.append(float(use_l1) * F.l1_loss(output, target))
        names.append("L1")

    if use_huber:
        losses.append(float(use_huber) * F.huber_loss(output, target))
        names.append("Huber")

    if use_var:
        losses.append(float(use_var) * F.mse_loss(output.var(dim=0), target.var(dim=0)))
        names.append("Var")

    if use_fluc:
        losses.append(float(use_fluc) * F.mse_loss(output - output.mean(dim=0), target - target.mean(dim=0)))
        names.append("Fluctuations")

    if use_smooth:
        losses.append(float(use_smooth) * ((torch.diff(output, dim=-3)**2).mean() + (torch.diff(output, dim=-2)**2).mean()))
        names.append("Smooth")

    if use_spectral:
        losses.append(float(use_spectral) * F.mse_loss(_psd_spectral(output), _psd_spectral(target)))
        names.append("Spectral")

    for key in extra_losses.keys():
        assert(isinstance(extra_losses[key], torch.Tensor))
        losses.append(extra_losses[key])
        names.append(key)

    if debug: print(f"[losses]: {names} = {losses}")

    return losses, names

def _psd_spectral(
    model_output: torch.Tensor,
    frequency: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Welch PSD of the fluctuations (batch mean removed) with the batch axis as time, used by the Spectral term.

    Input:
        model_output: [batch, x, y, channels]
        frequency: sampling frequency
    Output:
        psd: [win_length // 2 + 1], the mean over every grid point and channel
    """
    b, x, y, c = model_output.shape

    x = model_output - model_output.mean(dim=0)
    signals = x.flatten(start_dim=1)
    _, pxx = _welch(signals, frequency)
    return pxx

def _welch(
    signals:              torch.Tensor,
    frequency:            int,
    win_length_devisor:   int = 4
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """
    One sided Welch PSD (hann window, half overlapping segments) of every signal, averaged over the signals.

    Input:
        signals: [time, n_signals]
        frequency: sampling frequency
        win_length_devisor: win_length = time // win_length_devisor
    Output:
        (freq, psd): both [win_length // 2 + 1]
    """
    assert(signals.ndim == 2)

    win_length      = max(signals.shape[0]//win_length_devisor, 2)
    freq            = torch.fft.rfftfreq(win_length, d=1/frequency).to(signals.device)
    window          = torch.hann_window(win_length).to(signals.device)
    window_power    = (window ** 2).sum()

    power = torch.stft(signals.T, n_fft=win_length, hop_length=win_length // 2, win_length=win_length, window=window, return_complex=True, center=False)
    psd = (power.abs() ** 2).mean(dim=-1) / (frequency * window_power)
    psd = psd.mean(dim=0)

    psd[1:-1] = psd[1:-1] * 2
    return freq, psd
