
"""
Every plot function used in this folder, all drawn under the shared PLOT_STYLE.

Globals:
    - PLOT_STYLE: matplotlib rc applied to every function, latex text is used only when a latex install is found
"""

import os
import shutil
import torch
import numpy as np
import scipy.signal as signal
from cycler import cycler
import matplotlib.pyplot as plt
from SFM.utils.utils import to_numpy

PLOT_STYLE: dict = {
    "text.usetex":          shutil.which("latex") is not None,
    "font.family":          "serif",
    "font.serif":           ["Computer Modern Roman"],
    "axes.linewidth":       0.7,
    "axes.spines.top":      True,
    "axes.spines.right":    True,
    "xtick.direction":      "in",
    "ytick.direction":      "in",
    "xtick.top":            True,
    "ytick.right":          True,
    "xtick.minor.visible":  True,
    "ytick.minor.visible":  True,
    "xtick.major.width":    0.7,
    "ytick.major.width":    0.7,
    "xtick.minor.width":    0.5,
    "ytick.minor.width":    0.5,
    "xtick.major.size":     3.2,
    "ytick.major.size":     3.2,
    "xtick.minor.size":     1.8,
    "ytick.minor.size":     1.8,
    "lines.solid_capstyle": "round",
    "axes.prop_cycle":      cycler(color=["black", "red", "blue", "green", "purple", "gray", "brown"])
}

def spectra(sig, freq: int = 1, overlap: float = 0.5):
    """
    Welch mean PSD of one signal (hamming window, 4 segments at the given overlap, nfft 2**16).

    Input:
        sig: 1d signal
        freq: sampling frequency
        overlap: fraction of segment overlap
    Output:
        (freqs, psd)
    """
    seg = int(len(sig) // (4 - 3 * overlap))   # 4 segments at the given overlap; = N/2.5 for 0.5
    win = signal.windows.hamming(seg, sym=True)
    nfft = 2**16
    
    freqs, mean_spectrum = signal.welch(x=sig, fs=freq,
                                        window=win,
                                        nperseg=seg,
                                        noverlap=int(seg * overlap),
                                        nfft=nfft,
                                        detrend=False,
                                        scaling="density")

    return freqs, mean_spectrum

@plt.rc_context(PLOT_STYLE)
def _generate_imfigure(fig,
                       ax,
                       image,
                       label,
                       vmin                   = None,
                       vmax                   = None,
                       x_label                = None,
                       y_label                = None,
                       colorbar               = False,
                       extent                 = None,
                       fontsize               = 9,
                       aspect                 = 1,
                       cbar_label             = "",
                       sensor_pts: np.ndarray = None):
            """
            Draws one [x, y] field onto an axis with optional colorbar, limits and sensor points.

            Input:
                fig / ax: the matplotlib figure and axis to draw on
                image: [x, y] array
                label: title of the axis
                vmin / vmax: colour limits, give both or neither
                x_label / y_label: axis labels, give both or neither
                colorbar: adds a colorbar labelled cbar_label
                extent: ((x0, x1), (y0, y1)) arrays the axis limits are taken from, defaults to the image size
                fontsize: fontsize of the title and labels
                aspect: box aspect of the axis, None leaves it free
                sensor_pts: [n, 2] (x, y) points drawn as black dots
            Output:
                None
            """

            #! assumed last two dimentions are [x, y]

            if not None in [vmin, vmax]:
                im = ax.imshow(image.T, cmap='RdBu_r', vmin=vmin, vmax=vmax, aspect = "auto" if aspect is not None else aspect)
            else:
                im = ax.imshow(image.T, cmap='RdBu_r', aspect = "auto" if aspect is not None else aspect)
    
            if not None in [x_label, y_label]:
                ax.set_xlabel(x_label, fontsize = fontsize)
                ax.set_ylabel(y_label, fontsize = fontsize)
    
            if colorbar:
                fig.colorbar(im, ax=ax, label = cbar_label)

            if extent is not None:
                ax.set_xlim(extent[0][0].min(), extent[0][1].max())
                ax.set_ylim(extent[1][0].min(), extent[1][1].max())
            else:
                ax.set_xlim(0, image.shape[0])
                ax.set_ylim(0, image.shape[1])

            if sensor_pts is not None:
                x_coords = sensor_pts[:, 0]
                y_coords = sensor_pts[:, 1]

                ax.scatter(x_coords, y_coords, color='black', marker='o', s=4, label='Probe Locations')

            if aspect is not None:
                ax.set_box_aspect(aspect)
                
            ax.set_title(label, fontsize = fontsize)

def _compute_jpdf(A:     np.ndarray,
                  B:     np.ndarray,
                  bins:  int                      = 100,
                  range: list[list[float]] | None = None):
    """
    2d density histogram of two arrays.

    Input:
        A / B: arrays of the same shape
        bins: bins per side
        range: [[a_min, a_max], [b_min, b_max]], defaults to the data limits
    Output:
        (H [bins, bins], xedges, yedges)
    """
    
    assert A.shape == B.shape

    a = A.flatten()
    b = B.flatten()

    H, xedges, yedges = np.histogram2d(a, b, bins=bins, range=range, density=True)

    return H, xedges, yedges

def specific_spectra_plot(sigs:      list | np.ndarray,
                          labels:    list,
                          title:     str   = "PSD",
                          frequency: int   = 40000,
                          colors:    list  = ["black", "red", "blue"],
                          axis:      int   = -1,
                          show:      bool  = True,
                          save:      bool  = True,
                          save_path: str   = None,
                          fontsize:  int   = 8,
                          overlap:   float = 0.5,
                          cutoff:    int   = 0,
                          xlim:      float = None,
                          ylim:      float = None):
    """
    Log-log mean PSD of one or more signals with a -5/3 reference line, computed with spectra and drawn with simple_line.

    Input:
        sigs: list of 1d signals, or an array split along axis
        labels: one per signal
        title: title of the figure
        frequency: sampling frequency
        colors: one per signal
        axis: the axis an array is split along
        show / save: shows the figure / saves it to save_path
        fontsize: fontsize of the title and labels
        overlap: fraction of segment overlap, more overlap gives more averages at the same resolution
        cutoff: leading frequency bins left out
        xlim / ylim: left x limit / top y limit, default to two decades past the spectral peak
    Output:
        None
    """

    if isinstance(sigs, np.ndarray):
        sigs = np.unstack(sigs, axis=axis)
    
    assert(len(sigs) <= len(colors))
    if save: assert(save_path is not None)

    freqs, pxxs = zip(*(spectra(sig, frequency, overlap) for sig in sigs))
    ref_y = max(pxx.max() for pxx in pxxs)
    ref_x = max(f[pxx.argmax()] for f, pxx in zip(freqs, pxxs))

    # --- -5/3 line ---
    f_ref = np.arange(ref_x * 2, ref_x * 36, step= ref_x * 2)
    pxx_ref = 5 * ref_y * (f_ref / ref_x) ** (-5 / 3)

    simple_line([pxx[cutoff:] for pxx in pxxs] + [pxx_ref],
                path        = save_path if save else None,
                x           = [f[cutoff:] for f in freqs] + [f_ref],
                labels      = list(labels[:len(sigs)]) + ["$f^{-5/3}$"],
                title       = title,
                label_x     = "$f$",
                label_y     = "$PSD$",
                fontsize    = fontsize,
                logx        = True,
                logy        = True,
                ls          = ["-"] * len(sigs) + ["--"],
                marker      = None,
                linewidth   = 0.9,
                colors      = list(colors[:len(sigs)]) + ["black"],
                xlim        = (ref_x / 100 if xlim is None else xlim, None),
                ylim        = (None, ref_y * 100 if ylim is None else ylim),
                grid        = True,
                figsize     = (5, 5),
                show        = show)

@plt.rc_context(PLOT_STYLE)
def fancy_jpdf(title:       str,
               component_1: torch.Tensor | np.ndarray,
               component_2: torch.Tensor | np.ndarray,
               path:        str,
               xlabel:      str                      = "",
               ylabel:      str                      = "",
               bins:        int                      = 1000,
               range:       list[list[float]] | None = None):
    """
    Joint pdf of two components drawn as a 2d histogram.

    Input:
        title: title of the figure
        component_1 / component_2: tensors or arrays with the same number of points
        path: where the figure is saved
        xlabel / ylabel: axis labels
        bins: bins per side
        range: [[x_min, x_max], [y_min, y_max]], defaults to the data limits
    Output:
        None
    """

    component_1, component_2 = to_numpy(component_1).ravel(), to_numpy(component_2).ravel()
    assert component_1.size == component_2.size, f"[fancy_jpdf]: components need the same number of points, got {component_1.size} and {component_2.size}"

    H, xedges, yedges = _compute_jpdf(component_1, component_2, bins=bins, range=range)
    vmin, vmax = np.percentile(H, [2, 98])

    fig, ax = plt.subplots(layout="constrained")

    mesh = ax.pcolormesh(xedges, yedges, H.T, cmap="RdBu_r", shading="auto", vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.colorbar(mesh, ax=ax, label="PDF")

    fig.savefig(path, dpi=600)
    plt.close()

@plt.rc_context(PLOT_STYLE)
def fancy_pdf(title:         str,
              datas:         list[torch.Tensor | np.ndarray],
              labels:        list[str],
              path:          str,
              xlabel:        str       = "",
              colors:        list[str] = ["black", "red", "blue", "green"],
              bins:          int       = 1000,
              mean_subtract: bool      = True,
              fontsize:      int       = 9):
    """
    Log pdf (step histogram) of the temporal fluctuations of one or more fields, e.g. truth vs prediction.

    Input:
        title: title of the figure
        datas: list of tensors / arrays, time on axis 0
        labels / colors: one per field
        path: where the figure is saved
        xlabel: x axis label
        bins: number of bins
        mean_subtract: removes the temporal mean so the pdf is of the fluctuations
        fontsize: fontsize of the title and labels
    Output:
        None
    """

    # time assumed on axis 0, temporal mean removed so the pdf is of the fluctuations
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize = (3.5, 3.5), layout='constrained')

    for data, label, color in zip(datas, labels, colors):
        data = to_numpy(data)
        if mean_subtract:
            data = data - data.mean(axis=0, keepdims=True)
        data = data.ravel()

        ax.hist(data, bins=bins, color=color, density=True, histtype='step', label=label)

    ax.set_title(title, fontsize = fontsize)
    ax.tick_params(direction="in", axis="both")
    ax.set_yscale("log")
    ax.set_ylabel("Probability", fontsize = 8)
    ax.set_xlabel(xlabel, fontsize = fontsize)
    ax.legend(frameon=False, fontsize = 6)
    fig.savefig(path, dpi = 600, bbox_inches='tight')
    plt.close(fig)

@plt.rc_context(PLOT_STYLE)
def fancy_scatter(title:       str,
                  component_1: torch.Tensor | np.ndarray,
                  component_2: torch.Tensor | np.ndarray,
                  path:        str,
                  xlabel:      str = "",
                  ylabel:      str = "",
                  n_samples:   int = 50000):
    """
    Scatter of a random subsample of two components against each other.

    Input:
        title: title of the figure
        component_1 / component_2: tensors or arrays with the same number of points
        path: where the figure is saved
        xlabel / ylabel: axis labels
        n_samples: points drawn
    Output:
        None
    """
    
    c1, c2 = to_numpy(component_1).ravel(), to_numpy(component_2).ravel()
    assert c1.size == c2.size, f"[fancy_scatter]: components need the same number of points, got {c1.size} and {c2.size}"

    points = np.random.default_rng().choice(len(c1), min(n_samples, len(c1)), replace=False)

    fig, ax = plt.subplots(layout="constrained")

    ax.scatter(c1[points], c2[points], s=1, alpha=1, linewidths=0, color="black", rasterized=True)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    fig.savefig(path, dpi=600)
    plt.close()

@plt.rc_context(PLOT_STYLE)
def fancy_vorticity(title:            str,
                    component_1:      torch.Tensor,
                    component_2:      torch.Tensor,
                    path:             str,
                    return_vorticity: bool                  = False,
                    transpose:        bool                  = False,
                    sensor_pts:       list[tuple[int, int]] = None):
    """
    Computes the vorticity d(component_2)/dx - d(component_1)/dy of one snapshot and draws it.

    Input:
        title: title of the figure
        component_1 / component_2: [x, y] tensors, the x-velocity and the y-velocity
        path: where the figure is saved
        return_vorticity: returns the vorticity instead of None
        transpose: swaps x and y first
        sensor_pts: (x, y) points drawn as black dots
    Output:
        the [x, y] vorticity tensor, or None
    """
    
    # assuming shape [x, y], component_1 is the x-velocity and component_2 the y-velocity
    assert component_1.ndim == 2 and component_2.ndim == 2, f"[fancy_vorticity]: expected 2d [x, y] components, got {tuple(component_1.shape)} and {tuple(component_2.shape)}"

    if transpose:
        component_1 = component_1.transpose(0, 1)
        component_2 = component_2.transpose(0, 1)

    vorticity = torch.gradient(component_2, spacing=1, dim=(0,), edge_order=2)[0] - torch.gradient(component_1, spacing=1, dim=(1,), edge_order=2)[0]
    
    fig, ax = plt.subplots()

    _generate_imfigure(fig, ax, to_numpy(vorticity), title, colorbar=True, sensor_pts = sensor_pts, aspect=None)

    fig.savefig(os.path.join(path), dpi = 600)
    plt.close()
    
    if return_vorticity:
        return vorticity
    else:
        return None

@plt.rc_context(PLOT_STYLE)
def quick_plot(im:        torch.Tensor | np.ndarray,
               show:      bool = False,
               save:      bool = True,
               path:      str  = None,
               transpose: bool = False):
    """
    Fast imshow of a single 2d field with a colorbar.

    Input:
        im: [x, y] tensor or array
        show / save: shows the figure / saves it to path
        path: where the figure is saved, needed when save
        transpose: draws im as given instead of transposed
    Output:
        None
    """

    im = to_numpy(im)
    assert im.ndim == 2, f"[quick_plot]: expected a 2d [x, y] image, got shape = {im.shape}"

    fig, ax = plt.subplots()
    if transpose:
        im = ax.imshow(im, cmap="RdBu_r")
    else:
        im = ax.imshow(im.T, cmap="RdBu_r")

    fig.colorbar(im)
    
    if show:
        plt.show()
    if save:
        assert(path is not None)
        fig.savefig(path)

    plt.close()

@plt.rc_context(PLOT_STYLE)
def simple_line(lines:       np.ndarray | torch.Tensor | list,
                path:        str | None                              = None,
                labels:      str | list[str] | None                  = None,
                title:       str                                     = "",
                label_x:     str                                     = "",
                label_y:     str                                     = "",
                x:           np.ndarray | torch.Tensor | list | None = None,
                fontsize:    int                                     = 8,
                logx:        bool                                    = False,
                logy:        bool                                    = False,
                unbind_axis: int                                     = -1,
                ls:          str | list[str]                         = "-",
                marker:      str | None                              = "o",
                ms:          int                                     = 4,
                linewidth:   float                                   = 0.8,
                colors:      list[str] | None                        = None,
                y_line:      float                                   = None,
                x_line:      float                                   = None,
                xlim:        tuple | None                            = None,
                ylim:        tuple | None                            = None,
                grid:        bool                                    = False,
                figsize:     tuple                                   = (5, 5.5),
                show:        bool                                    = False):
    """
    Plots one or multiple lines, used by the gradient analysis and as the quick line plot. Every option defaults.

    Input:
        lines: 1d signal, 2d array split along unbind_axis, or a list of 1d signals (any length each)
        path: where the figure is saved, None does not save
        labels: one per line, default the line index
        title / label_x / label_y: title and axis labels
        x: None (index), one shared 1d array, or a list with one 1d array per line
        fontsize: fontsize of the title and labels
        logx / logy: log axes
        unbind_axis: the axis a 2d array is split along
        ls: linestyle, one for all or one per line
        marker / ms: marker and its size, None for no marker
        linewidth: width of the lines
        colors: one per line, default from PLOT_STYLE
        y_line / x_line: dashed red reference line at this y / x
        xlim / ylim: axis limits
        grid: major and minor grid
        figsize: size of the figure
        show: shows the figure
    Output:
        None
    """

    # lines: 1d signal, 2d array split along unbind_axis, or a list of 1d signals (any length each)
    if isinstance(lines, (list, tuple)):
        lines = [to_numpy(line) for line in lines]
    else:
        lines = to_numpy(lines)
        if lines.ndim == 1:
            lines = [lines]
        elif lines.ndim == 2:
            lines = list(np.unstack(lines, axis=unbind_axis))
        else:
            raise NotImplementedError(f"[simple_line]: Line plot only supports 1-d or 2-d arrays. got shape = {lines.shape}")

    # x: None (index), one shared 1d array, or a list with one 1d array per line
    if x is None: x = [np.arange(len(line)) for line in lines]
    elif isinstance(x, (list, tuple)): x = [to_numpy(xi) for xi in x]
    else: x = [to_numpy(x)] * len(lines)

    if labels is None: labels = [f"{i}" for i in range(len(lines))]
    elif isinstance(labels, str): labels = [labels]
    if colors is None: colors = [c["color"] for c, _ in zip(plt.rcParams["axes.prop_cycle"]() , range(len(lines)))]
    if isinstance(ls, str): ls = [ls] * len(lines)

    assert len(x) == len(lines), f"[simple_line]: got {len(x)} x arrays for {len(lines)} lines"
    assert len(labels) == len(lines), f"[simple_line]: got {len(labels)} labels for {len(lines)} lines"
    assert len(colors) >= len(lines), f"[simple_line]: got {len(colors)} colors for {len(lines)} lines"
    assert len(ls) == len(lines), f"[simple_line]: got {len(ls)} linestyles for {len(lines)} lines"

    fig, ax = plt.subplots(figsize = figsize, layout='constrained')

    for xi, line, label, color, lsi in zip(x, lines, labels, colors, ls):
        ax.plot(xi, line, ls=lsi, color = color, linewidth=linewidth, label=label, marker=marker, ms=ms, mfc="black")

    if y_line is not None:
        ax.axhline(y_line, color='red', linestyle='--')

    if x_line is not None:
        ax.axvline(x_line, color='red', linestyle='--')

    if logx: ax.set_xscale("log")
    if logy: ax.set_yscale("log")
    if xlim is not None: ax.set_xlim(xlim)
    if ylim is not None: ax.set_ylim(ylim)
    if grid:
        ax.grid(True)
        ax.grid(True, which='minor', linestyle='--', linewidth=0.3)
    ax.set_title(title, fontsize = fontsize)
    ax.set_xlabel(label_x, fontsize = fontsize)
    ax.set_ylabel(label_y, fontsize = fontsize)
    ax.legend(frameon=False, fontsize = 6)
    ax.tick_params(direction='in', which='both')

    if path is not None:
        fig.savefig(path, dpi = 600, bbox_inches='tight')
    if show:
        plt.show()

    plt.close(fig)

@plt.rc_context(PLOT_STYLE)
def quick_multiplot(datas:      np.ndarray | torch.Tensor,
                    x_labels:   list | str | None,
                    y_labels:   list | str | None,
                    titles:     list | str | None,
                    colorbars:  list | bool | None,
                    n_rows:     int,
                    n_cols:     int             = 1,
                    figsize:    tuple[int, int] = None,
                    col_titles: list            = None,
                    row_titles: list            = None,
                    suptitle:   str             = None,
                    mode:       str             = "sig2d",  # ["sigs1d", "sigs2d", "ims"]
                    save_path:  str             = None):
    """
    Grid of images ("ims"), single lines ("sig1d") or line bundles ("sig2d") with per panel labels, row / col titles and a suptitle.

    Input:
        datas: one entry per panel, [n_rows * n_cols, x, y] for "ims", [n_rows * n_cols, t] for "sig1d", [n_rows * n_cols, lines, t] for "sig2d"
        x_labels / y_labels / titles: one per panel, one str for all, or None
        colorbars: one bool per panel, one for all, or None
        n_rows / n_cols: shape of the grid
        figsize: size of the figure
        col_titles / row_titles: one per column / row, give both or neither
        suptitle: title of the whole figure
        mode: "ims", "sig1d" or "sig2d"
        save_path: where the figure is saved, None shows it instead
    Output:
        None
    """

    if x_labels is None: x_labels = [None] * (n_rows*n_cols)
    elif isinstance(x_labels, str): x_labels = [x_labels] * (n_rows*n_cols)
    if y_labels is None: y_labels = [None] * (n_rows*n_cols)
    elif isinstance(y_labels, str): y_labels = [y_labels] * (n_rows*n_cols)
    if titles is None: titles = [None] * (n_rows*n_cols)
    elif isinstance(titles, str): titles = [titles] * (n_rows*n_cols)
    if colorbars is None: colorbars = [False] * (n_rows*n_cols)
    elif isinstance(colorbars, bool): colorbars = [colorbars] * (n_rows*n_cols)
    
    datas = to_numpy(datas)

    if mode == "ims":
        assert(datas.ndim == 3)
    elif mode == "sig1d":
        assert(datas.ndim == 2)
    elif mode == "sig2d":
        assert(datas.ndim == 3)
    else:
        raise TypeError(f"Please Select a mode from ['sigs1d', 'sigs2d', 'ims'], got {mode}")

    nplots = datas.shape[0]

    assert(nplots == n_cols*n_rows)
    assert(len(x_labels) == nplots)
    assert(len(y_labels) == nplots)
    assert(len(titles) == nplots)
    assert(len(colorbars) == nplots)

    if col_titles is not None or row_titles is not None:
        assert(len(col_titles) == n_cols)
        assert(len(row_titles) == n_rows)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize = figsize, layout='constrained')
    
    if suptitle is not None:
        fig.suptitle(suptitle)

    counter = 0
    for i in range(n_rows):
        for j in range(n_cols):
            if n_cols == 1 and n_rows == 1:
                ax = axes
            elif n_cols == 1:
                ax = axes[i]
            elif n_rows == 1:
                ax = axes[j]
            else:
                ax = axes[i, j]

            if j == 0 and row_titles is not None:
                ax.text(
                    -0.3, 0.5,
                    row_titles[i],
                    transform=ax.transAxes,
                    rotation=0,
                    ha='right',
                    va='center',
                )

            if i == 0 and col_titles is not None:
                ax.text(
                    0.5, 1.2,
                    col_titles[j],
                    transform=ax.transAxes,
                    rotation=0,
                    ha='center',
                    va='center',
                )

            if mode == "ims":
                _generate_imfigure(fig, ax, 
                    datas[counter], 
                    label = titles[counter], 
                    x_label = x_labels[counter],
                    y_label = y_labels[counter],
                    colorbar = colorbars[counter],
                    aspect=None
                )

            if mode == "sig1d":
                ax.plot(datas[counter], ls="-", linewidth=0.8, label=titles[counter])
                ax.set_title(titles[counter], fontsize = 8)
                ax.set_xlabel(x_labels[counter], fontsize = 8)
                ax.set_ylabel(y_labels[counter], fontsize = 8)

            if mode == "sig2d":
                for line in datas[counter]:
                    ax.plot(line, ls="-", linewidth=0.8, label=titles[counter])
                ax.set_title(titles[counter], fontsize = 8)
                ax.set_xlabel(x_labels[counter], fontsize = 8)
                ax.set_ylabel(y_labels[counter], fontsize = 8)

            counter += 1

    if save_path is not None:
        fig.savefig(save_path, dpi = 600, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()
