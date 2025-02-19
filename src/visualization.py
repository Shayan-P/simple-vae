import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection


def colorful_curve(xs, ys):
    if isinstance(xs, torch.Tensor):
        xs = xs.detach().cpu().numpy()
    if isinstance(ys, torch.Tensor):
        ys = ys.detach().cpu().numpy()
    points = np.array([xs, ys]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap='cool', norm=plt.Normalize(0, 1), alpha=0.5)
    lc.set_array(np.linspace(0, 1, len(xs)))
    lc.set_linewidth(2)
    plt.gca().add_collection(lc)
    return lc

def plot_mean_and_std(idxs, mean, std):
    if isinstance(idxs, torch.Tensor):
        idxs = idxs.detach().cpu().numpy()
    if isinstance(mean, torch.Tensor):
        mean = mean.detach().cpu().numpy()
    if isinstance(std, torch.Tensor):
        std = std.detach().cpu().numpy()
    plt.plot(idxs, mean)
    plt.fill_between(idxs, mean - std, mean + std, alpha=0.2)


def plot_image(image, ax=None, nx=None, ny=None, lx=None, ly=None, rx=None, ry=None):
    if ax is None:
        ax = plt.gca()
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    if (nx is not None and ny is not None):
        image = np.rot90(image.reshape(nx, ny))
    kwargs = {}
    if (lx is not None and ly is not None and rx is not None and ry is not None):
        extent=[lx, rx, ly, ry]
        kwargs['extent'] = extent
    im = ax.imshow(image, **kwargs)
    return im

def plot_line(ys):
    if isinstance(ys, torch.Tensor):
        ys = ys.detach().cpu().numpy()
    plt.plot(ys)
