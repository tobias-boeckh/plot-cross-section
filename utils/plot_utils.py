import os
from pathlib import Path

import matplotlib.pyplot as plt
import skunk
from cycler import Cycler
from matplotlib.offsetbox import AnnotationBbox

from utils.colors import TangoColors
from utils.utils import get_figure_path, remove_extension


def setup(color_cycler: Cycler = TangoColors.cycler, errorbar_size: int = 2) -> None:
    """Setup matplotlib font size, error capsize, legend and colors"""
    plt.rcParams.update(
        {
            "font.size": 13,
            "errorbar.capsize": errorbar_size,
            "legend.frameon": False,
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "DejaVu Sans"],
            "text.usetex": True,
            "text.latex.preamble": r"\usepackage{amsmath}",
            # "figure.facecolor": None,
            # "axes.facecolor": None,
            # "savefig.transparent": True,
        }
    )
    plt.rc("axes", prop_cycle=color_cycler)


def add_logo(
    ax: plt.Axes,
    xy: tuple[float] = (0, 1.005),
    scale: float = 1,
    pad: float = 0.0,
    logo_size: tuple[float, float] = (70, 20),
    text: str = "",
    text_fontsize: int = 12,
    **kwargs,
) -> str:
    """
    Add vector graphic of FASER logo to the plot.
    ax: matplotlib axes object.
    xy: Fraction of the axes to position the logo from the lower left.
    scale: Scale of image size.
    pad: Padding of the logo from the axes.
    logo_size: Size of the logo in points.
    logo_path: Path to the logo file.
    text: Text to add next to the logo.
    kwargs: Additional arguments for the text.
    return: SVG string of plot.
    """
    box = skunk.Box(logo_size[0] * scale, logo_size[1] * scale, "sk_logo")
    ab = AnnotationBbox(
        box,
        xy=xy,
        xybox=xy,
        xycoords="axes fraction",
        boxcoords="offset points",
        box_alignment=(0, 0),
        pad=pad,
        frameon=False,
    )
    ax.add_artist(ab)

    if text != "":
        # get width and height of the logo
        renderer = ax.figure.canvas.get_renderer()
        bbox = ab.get_window_extent(renderer=renderer)
        bbox_data = bbox.transformed(ax.transAxes.inverted())
        width = bbox_data.width
        height = bbox_data.height

        x, y = ab.xy
        ax.text(
            x + width + 0.02,
            y + height / 2,
            text,
            ha="left",
            va="center",
            transform=ax.transAxes,
            fontsize=text_fontsize,
            **kwargs,
        )


def show(
    logo_path: str | os.PathLike = "/home/tboeckh/Downloads/faser_logo.svg",
) -> None:
    """Show svg plot."""
    # plt.tight_layout()
    svg = skunk.insert({"sk_logo": logo_path})
    skunk.display(svg)
    plt.clf()


def save(
    name: str | Path,
    dir: str | Path | None = None,
    show_plot: bool = True,
    save_svg: bool = True,
    logo_path: str | os.PathLike = "/home/tboeckh/Downloads/faser_logo.svg",
) -> None:
    """Save svg as PDF and/or SVG."""

    # plt.tight_layout()
    svg = skunk.insert({"sk_logo": logo_path})

    if isinstance(dir, str):
        _dir = Path(dir)
    elif isinstance(dir, Path):
        _dir = dir
    else:
        _dir = get_figure_path()
    if name is Path:
        _name = name.name
        if dir is None:
            _dir = name.parent
    else:
        _name = name
    _name = remove_extension(_name)
    if save_svg:
        with open(_dir / f"{_name}.svg", "w") as f:
            f.write(svg)
    if show_plot:
        show()
    else:
        # save without showing, e.g. if there are multiple versions with and without logo
        plt.close()


def show_svg(
    logo_path: str | os.PathLike | None = None,
) -> None:
    """Show svg plot."""
    if logo_path is None:
        logo_path = str(get_figure_path() / "faser_logo.svg")
    svg = skunk.insert({"sk_logo": logo_path})
    skunk.display(svg)
    plt.clf()


def save_svg(
    name: str | Path,
    dir: str | Path | None = None,
    show_plot: bool = True,
    save_svg: bool = True,
    logo_path: str | os.PathLike | None = None,
) -> None:
    """Save svg as PDF and/or SVG."""

    if logo_path is None:
        logo_path = str(get_figure_path() / "faser_logo.svg")
    svg = skunk.insert({"sk_logo": logo_path})

    if isinstance(dir, str):
        _dir = Path(dir)
    elif isinstance(dir, Path):
        _dir = dir
    else:
        _dir = get_figure_path()
    if name is Path:
        _name = name.name
        if dir is None:
            _dir = name.parent
    else:
        _name = name
    _name = remove_extension(_name)
    if save_svg:
        with open(_dir / f"{_name}.svg", "w") as f:
            f.write(svg)
    if show_plot:
        show_svg()
    else:
        # save without showing, e.g. if there are multiple versions with and without logo
        plt.close()


def save(name: str, show_logo: bool, dir: str = ".", show_plot: bool = True) -> None:
    if show_logo:
        save_svg(dir=get_figure_path() / dir, name=name, show_plot=show_plot)
    else:
        plt.savefig(get_figure_path() / dir / f"{name}.pdf", bbox_inches="tight")
        if show_plot:
            plt.show()
        else:
            # save without showing, e.g. if there are multiple versions with and without logo
            plt.close()


def show(show_logo: bool):
    if show_logo:
        show_svg()
    else:
        plt.show()
