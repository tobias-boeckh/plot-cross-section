from pathlib import Path

import numpy as np


def get_luminosity() -> float:
    return 65.6  # \fb


def get_project_path() -> Path:
    """Get path of project directory which contains e.g. pyproject.toml file"""
    current_path = Path.cwd()
    while current_path != current_path.parent:
        if (current_path / "pyproject.toml").is_file():
            return current_path
        current_path = current_path.parent
    raise FileNotFoundError("pyproject.toml not found in the project structure.")


def get_figure_path() -> Path:
    figure_path = get_project_path() / "figures"
    figure_path.mkdir(exist_ok=True)
    return figure_path


def remove_extension(name: str) -> str:
    """Remove '.pdf' or '.svg' extension from filename."""
    if name.endswith((".pdf", ".svg")):
        name = name.rsplit(".", maxsplit=1)[0]
    return name


def get_qop_edges() -> np.ndarray:
    """Edges of qop bins in GeV"""
    return np.array(
        [-1 / 100, -1 / 300, -1 / 600, -1 / 1000, 1 / 1000, 1 / 300, 1 / 100]
    )


def get_inv_qop_bins(
    edges: np.ndarray | None = None, upper_edge: float = 1868.2
) -> np.ndarray:
    """
    Momentum bins from inverse qop bins in GeV
    :param upper_edge: momentum for upper edge of common bin
    :return: 2d-array, where axis 0 is the bin number and axis 1 is the lower and upper
    edge of the bin
    """
    if edges is None:
        edges = get_qop_edges()
    qop_bins = np.array(list(zip(edges[:-1], edges[1:])))
    energy_bins = np.abs(1 / qop_bins)
    energy_bins.sort(axis=1)
    # common bin is from -1/1000 to 1/1000, thus both inverse edges are the same
    energy_bins[energy_bins[:, 0] == energy_bins[:, 1], 1] = upper_edge
    return energy_bins


def get_num_qop_bins() -> int:
    return len(get_qop_edges()) - 1


def get_log_mean(a: float, b: float) -> float:
    if a <= 0 or b <= 0:
        raise ValueError("a and b must be positive")
    return 10 ** (0.5 * (np.log10(a) + np.log10(b)))
