# Cross Section Plot

Plot cross section measurements and compare with results from previous results and MC expectation.

The `plot-cross-section.ipynb` is the main plotting script. The `utils` directory contains utility scripts to define plot style, calculate the simulated cross section and plotting scripts, the `data` directory contains new measured cross section, the `figures` directory the final plots.

All requirements are specified in the `pyproject.toml` file. The following shows how to setup a python environment with `uv`.

```
git clone git@github.com:tobias-boeckh/plot-cross-section.git
cd plot-cross-section
uv venv --python 3.13
source .venv/bin/activate
uv pip install -e .
```
