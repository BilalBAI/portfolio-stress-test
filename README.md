# PortRisk

A python-based automatic stress testing tool to evaluate the portfolio risk under various scenarios.

## Installation

Run the following to create the portrisk conda environment:

```bash
conda create -n portrisk python=3.8 pip wheel
conda activate portrisk
python -m pip install -e .
```

## Usage


## `.env`

In the `.env` you need the following keys:

```ini
# Dir to a sqlite db for BBG Cached data. e.g. C:/portrisk/db/ 
# Default to ./ if not provided
# Auto create new db if db doesn't exist
BBG_CACHE_DB_DIR=<DIR>
```
Bloomberg API access is required.
