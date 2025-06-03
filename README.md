# QRiskLib

qrisklab is a python-based Quantitative Risk Analytics Toolkit including:
* stress testing tools designed to evaluate portfolio risk under complex stress scenarios 
* vol curves fitting models (WingModel, SVIModel)
* option pricing, greeks calcualtion, greeks pnl attribution
* visualization tools

## Installation

Run the following to create a qrisklab conda environment:

```bash
conda create -n qrisklab python=3.12 pip wheel
conda activate qrisklab

# Download and install package
python -m pip install -e .
# Or install directly
pip install git+https://github.com/BilalBAI/qrisklab.git

```

Run the following to create a qrisklab venv:

```bash
# Create the virtual environment with Python 3.12
which python3.12
python3.12 -m venv qrisklab_env

# Activate the virtual environment
# Windows
qrisklab_env\Scripts\activate
# macOS / Linux
source qrisklab_env/bin/activate

# Verify the Python version
python --version

# Download and install package
python -m pip install -e .
# Or install directly
pip install git+https://github.com/BilalBAI/qrisklab.git

# Deactivate the virtual environment when done
deactivate
```

### `.env`

In the `.env` you need the following keys:

```ini
# Dir to a sqlite db for BBG Cached data. e.g. C:/qrisklab/db/ 
# Default to ./ if not provided
# Auto create new db if db doesn't exist
BBG_CACHE_DB_DIR=<DIR>
```

Bloomberg API access is required for equity stress test.
