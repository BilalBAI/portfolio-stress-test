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


## Repository Structure

```
portfolio-stress-test/
│
├── examples/
│   └── example_usage
├── qrisklab/
│   ├── core/
│       ├── black_scholes.py
│       ├── stress_tests.py
│       ├── utils.py
│       ├── vol_surface_shocks.py
│   ├── clients.py
│   ├── crypto.py
│   ├── equity.py
│   ├── fx.py
├── tests/
│   ├── test_data.py
│   ├── test_models.py
│   └── test_stress_tests.py
├── Dockerfile
├── README.md
├── requirements-docker.txt
├── requirements-linux.txt
├── setup.cfg
└── setup.py
```

## Modules

### Core Modules (`core`)
#### Black Scholes (`black_sholes.py`)
A implementation of Generalized Black-Scholes-Merton Option Pricing model. Along with delta, vega and gamma calculation. 

#### Spot Vol Stress `spot_vol_stress.py`
Implements spot and vol shocks: 
1. Vanilla stress test for linear positions and options
2. A stress tree
3. Multilevel stress tests by combining vanilla stress test with stress tree

#### Utils (`utils.py`)
Includes various utility functions that assist with parameter processing, datetime manipulation, and other common tasks used throughout the project.

#### Vol Surface Stress`vol_surface_stress.py`
Implements volatility surface shocks:
1. Parallel shock
2. Term structure shock
3. Skew shock


### Clients (`clients.py`)
This module handles the fetching and processing of required data for a given portfolio. It interfaces with the Bloomberg API to retrieve necessary data and supports caching to improve performance.

### Equity `equity.py`
Implements equity shocks:
1. Macro
2. Sector
3. Concentration
4. Relative Value
5. Delta liquidation

### Crypto `crypto.py`
Implements crypto shocks:
1. Spot and Vol shocks
2. Vol Surface Shocks