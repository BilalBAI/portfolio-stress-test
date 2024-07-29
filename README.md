# PortRisk

PortRisk is a Python-based stress testing tool designed to evaluate portfolio risk under various stress scenarios. It leverages the Bloomberg API for data retrieval and offers a modular structure for customizable and flexible usage.

[High-level Framework Introduction](https://docs.google.com/document/d/1AtpFLNE6FaWGK_ipJyhvnKF6hI-dhDRQBLVtB7vZpWI/edit?usp=sharing)

## Installation

Run the following to create the portrisk conda environment:

```bash
conda create -n portrisk python=3.12 pip wheel
conda activate portrisk
python -m pip install -e .
```

### `.env`

In the `.env` you need the following keys:

```ini
# Dir to a sqlite db for BBG Cached data. e.g. C:/portrisk/db/ 
# Default to ./ if not provided
# Auto create new db if db doesn't exist
BBG_CACHE_DB_DIR=<DIR>
```

Bloomberg API access is required.


## Repository Structure

```
portfolio-stress-test/
│
├── examples/
│   └── example_usage.py
├── portrisk/
│   ├── core/
│       ├── equity.py
│       ├── fx.py
│       ├── option.py
│       ├── stress_tests.py
│   ├── black_scholes.py
│   ├── clients.py
│   ├── utils.py
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

### Data Module (`clients.py`)

This module handles the fetching and processing of required data for a given portfolio. It interfaces with the Bloomberg API to retrieve necessary data and supports caching to improve performance.

### Black Scholes Module (`black_sholes.py`)

A implementation of Generalized Black-Scholes-Merton Option Pricing model. Along with delta, vega and gamma calculation. 

### Utils Module (`utils.py`)

Includes various utility functions that assist with parameter processing, datetime manipulation, and other common tasks used throughout the project.

### Core Module (`core`)

#### `stress_test.py`
Implements spot and vol shocks: 
1. Vanilla stress test for linear positions and options
2. A stress tree
3. Multilevel stress tests by combining vanilla stress test with stress tree

#### `option.py`
Implements volatility surface shocks:
1. Parallel shock
2. Term structure shock
3. Skew shock

#### `equity.py`
Implements equity shocks:
1. Macro
2. Sector
3. Concentration
4. Relative Value
5. Delta liquidation
