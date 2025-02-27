# Getting Started

## Installation
PGTG is available on [PyPi](https://pypi.org/project/pgtg/) and can be installed with all major package managers:
```bash
pip install pgtg
```

## Usage
The easiest way to use PGTG is to create the environment with gymnasium:
```python
import pgtg
env = gymnasium.make("pgtg-v2")
```

The package relies on ```import``` side-effects to register the environment name so, even though the package is never explicitly used, its import is necessary to access the environment.  

The environment constructor can also be used directly:
```python
from pgtg import PGTGEnv
env = PGTGEnv()
```