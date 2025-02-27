# ProcGrid Traffic Gym (PGTG)

ProcGrid Traffic Gym (PGTG) is a driving simulation on a grid with procedural generated maps and traffic. It is fully compatible with the [Gymnasium API standard](https://gymnasium.farama.org/).

## Getting Started

### Installation
PGTG is available on [PyPi](https://pypi.org/project/pgtg/) and can be installed with all major package managers:
```bash
pip install pgtg
```

### Usage
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

## [Documentation](https://pgtg-org.github.io/pgtg)

1. [Getting Started](https://pgtg-org.github.io/pgtg/Getting_Started.md)
2. [The Environment](https://pgtg-org.github.io/pgtg/Environment.md)
3. [Customizing PGTG](https://pgtg-org.github.io/pgtg/Constructor_Arguments.md)
4. [Version History](https://pgtg-org.github.io/pgtg/Version_History.md)