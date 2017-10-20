# Drag Utils

Python package with some helpful functions for working with empirical correlations 
for the fluid-particle drag force in multi-phase systems.

## Installation
Clone the repo and pip install

```bash
git checkout git@github.com:chrisk314/drag-utils.git
cd drag-utils
pip install .
```

## Example
Calculating the Ergun drag force on a collection of particles with diameters stored in 
a `numpy.ndarray` called `diams`

```python
import numpy as np
from drag_utils import p_vol
from drag_utils.correlations import Ergun

diam = np.linspace(1.e-4, 1.e-3, 1000)
total_pvol = p_vol(diam).sum()  # Calculate total particle volume

domain_vol = np.array([7.e-3,7.e-3,7.e-3]).prod()
phi = total_pvol / domain_vol  # Calculate solids fraction

fluid_vel = 5.e-4

# fluid density and dynamic viscosity can be specifed. Defaults to values of water at STP.
ergun_drag = Ergun(phi, diam, fluid_vel, norm=False)
```
