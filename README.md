# Scattering TMM
 
A Python-based scattering-matrix transfer matrix method (TMM) solver.

# Example Usage
There are two accepted formats for layer input. A layer can be described with its permittivity and permeability, like so:
```py
layer = [epsilon, myu, thickness]
```

Or, it can be represented with it's refractive index, like so:
```py
layer = [refractive index, thickness]
```
 
Thickness is assumed to be semi-infinite for the reflection and transmission layers
As a result we leave their values as `None`.
```py
layers = [
    [1.0, 1.0, None], # reflection layer
    [2.5, 2.0, 5e-4], # epsilon, myu, thickness
    [1.5     , 2e-4], # refractive index, thickness
    [3.5, 1.0, 5e-4], # epsilon, myu, thickness
    [1.0, 1.0, None], # transmission layer
] 

theta = 0
phi = 0
pte = 0.5
ptm = 0.5
wavelength = 1e-6

R, T = tmm_solve(layers, theta, phi, pte, ptm, wavelength)
```