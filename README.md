# Particle sedimentation for evaporating droplets

This repository corresponds to the simulations in "Coffee-ring deposits in concentrated suspensions of anisotropic colloids" by Nielsen et al.[^1] Here, particle suspensions of different volume fraction are sedimented in a triply periodic simulation box to obtain quantitative estimates for sedimentation velocities for spherical and rod-shaped particles.

[^1]: Nielsen, Samuel S., Brian C. Seper, Ryker Fish, Brennan Sprinkle, and Michelle M. Driscoll. 2026. “Coffee-Ring Deposits in Concentrated Suspensions of Anisotropic Colloids.” APS Open Science 1 (September): 000124. https://doi.org/10.1103/rzgy-j37g.

<table>
  <tr>
    <td>
      <video src="https://github.com/user-attachments/assets/f232611a-0443-444e-9272-d4503b539e79"></video>
    </td>
    <td>
      <video src="https://github.com/user-attachments/assets/ea1f2a63-1a91-4dd3-ba09-ded2de55535d"></video>
    </td>
  </tr>
</table>


## Instructions
To obtain the dependencies (requires GPU to simulate):
```bash
conda env create -f environment.yml
```

To run a simulation, you need a geometry for each individual particle and a configu


ration for where particles are in the simulation box. We provide particle geometry in [`structures/`](./structures/) and steric-equilibriated configurations in [`equilibrium_configs`](./equilibrium_configs/). These configurations have already been equilibriated by us on a short timescale corresponding to the repulsive potential between particles. Simulations could be performed for additional configurations but would need to be generated and equilibriated by a user. In short, this could be done by generating non-overlapping configurations for spheres and rods (we used Skoge et al.[^torquato] and PACKMOL[^packmol]). Our packings were imperfect and required equilibriation with a very short timestep to eliminate oscillations in velocity cause by nearly-overlapping particles and the steric repulsion used to separate particles.

The main simulation file [`velocity.py`](./velocity.py) is configured to re-produce results from the paper, however it requires manually assessing runs for convergence using [`test_convergence.py`](./test_convergence.py). If a run is not converged, it can be restarted from its last configuration to collect more data until convergence.

Finally, [`parse_velocities.py`](./parse_velocities.py) will compile results into a .csv which can be plotted using [`sedimentation_plot.py](./sedimentation_plot.py). 

Animations (as shown above) of a simulation can be created by changing the `save_blob_data` flag within the simulation file, converting formats using [`parse_to_spunto.py`](./parse_to_spunto.py), and visualizing using `spunto`. While more details can be found at the [superpunto repository](https://github.com/RaulPPelaez/superpunto), a good place to start is

```bash
spunto --background 1.0 1.0 1.0 --palette viridis <filename>.spunto
```

[^torquato]: Skoge, Monica, Aleksandar Donev, Frank H. Stillinger, and Salvatore Torquato. 2006. “Packing Hyperspheres in High-Dimensional Euclidean Spaces.” Physical Review E 74 (4): 041127. https://doi.org/10.1103/PhysRevE.74.041127.

[^packmol]: Martínez, L., R. Andrade, E. G. Birgin, and J. M. Martínez. 2009. “PACKMOL : A Package for Building Initial Configurations for Molecular Dynamics Simulations.” Journal of Computational Chemistry 30 (13): 2157–64. https://doi.org/10.1002/jcc.21224.
