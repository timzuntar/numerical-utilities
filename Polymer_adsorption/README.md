3D simulation of polymer diffusion and precursor-mediated adsorption

The approach is roughly as follows: bulk/air interfaces are presented as a 2D grid, the bulk as a 3D grid; each cell has its own adsorbed concentration ranging from 0 to c0 (saturated). A cell's likelihood to adsorb additional molecules is modified by its own surface concentration (and possibly that of neighboring cells) so in the event of a rupture the film regrowth is not entirely homogeneous. This is represented by a precursor-mediated adsorption process in which the adsorption rate depends both on the polymer concentration immediately below the interface and the amount of already adsorbed material. Stop conditions for the simulation are either exceeded maximum time or concentrations in each surface cell surpassing a specified minimum limit.

Fitting to experimental data is achieved with defining a ROI for space-averaging of data; a single curve is obtained, which is then fitted against experimental signals. Ideally, the ROIs should be as similar as possible (exact matching is unlikely due to downsampling of images).
The model is 3-parametric in practice. The neighbour weighing parameter "a" can be added as a 4th.

Parallelisation with numba does not reduce time demands in all cases, but is included as an option.
