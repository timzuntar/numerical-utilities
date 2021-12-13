### Purpose

explicitscheme.py contains routines for simplified 3D simulation of polymer diffusion and precursor-mediated adsorption to an interface. The adsorption mechanics of large, complex molecules with variable spatial conformations are (to say the least) nontrivial and although reconfiguration at the interface is not presently accounted for in this model, it succeeded in modeling several sets of experimental data.

### Function

The approach is roughly as follows: the liquid solution is represented with a 3D matrix of concentrations V<sub>ijk</sub> between 0 (depleted) and c<sub>b</sub> (initial), with the adsorbed interfacial film consisting of a 2D matrix C<sub>ij</sub>; in the latter case, the adsorbed concentration can range from 0 (empty) to c<sub>0</sub> (saturated).

In the first part of each timestep, a modified precursor-mediated (Kisliuk) adsorption isotherm is used to determine the local rates of additional molecule adsorption.

![](https://github.com/timzuntar/numerical-utilities/blob/master/Polymer_adsorption/figures/adsorption_rate.png?raw=true)

These depend both on the adsorption coefficient R and on a "cooperation parameter" k<sub>e</sub>; a positive value of k<sub>e</sub> means that molecules preferentially adsorb to sites with previously adsorbed molecules, mimicking favourable intermolecular interactions in their adsorbed state. This means that replacement of a ruptured adsorbed layer is not entirely homogeneous even after accounting for differences in diffusion rates - the rupture will fill from the outside in.

In the second part, the diffusion process within the solution is simulated:


![](https://github.com/timzuntar/numerical-utilities/blob/master/Polymer_adsorption/figures/concentration_timestep.png?raw=true)


Stop conditions are either exceeded maximum time or concentrations in each surface cell surpassing a specified minimum limit. Fitting of experimental data is then achieved with defining a ROI for space-averaging of data; the obtained c(t) curve can then be compared with measured concentrations. Ideally, the ROI defined in the simulation should be as similar as possible to the area which the measurement represents (exact matching is unlikely due to downsampling of images). The fitting parameters are then iteratively adjusted with a minimization routine until convergence is achieved. An example of a best fit is shown below - it should be noted that this is very much a best case scenario, though.

![](https://github.com/timzuntar/numerical-utilities/blob/master/Polymer_adsorption/figures/example_fit.png?raw=true)

The expected number of fitting parameters is 3 - you will need to modify some lines if you're working with fewer, or with different constraints. A next-neighboring-site weighing parameter "a" can be added as a 4th.

Values like the desired timestep, bulk concentration, maximum concentration, ROI bounds, initial parameter values etc. are read from a table of inputs. The final value is a time delay - that is, the number of seconds between the formation of a fresh solution/air interface and the start of signal acquisition.

Parallelisation with numba does not reduce time demands in all cases, but is included as an option.
