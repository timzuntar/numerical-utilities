### Purpose

Trapping.py contains routines for simulation of optical trapping for a homogeneous spherical particle confined to the axis of a Gaussian beam by three approaches, namely:

- generalized Lorenz-Mie theory
- small particle (Rayleigh) limit
- large particle (ray optics / geometric) limit

and the testing of their parameter space. The code was created as part of final project for the Model Analysis 2 course in February 2021 at UL FMF. Accordingly, ease of use and readability were not top priorities. If you would like to reuse some of it and have questions about anything, feel free to shoot me a message.

### The basics

The theory behind optical trapping has been explained a million times with much higher quality than can be provided by me, so there is little sense in attempting to repeat it. Ultra-condensed version: the intensity of the TEM00 mode of a Gaussian beam with waist w<sub>0</sub>= w(z=0) at a distance r from the axis is defined as

![](https://github.com/timzuntar/numerical-utilities/blob/master/Optical_trapping/figures/intensity.png?raw=true)

How a particle in such a beam will behave depends on its size and optical properties (at a minimum, they need to differ from those of the medium). Due to the shape of the beam, a gradient force arises and the particle's potential in the EM field will - under some circumstances - possess a local minimum near the beam waist where the radiation pressure and gradient forces are in balance. If that potential well is much deeper than kT, the particle will be effectively trapped. The intensity gradients required to achieve trapping are only feasible with laser illumination. In particular, the trapping efficiency Q is defined as the ratio of trapping force magnitude divided by incident momentum per unit of time,

![](https://github.com/timzuntar/numerical-utilities/blob/master/Optical_trapping/figures/Q_factor.png?raw=true)

The particle's interaction with the field is highly complex and usually computed numerically, with many different approaches (a comprehensive comparison is provided in [[1]](#1)). Fortunately, it's often possible to make useful approximations.

### GLMT

In generalized Lorenz-Mie theory, the incident and scattered EM fields are decomposed in a base of vector spherical harmonics

![Geometry in the ray optics regime](https://github.com/timzuntar/numerical-utilities/blob/master/Optical_trapping/figures/LM_series.png?raw=true)

The coefficients are determined by lots of Bessel function evaluations and shape coefficients of the beam, which describe a slightly modified beam shape, not the actual Gaussian beam. This is also the reason the simulation breaks down at very narrow beam waists. "Fortunately", the diffraction limit rears its ugly head by then, so you shouldn't be using waists much narrower than a wavelength anyway. The relevant formulas were taken from [[2]](#2).

### Rayleigh limit

If its diameter a is much smaller than the wavelength of light, the particle can be approximated by an ideal dipole, for which an analytical solution exists.

### Ray optics limit

On the other hand, if a >> λ, the beam can be thought of as consisting of rays which reflect off and diffract through it, imparting a change in momentum in the process. In the on-axis case, light from each infinitesimally narrow annulus contacts the particle at the same angle:

![Geometry in the ray optics regime](https://github.com/timzuntar/numerical-utilities/blob/master/Optical_trapping/figures/rayoptics_diagram.png?raw=true)

Some lines of trigonometry later, we get the following equations

![](https://github.com/timzuntar/numerical-utilities/blob/master/Optical_trapping/figures/geometric_Fz.png?raw=true)
![](https://github.com/timzuntar/numerical-utilities/blob/master/Optical_trapping/figures/geometric_Fax.png?raw=true)
![](https://github.com/timzuntar/numerical-utilities/blob/master/Optical_trapping/figures/geometric_Ftot.png?raw=true)

T and R are the Fresnel transmission and reflection coefficients, while dP represents an infinitesimal amount of optical power associated with the ray. How much is the total force? While I am sure that the equations are somehow integrable, I did not bother with attempting. Instead, the beam plane is arbitrarily divided into a countable amount of annuli. The power associated with each of them is then assumed to be incident under the same angle. It's not as bad as it sounds - even n=10 gives decent results for low beam divergences.

### Demonstration

The following pictures show comparisons of some force profiles for 500 and 50 nm particles (for 100 mW beam power and a 500 nm beam waist in all cases):

![](https://github.com/timzuntar/numerical-utilities/blob/master/Optical_trapping/figures/force_comp_500nm.png?raw=true) ![](https://github.com/timzuntar/numerical-utilities/blob/master/Optical_trapping/figures/force_comp_50nm.png?raw=true)

Evidently the Rayleigh limit approach doesn't foresee a "bound state" for the 500 nm particle, likewise for the ray optics approach in the 50 nm case. Generalized Lorenz-Mie theory bridges the gap between edge cases. This is especially visible if the potential well depths are visualized for a range of particle sizes:

![](https://github.com/timzuntar/numerical-utilities/blob/master/Optical_trapping/figures/potential_well_depths.png?raw=true)

Of course, the axial trap stiffness of simulated traps can be simply calculated by fitting a parabolic function to the minimum in the potential function, but where's the fun in that? The approach used here is to first simulate the particles' Brownian motion within the trap, then calculate the power spectral density by Fourier transforming the list of z(t) data. Since the spectrum (for a simple sphere, at least) is given by the following formula:

![](https://github.com/timzuntar/numerical-utilities/blob/master/Optical_trapping/figures/Fourier_spectrum.png?raw=true)

you can estimate the cutoff frequency ω<sub>0</sub> by fitting it to the data.

![](https://github.com/timzuntar/numerical-utilities/blob/master/Optical_trapping/figures/FT_Rayleigh.png?raw=true)

## References
<a id="1">[1]</a> 
A. A. M. Bui et al. (2016). 
Theory and practice of simulation of optical tweezers.
Journal of Quantitative Spectroscopy and Radiative Transfer, 195, 66-75.

<a id="2">[2]</a> 
J. A. Lock (2004). 
Calculation of the radiation trapping force for laser tweezers by use of generalized Lorenz-Mie theory II: On-axis trapping force.
Applied Optics, 43.12, 2545.
