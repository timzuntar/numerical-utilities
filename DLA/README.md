Very simple random walk model of 2D diffusion-limited aggregation. A "seed" with value 1 is placed in the centre of a zero-initialised array. A "Particle" added at a random position then undergoes a random walk, rolling the dice to which of its neighboring positions in the array it moves every time step. If it moves next to the seed, it adheres to it by flipping the corresponding value in the array from 0 to 1. The process is repeated and the aggregate grows. I haven't bothered to add periodic boundary conditions so in the case of a move beyond array boundaries it is forgotten about and a new random walk is started.

<figure>
  <img
  src="https://github.com/timzuntar/numerical-utilities/blob/master/DLA/cluster6-6000pts.png?raw=true"
  alt="DLA aggregate">
  width="350"
  <figcaption>Example of a generated aggregate</figcaption>
</figure>

You can track the aggregates' radii of gyration and density correlations, however function fitting to calculate their Hausdorff dimensions was done in an external program and the functionality isn't included here.

<figure>
  <img
  src="https://github.com/timzuntar/numerical-utilities/blob/master/DLA/cluster_density_correlations.png?raw=true"
  alt="Dimension fit">
  width="500"
  <figcaption>An average over 4 clusters gives D = 1.609 +/- 0.005, but fit is poor for larger values</figcaption>
</figure>

Created as part of a seminar for the Soft Matter Physics course at UL FMF in January 2020.
