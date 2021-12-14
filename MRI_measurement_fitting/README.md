### Purpose

Script for depth and shape analysis of atomic layers from secondary ion mass spectroscopy data created during the Experimental Physics of Interfaces course in August 2020. The objective was to determine whether signals from an embedded element correspond to monolayers or an extended structure.

### Function

Several approximations (infinitely thin layer, homogeneous layer of finite thickness etc.) can be evaluated. All are based on the atomic mixing depth - surface roughness - information depth model, a good overview of which is available in [[1]](#1). The primary input axis needs to already be converted from time to depth in nanometers.

<figure>
  <img
  src="https://github.com/timzuntar/numerical-utilities/blob/master/MRI_measurement_fitting/figures/sample_edge_fit.png?raw=true"
  alt="Signal falloff"
  width="500">
  <figcaption>Signal falloff at the trailing edge of a thick layer. Information about the sample has been removed for anonymity</figcaption>
</figure>

<figure>
  <img
  src="https://github.com/timzuntar/numerical-utilities/blob/master/MRI_measurement_fitting/figures/sample_monolayer_fit.png?raw=true"
  alt="Monolayer fit"
  width="500">
  <figcaption>Sample comparison of included models</figcaption>
</figure>

### References

<a id="1">[1]</a> 
S. Hofmann et al. (2014). 
Analytical and numerical depth resolution functions in sputter profiling.
Applied Surface Science, 314, 942-955.
