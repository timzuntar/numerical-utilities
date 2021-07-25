Script for depth and shape analysis of atomic layers from secondary ion mass spectroscopy data.

Its purpose was to determine whether signal from an embedded element corresponds to a monolayer or an extended structure. Several approximations (infinitely thin layer, homogeneous layer of finite thickness etc.) can be evaluated. All are based on the atomic mixing depth - surface roughness - information depth model. The primary input axis needs to already be converted from time to depth in nanometers.

<figure>
  <img
  src="https://github.com/timzuntar/numerical-utilities/blob/master/MRI_measurement_fitting/sample_edge_fit.png?raw=true"
  alt="Signal falloff"
  width="500">
  <figcaption>Signal falloff at the trailing edge of a thick layer. Information about the sample has been removed for anonymity</figcaption>
</figure>

<figure>
  <img
  src="https://github.com/timzuntar/numerical-utilities/blob/master/MRI_measurement_fitting/sample_monolayer_fit.png?raw=true"
  alt="Monolayer fit"
  width="500">
  <figcaption>Sample comparison of included models</figcaption>
</figure>

Created during the Experimental Physics of Interfaces course in August 2020.
