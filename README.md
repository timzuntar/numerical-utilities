# numerical-utilities
A collection of poorly written code snippets for numerical computing and data analysis. Documentation is a work-in-progress (as much as I would like to say the code is self-documenting, that is far from true).

Most C programs make use of GSL libraries. These are compiled with flags -lgsl and -lgslcblas. Any Fourier transforms that made their way in rely on FFTW, so add the flag -lfftw3.
