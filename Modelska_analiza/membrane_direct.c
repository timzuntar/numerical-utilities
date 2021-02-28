//Determines sag of weighed membrane with direct method.
//Three slightly different approaches in Fourier space are given

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <time.h>
#include <sys/time.h>

#include <fftw3.h>  //for solving Poisson's equation using FFT

#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_complex.h>
#include <gsl/gsl_complex_math.h>
#include <gsl/gsl_linalg.h>

uint64_t get_posix_clock_time ()
{
    struct timespec ts;

    if (clock_gettime (CLOCK_MONOTONIC, &ts) == 0)
        return (uint64_t) (ts.tv_sec * 1000000 + ts.tv_nsec / 1000);
    else
        return 0;
}

bool odd_extension(gsl_matrix * U, gsl_matrix_complex * U_in, int Nx, int Ny) {
    //extends the matrix; needed for FFT operations
    for (int i = 1; i < Ny-1; i++) {
        for (int j = 1; j < Nx-1; j++) {
            gsl_matrix_complex_set(U_in, i-1, j+Nx-3, gsl_complex_rect(gsl_matrix_get(U,i,j),0.0)); //right upper quadrant
            gsl_matrix_complex_set(U_in, i-1, j-1, gsl_complex_rect(-gsl_matrix_get(U,i,Nx-j-1),0.0));    //left upper
            gsl_matrix_complex_set(U_in, i+Ny-3, j-1, gsl_complex_rect(gsl_matrix_get(U,Ny-i-1,Nx-j-1),0.0)); //left lower
            gsl_matrix_complex_set(U_in, i+Ny-3, j+Nx-3, gsl_complex_rect(-gsl_matrix_get(U,Ny-i-1,j),0.0)); //right lower
        }
    }
    return true;
}

bool poisson_solver_2DFFT(gsl_matrix * U, gsl_matrix * weights, gsl_matrix_complex * poisson_U, gsl_matrix_complex * poisson_weights, int Nx, int Ny, double dx) {
    //solves the poisson equation (laplacian of U = weights)
    gsl_complex value = gsl_complex_rect(0.0,0.0);

    odd_extension(U, poisson_U, Nx, Ny);
    odd_extension(weights, poisson_weights, Nx, Ny);

	fftw_complex* in_U = (fftw_complex*) poisson_U->data;
	fftw_complex* in_vort = (fftw_complex*) poisson_weights->data;

    fftw_plan Uforward = fftw_plan_dft_2d(2*(Ny-2), 2*(Nx-2), in_U, in_U, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_plan vortforward = fftw_plan_dft_2d(2*(Ny-2), 2*(Nx-2), in_vort, in_vort, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_plan Ubackward = fftw_plan_dft_2d(2*(Ny-2), 2*(Nx-2), in_U, in_U, FFTW_BACKWARD, FFTW_ESTIMATE);

    fftw_execute(Uforward);
    fftw_execute(vortforward);

    for (int i = 0; i < 2*(Ny-2); i++) {
        for (int j = 0; j < 2*(Nx-2); j++) {
            //adjust coefficients
            //value = gsl_complex_mul_real(gsl_matrix_complex_get(poisson_weights,i,j), 0.5*dx*dx/(cos(2*M_PI*i/Ny) + cos(2*M_PI*j/Nx)-2.0) );
            value = gsl_complex_mul_real(gsl_matrix_complex_get(poisson_weights,i,j), 0.5*dx*dx/(cos(M_PI*i/(Ny-2)) + cos(M_PI*j/(Nx-2))-1.999999999999) );
            gsl_matrix_complex_set(poisson_U,i,j,value);
        }
    }
    gsl_matrix_complex_scale(poisson_U, gsl_complex_rect(1.0/(4*(Nx-2)*(Ny-2)),0));

    fftw_execute(Ubackward);  //after the reverse transform, poisson_U contains the coefficients we seek

    for (int i = 0; i < Ny-2; i++) {
        for (int j = 0; j < Nx-2; j++) {
            gsl_matrix_set(U, i+1,j+1, GSL_REAL(gsl_matrix_complex_get(poisson_U,i,j+Nx-2)));
            //gsl_matrix_set(U, i+1,j+1, GSL_REAL(gsl_matrix_complex_get(poisson_U,i,j)));
        }
    }  

    fftw_destroy_plan(Uforward);  //cleanup
    fftw_destroy_plan(Ubackward);
    fftw_destroy_plan(vortforward);

    return true;
}

bool poisson_solver_2DsineFFT(gsl_matrix * U, gsl_matrix * weights, int Nx, int Ny, double dx) {
    //we don't need extra matrices this time because the real-valued transforms can be computed in-place
    double value;
	double* in_weights = (double*) weights->data;
    double* in_U = (double*) U->data;
    int Nrow[] = {Nx};
    int Ncol[] = {Ny};
    const fftw_r2r_kind kind[] = {FFTW_RODFT00};

    fftw_plan columns = fftw_plan_many_r2r(1, Ncol, Nx, in_weights, Ncol,Nx, 1, in_weights, Ncol, Nx, 1, kind, FFTW_ESTIMATE);
    fftw_plan rows = fftw_plan_many_r2r(1, Nrow, Ny, in_weights, Nrow,1, Nx, in_weights, Nrow, 1, Nx, kind, FFTW_ESTIMATE);
    fftw_plan Ucolumns = fftw_plan_many_r2r(1, Ncol, Nx, in_U, Ncol,Nx, 1, in_U, Ncol, Nx, 1, kind, FFTW_ESTIMATE);
    fftw_plan Urows = fftw_plan_many_r2r(1, Nrow, Ny, in_U, Nrow,1, Nx, in_U, Nrow, 1, Nx, kind, FFTW_ESTIMATE);

    fftw_execute(columns);
    fftw_execute(rows);

    for (int i = 0; i < Ny; i++) {
        for (int j = 0; j < Nx; j++) {
            //adjust coefficients
            value = gsl_matrix_get(weights,i,j) * (0.5*dx*dx/(cos(M_PI*(i+1)/(Ny)) + cos(M_PI*(j+1)/(Nx))-2.0));
            gsl_matrix_set(U,i,j,value);
        }
    }

    fftw_execute(Urows);
    gsl_matrix_scale(U, 1.0/(2*Nx));
    fftw_execute(Ucolumns);
    gsl_matrix_scale(U, 1.0/(2*Ny));

    fftw_destroy_plan(columns);
    fftw_destroy_plan(rows);
    fftw_destroy_plan(Urows);
    fftw_destroy_plan(Ucolumns);

    return true;
}

bool poisson_solver_sineFFT_thomas(gsl_matrix * U, gsl_matrix * weights, int Nx, int Ny, double dx) {
	double* in_weights = (double*) weights->data;
    double* in_U = (double*) U->data;
    int Ncol[] = {Ny};
    const fftw_r2r_kind kind[] = {FFTW_RODFT00};

    fftw_plan columns = fftw_plan_many_r2r(1, Ncol, Nx, in_weights, Ncol,Nx, 1,in_weights, Ncol, Nx, 1, kind, FFTW_ESTIMATE);
    fftw_plan Ucolumns = fftw_plan_many_r2r(1, Ncol, Nx, in_U, Ncol,Nx, 1,in_U, Ncol, Nx, 1, kind, FFTW_ESTIMATE);

    gsl_vector * diag = gsl_vector_alloc(Nx);
    gsl_vector * superdiag = gsl_vector_alloc(Nx-1);
    gsl_vector * subdiag = gsl_vector_alloc(Nx-1);
    gsl_vector * Gline = gsl_vector_alloc(Nx);
    gsl_vector * Uline = gsl_vector_alloc(Nx);
    gsl_vector_set_all(superdiag,1.0);
    gsl_vector_set_all(subdiag,1.0);


    fftw_execute(columns);
    for (int m = 0; m < Ny; m++) {  //we solve a tridiagonal const.-coefficient system for each line
        gsl_vector_set_all(diag,2.0*cos((m+1)*M_PI/(Ny))-4.0);

        for (int l = 0; l < Nx; l++) {  //for each element in line
            gsl_vector_set(Gline,l,dx*dx*gsl_matrix_get(weights,m,l));
        }
        gsl_linalg_solve_tridiag(diag, superdiag, subdiag, Gline, Uline);
        for (int l = 0; l < Nx; l++) {
            gsl_matrix_set(U,m,l,gsl_vector_get(Uline,l));
        }        
    }
    //we now have a system of solved semi-Fourier coefficients
    //the actual solution is computed with another reverse FFT
    fftw_execute(Ucolumns);
    gsl_matrix_scale(U, 1.0/(4*(Ny+1)));

    fftw_destroy_plan(columns);
    fftw_destroy_plan(Ucolumns);
    gsl_vector_free(diag);
    gsl_vector_free(superdiag);
    gsl_vector_free(subdiag);
    gsl_vector_free(Gline);
    gsl_vector_free(Uline);

    return true;
}

bool heateq_solver_sineFFT_thomas(gsl_matrix * U, gsl_matrix * weights, int Nr, int Nh, double dr,double Tdiff) {
	double* in_weights = (double*) weights->data;
    double* in_U = (double*) U->data;
    int Ncol[] = {Nh};
    const fftw_r2r_kind kind[] = {FFTW_RODFT00};

    fftw_plan columns = fftw_plan_many_r2r(1, Ncol, Nr, in_weights, Ncol,Nr, 1,in_weights, Ncol, Nr, 1, kind, FFTW_ESTIMATE);
    fftw_plan Ucolumns = fftw_plan_many_r2r(1, Ncol, Nr, in_U, Ncol,Nr, 1,in_U, Ncol, Nr, 1, kind, FFTW_ESTIMATE);

    gsl_vector * diag = gsl_vector_alloc(Nr);
    gsl_vector * superdiag = gsl_vector_alloc(Nr-1);
    gsl_vector * subdiag = gsl_vector_alloc(Nr-1);
    gsl_vector * Gline = gsl_vector_alloc(Nr);
    gsl_vector * Uline = gsl_vector_alloc(Nr);

    fftw_execute(columns);
    for (int n = 0; n < Nh; n++) {  //we solve a tridiagonal const.-coefficient system for each line
        //prepare diagonal coefficients
        for (int r = 0; r < Nr-1; r++) {
            gsl_vector_set(superdiag,r,4.0+2.0/r);
            gsl_vector_set(subdiag,r,4.0-2.0/(r+1.0));
        }
        gsl_vector_set(superdiag,0,-1.0);
        gsl_vector_set(subdiag,Nr-2,0.0);
        gsl_vector_set_all(diag,2.0*cos((n+1)*M_PI/(Nh))-10.0);
        gsl_vector_set(diag,0,1.0);
        gsl_vector_set(diag,Nr-1,1.0);

        for (int l = 0; l < Nr; l++) {  //for each element in line
            gsl_vector_set(Gline,l,gsl_matrix_get(weights,n,l));
        }
        gsl_linalg_solve_tridiag(diag, superdiag, subdiag, Gline, Uline);
        for (int l = 0; l < Nr; l++) {
            gsl_matrix_set(U,n,l,gsl_vector_get(Uline,l));
        }        
    }
    //we now have a system of solved semi-Fourier coefficients
    //the actual solution is computed with another reverse FFT
    fftw_execute(Ucolumns);
    gsl_matrix_scale(U, 1.0/(2*(Nh+1)));

    fftw_destroy_plan(columns);
    fftw_destroy_plan(Ucolumns);
    gsl_vector_free(diag);
    gsl_vector_free(superdiag);
    gsl_vector_free(subdiag);
    gsl_vector_free(Gline);
    gsl_vector_free(Uline);

    return true;
}

bool set_profile_Q(gsl_matrix * profile, int Nx, int Ny, double beta) {
   //needs to be divisible by 5, so:
   if (Nx%5 != 0 || Ny != Nx) {
      return false;
   }
   int Npart = Nx/5;

   gsl_matrix * placeholder = gsl_matrix_alloc(Ny+1,Nx+1);
   gsl_matrix_set_all(placeholder,1.0);

   for (int i=0; i<Nx+1;i++) {
      for (int j=0; j<Ny+1;j++) {
         if (j >= 2*Npart && j <= 3*Npart && i >= Npart && i <= 3*Npart) {gsl_matrix_set(placeholder,i,j,beta);}
         if (j >= 0 && j <= 2*Npart && i >= 4*Npart && i <= 5*Npart) {gsl_matrix_set(placeholder,i,j,beta);}

         if (j >= 0 && j <= Npart && i >= 0 && i <= Npart && j+i <= Npart) {gsl_matrix_set(placeholder,i,j,beta);}
         if (j >= 0 && j <= Npart && i >= 3*Npart && i <= 4*Npart && i-j >= 3*Npart) {gsl_matrix_set(placeholder,i,j,beta);}
         if (j >= 2*Npart && j <= 3*Npart && i >= 4*Npart && i <= 5*Npart && i-j >= 2*Npart) {gsl_matrix_set(placeholder,i,j,beta);}
         if (j >= 4*Npart && j <= 5*Npart && i >= 0 && i <= Npart && j-i >= 4*Npart) {gsl_matrix_set(placeholder,i,j,beta);}
         if (j >= 0 && j <= Npart && i >= 0 && i <= Npart && j+i <= Npart) {gsl_matrix_set(placeholder,i,j,beta);}
         if (j >= 4*Npart && j <= 5*Npart && i >= 4*Npart && i <= 5*Npart && j-i >= 0) {gsl_matrix_set(placeholder,i,j,beta);}
         if (j >= 4*Npart && j <= 5*Npart && i >= 3*Npart && i <= 4*Npart && j+i >= 8*Npart) {gsl_matrix_set(placeholder,i,j,beta);}

         if (j >= Npart && j <= 2*Npart && i >= Npart && i <= 2*Npart && j+i >= 3*Npart) {gsl_matrix_set(placeholder,i,j,beta);}
         if (j >= 3*Npart && j <= 4*Npart && i >= 2*Npart && i <= 3*Npart && j+i <= 6*Npart) {gsl_matrix_set(placeholder,i,j,beta);}
         if (j >= Npart && j <= 2*Npart && i >= 2*Npart && i <= 3*Npart && i-j <= Npart) {gsl_matrix_set(placeholder,i,j,beta);}
         if (j >= 3*Npart && j <= 4*Npart && i >= Npart && i <= 2*Npart && j-i <= 2*Npart) {gsl_matrix_set(placeholder,i,j,beta);}
      }
   }

   for (int i=0; i<Nx+1;i++) {
      for (int j=0; j<Ny+1;j++) {
          gsl_matrix_set(profile,i,j,gsl_matrix_get(placeholder,i,j));
      }
   }

   gsl_matrix_free(placeholder);
   return true;
}

bool print_matrix(gsl_matrix * in, int Nx, int Ny) {
    for (int i=0; i<Nx;i++) {
        for (int j=0; j<Ny;j++) {
        printf("%.0f ",gsl_matrix_get(in,i,j));
        }
        printf("\n");
    }
    return true;
}

bool fprint_matrix_pm3d(FILE * output, gsl_matrix * in, int Nx, int Ny) {
    for (int j=0; j<Ny;j++) {
        for (int i=0; i<Nx;i++) {
            fprintf(output, "%f %f %f\n",(double)j/(Ny-1),(double)i/(Nx-1),gsl_matrix_get(in,j,i));
        }
        fprintf(output, "\n");
    }

    return true;
}

bool crop_edges(gsl_matrix * cropped, gsl_matrix * in, int Nx, int Ny) {
    for (int i=0; i<Ny-2;i++) {
        for (int j=0; j<Nx-2;j++) {
            gsl_matrix_set(cropped,i,j,gsl_matrix_get(in,i+1,j+1));
        }
    }
    return true;
}

bool add_edges(gsl_matrix * cropped_solution, gsl_matrix * out, int Nx, int Ny) {
    gsl_matrix_set_all(out,0.0);
    for (int i=0; i<Ny-2;i++) {
        for (int j=0; j<Nx-2;j++) {
            gsl_matrix_set(out,i+1,j+1,gsl_matrix_get(cropped_solution,i,j));
        }
    }
    return true;
}

double get_min_element(gsl_matrix * in, int Nx, int Ny) {
    double minval = 0.0;
    for (int i=0; i<Ny;i++) {
        for (int j=0; j<Nx;j++) {    
            if (gsl_matrix_get(in,i,j) < minval) {
                minval = gsl_matrix_get(in,i,j);
            }
        }
    }
    return minval;
}

double get_mean_value(gsl_matrix * in, int Nx, int Ny) {
    double mean = 0.0;
    for (int i=0; i<Ny;i++) {
        for (int j=0; j<Nx;j++) {    
            mean += gsl_matrix_get(in,i,j);
        }
    }
    return mean/(Nx*Ny);
}

////////
//MAIN//
////////

//Sag of weighted square membrane 
int main(void) {
    int Nx = 254, Ny = 254;   //actual size of array that will get passed to FFT
    double dx = 1.0/((double)Nx-1);
    double mean;

    uint64_t prev_time_value, time_value;   //for timekeeping
    uint64_t time_diff;
    char bufferdir[100];

    gsl_matrix * profile = gsl_matrix_alloc(Ny+2,Nx+2);
    gsl_matrix * weights = gsl_matrix_alloc(Ny,Nx);
    gsl_matrix * solution = gsl_matrix_alloc(Ny,Nx);
    gsl_matrix_complex * weights_extended = gsl_matrix_complex_alloc (2*(Ny-2), 2*(Nx-2));
    gsl_matrix_complex * solution_extended = gsl_matrix_complex_alloc (2*(Ny-2), 2*(Nx-2));

    snprintf(bufferdir, 100, "rezultati/betavar_sag_N64_normalized.dat");
    FILE * sag = fopen(bufferdir, "a+");

    for (double beta = 1.0; beta < 1.01; beta += 1.0) {
        set_profile_Q(profile,Ny+1,Nx+1,beta);
        //gsl_matrix_scale(profile,1.0/get_mean_value(profile,Nx,Ny));    //this "normalizes" the mean density to 1

        crop_edges(weights,profile,Nx+2,Ny+2);
        //print_matrix(weights,Ny,Nx);

        prev_time_value = get_posix_clock_time ();
        //solver code goes here

        
        //ordinary 2D FFT
        poisson_solver_2DFFT(solution, weights, weights_extended, solution_extended, Ny, Nx, dx);
        
        
        //double 1D sine FFT
        //poisson_solver_2DsineFFT(solution, weights, Nx, Ny, dx);
        
        
        //sine FFT along one dimension plus tridiagonal system solving
        //poisson_solver_sineFFT_thomas(solution, weights, Nx, Ny, dx);
        
        time_value = get_posix_clock_time ();
        time_diff = time_value - prev_time_value;
        printf("\nExecution time was %f milliseconds.",time_diff/1000.0);

        add_edges(solution,profile,Ny,Nx);
/*
        snprintf(bufferdir, 100, "rezultati/betavar_times.dat");
        FILE * times_output = fopen(bufferdir, "a+");
        fprintf(times_output,"%d %d %f %f %.10f\n",Ny,Nx,beta,time_diff/1000.0,get_min_element(solution,Ny,Nx));
        fclose(times_output);
*/
        fprintf(sag,"%d %d %f %f %.10f\n",Ny,Nx,beta,time_diff/1000.0,get_min_element(solution,Ny,Nx));
        
            snprintf(bufferdir, 100, "rezultati/2DFFT_N%03d_b%.2f.dat", Nx, beta);
            FILE * profile_output = fopen(bufferdir, "wt");
            fprint_matrix_pm3d(profile_output,solution,Ny,Nx);
            fclose(profile_output);
        
    }
    fclose(sag);
    gsl_matrix_free(profile);
    gsl_matrix_free(weights);
    return 0;
}

/*
//Temperature profile in cylinder
int main(void) {
    int Nr = 2049, Nh = 2049;
    double dr = 1.0/((double)Nr-1); //dz = 2 dr
    double T1 = 1.0, T2 = 2.0;

    uint64_t prev_time_value, time_value;   //for timekeeping
    uint64_t time_diff;
    char bufferdir[100];

    gsl_matrix * weights = gsl_matrix_calloc(Nh,Nr);
    for (int h = 0; h < Nh; h++) {
        gsl_matrix_set(weights,h,Nr-1,T2-T1); //right-hand side represents the outer cylinder shell
    }
    gsl_matrix * solution = gsl_matrix_alloc(Nh,Nr);

    prev_time_value = get_posix_clock_time ();
    //solver code goes here
    //sine FFT along one dimension plus tridiagonal system solving
    heateq_solver_sineFFT_thomas(solution, weights, Nr, Nh, dr, T2-T1);

    time_value = get_posix_clock_time ();
    time_diff = time_value - prev_time_value;
    printf("\nExecution time was %f milliseconds.",time_diff/1000.0);

    snprintf(bufferdir, 100, "rezultati/cylinder_times.dat");
    FILE * times_output = fopen(bufferdir, "a+");
    fprintf(times_output,"%d %d %f %f %.10f\n",Nr,Nh,T2-T1,time_diff/1000.0,gsl_matrix_get(solution,Nh/2,0));
    fclose(times_output);

    snprintf(bufferdir, 100, "rezultati/cylinder_N%03d_Tdiff%.2f.dat", Nr, T2-T1);
    FILE * profile_output = fopen(bufferdir, "wt");
    fprint_matrix_pm3d(profile_output,solution,Nr,Nh);
    fclose(profile_output);

}
*/