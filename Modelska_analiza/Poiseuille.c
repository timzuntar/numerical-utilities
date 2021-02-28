//Determination of volume flow through pipe with a weird cross-section using SOR and Chebyshev acceleration

#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include <math.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>
#include <stdint.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <time.h>
#include <sys/time.h>


uint64_t get_posix_clock_time ()
{
    struct timespec ts;

    if (clock_gettime (CLOCK_MONOTONIC, &ts) == 0)
        return (uint64_t) (ts.tv_sec * 1000000 + ts.tv_nsec / 1000);
    else
        return 0;
}

double Jacobi_spectral_radius(int Nx, int Ny, double dx, double dy, double d) {
   double sigma = cos(M_PI/(Nx))/(dx*dx) + cos(M_PI/(Ny))/(dy*dy);
   sigma *= 2.0/d;
   return sigma;
}

double optimum_omega_calc(double sigma_Jacobi) {
   if (fabs(sigma_Jacobi) > 1.0) {return 0.0;}
   else {
      return 2.0/(1 + sqrt(1.0 - sigma_Jacobi*sigma_Jacobi));
   }
}

bool Jacobi_iteration(gsl_matrix * current, gsl_matrix * next, int Nx, int Ny, double dx, double dy, double rho, double d) {
   double xcomp, ycomp;

   for (int u=1;u<Nx;u++) {
      for (int v=1;v<Ny;v++) {
         xcomp = (gsl_matrix_get(current,u+1,v) + gsl_matrix_get(current,u-1,v))/(dx*dx);
         ycomp = (gsl_matrix_get(current,u,v+1) + gsl_matrix_get(current,u,v-1))/(dy*dy);
         gsl_matrix_set(next,u,v,(rho+xcomp+ycomp)/d);
      }
   }
   return true;
}

bool masked_Jacobi_iteration(gsl_matrix_int * mask, gsl_matrix * current, gsl_matrix * next, int Nx, int Ny, double dx, double dy, double rho, double d) {
   double xcomp, ycomp;

   for (int u=1;u<Nx;u++) {
      for (int v=1;v<Ny;v++) {
         xcomp = (gsl_matrix_get(current,u+1,v) + gsl_matrix_get(current,u-1,v))/(dx*dx);
         ycomp = (gsl_matrix_get(current,u,v+1) + gsl_matrix_get(current,u,v-1))/(dy*dy);
         gsl_matrix_set(next,u,v,(rho+xcomp+ycomp)/d);
      }
   }
   for (int u=0;u<Nx+1;u++) {
      for (int v=0;v<Ny+1;v++) {
         if (gsl_matrix_int_get(mask,u,v) != 1) {
            gsl_matrix_set(next,u,v,0.0);
         }
      }
   }
   return true;
}

double masked_GaussSeidel_iteration(gsl_matrix_int * mask, gsl_matrix * current, gsl_matrix * next, int Nx, int Ny, double dx, double dy, double rho, double d) {
   double xcomp, ycomp;
   for (int u=0;u<Nx+1;u++) {
      for (int v=0;v<Ny+1;v++) {
         if (gsl_matrix_int_get(mask,u,v) != 1) {
            gsl_matrix_set(next,u,v,0.0);
         }
         else {
            xcomp = (gsl_matrix_get(current,u+1,v) + gsl_matrix_get(next,u-1,v))/(dx*dx);
            ycomp = (gsl_matrix_get(current,u,v+1) + gsl_matrix_get(next,u,v-1))/(dy*dy);
            gsl_matrix_set(next,u,v,(rho+xcomp+ycomp)/d);
         }
      }
   }

   return true;
}

double masked_GaussSeidel_SOR_iteration(gsl_matrix_int * mask, gsl_matrix * current, gsl_matrix * next, gsl_matrix * temp, int Nx, int Ny, double dx, double dy, double rho, double d, double omega) {
   double xcomp, ycomp;
   for (int u=1;u<Nx;u++) {
      for (int v=1;v<Ny;v++) {
         if (gsl_matrix_int_get(mask,u,v) != 1) {
            gsl_matrix_set(temp,u,v,0.0);
         }
         else {
            xcomp = (gsl_matrix_get(current,u+1,v) + gsl_matrix_get(temp,u-1,v))/(dx*dx);
            ycomp = (gsl_matrix_get(current,u,v+1) + gsl_matrix_get(temp,u,v-1))/(dy*dy);
            gsl_matrix_set(temp,u,v,(rho+xcomp+ycomp)/d);
         }
      }
   }
   //The difference between the temporary and previous fields is now multiplied by omega
   for (int u=1;u<Nx;u++) {
      for (int v=1;v<Ny;v++) {
         if (gsl_matrix_int_get(mask,u,v) != 1) {
            gsl_matrix_set(next,u,v,0.0);
         }
         else {
            gsl_matrix_set(next,u,v,(1.0-omega)*gsl_matrix_get(current,u,v) + omega*(gsl_matrix_get(temp,u,v)));
         }
      }
   }
   return true;
}

double masked_Chebyshev_semiiteration(gsl_matrix_int * mask, gsl_matrix * current, gsl_matrix * next, int Nx, int Ny, double dx, double dy, double rho, double d, double omega, bool odd) {
   double xcomp, ycomp;
   for (int u=1;u<Nx;u++) {
      for (int v=1;v<Ny;v++) {
         if (gsl_matrix_int_get(mask,u,v) != 1) {
            gsl_matrix_set(next,u,v,0.0);
         }
         else {
            if (odd == true) {
               if ((u+v)%2 != 0) {
                  xcomp = (gsl_matrix_get(current,u+1,v) + gsl_matrix_get(current,u-1,v))/(dx*dx);
                  ycomp = (gsl_matrix_get(current,u,v+1) + gsl_matrix_get(current,u,v-1))/(dy*dy);
                  gsl_matrix_set(next,u,v,(1.0-omega)*gsl_matrix_get(current,u,v) + omega*(rho+xcomp+ycomp)/d);
               }
            }
            else if (odd == false) {
               if ((u+v)%2 == 0) {
                  xcomp = (gsl_matrix_get(current,u+1,v) + gsl_matrix_get(current,u-1,v))/(dx*dx);
                  ycomp = (gsl_matrix_get(current,u,v+1) + gsl_matrix_get(current,u,v-1))/(dy*dy);
                  gsl_matrix_set(next,u,v,(1.0-omega)*gsl_matrix_get(current,u,v) + omega*(rho+xcomp+ycomp)/d);
               }       
            }
         }
      }
   }
   return true;
}

double Chebyshev_update_omega(bool firstiteration, double sigma_Jacobi, double omega) {
   if (firstiteration == true) {
      return 1.0/(1.0-sigma_Jacobi*sigma_Jacobi/2.0);
   }
   else {
      return 1.0/(1.0 - omega*sigma_Jacobi*sigma_Jacobi/4.0);
   }
}

double Poiseuille_calculate_directSimpson(gsl_matrix * weights, gsl_matrix * current, int Nx, int Ny, double dx, double dy, double eta, double area) {
   double sum = 0.0;

   for (int i=0; i<Nx+1;i++) {
      for (int j=0; j<Ny+1;j++) {
         sum += gsl_matrix_get(weights,i,j)*gsl_matrix_get(current,i,j);
      }
   }

   return 8.0*M_PI*eta*sum/(area*area*(Nx+1)*(Ny+1)*9.0);
}

bool set_profile_halfmoon(gsl_matrix_int * profile, int Nx, int Ny) {
   //needs to be square and divisible by 4, so:
   if (Nx != Ny || Nx%4 != 0) {
      return false;
   }
   int Nquadrant = Nx/4;
   for (int i=0; i<Nx+1;i++) {
      for (int j=0; j<Ny+1;j++) {
         if (i <= Nquadrant && j <= 3*Nquadrant) {gsl_matrix_int_set(profile,i,j,1);}
         if (i >= Nquadrant && j >= 3*Nquadrant) {gsl_matrix_int_set(profile,i,j,1);}
         if (i >= Nquadrant && i <= 2*Nquadrant && j<=Nquadrant && j+i <= 2*Nquadrant) {gsl_matrix_int_set(profile,i,j,1);}
         if (i <= Nquadrant && j >= 3*Nquadrant && j-i <= 3*Nquadrant) {gsl_matrix_int_set(profile,i,j,1);}
         if (i >= Nquadrant && i <= 2*Nquadrant && j >= 2*Nquadrant && j <= 3*Nquadrant && j-i >= Nquadrant) {gsl_matrix_int_set(profile,i,j,1);}
         if (i >= 3*Nquadrant && j >= 2*Nquadrant && j <= 3*Nquadrant && i+j >= 6*Nquadrant) {gsl_matrix_int_set(profile,i,j,1);}
      }
   }

   return true;
}

bool set_profile_halfmoon_noborder(gsl_matrix_int * profile, int Nx, int Ny) {
   //needs to be square and divisible by 4, so:
   if (Nx != Ny || Nx%4 != 0) {
      return false;
   }
   int Nquadrant = Nx/4;
   for (int i=0; i<Nx+1;i++) {
      for (int j=0; j<Ny+1;j++) {
         if (i > 0 && i < Nquadrant && j > 0 && j < 3*Nquadrant) {gsl_matrix_int_set(profile,i,j,1);}
         if (i > Nquadrant && i < 4*Nquadrant && j > 3*Nquadrant && j < 4*Nquadrant) {gsl_matrix_int_set(profile,i,j,1);}
         if (i >= Nquadrant && i <= 2*Nquadrant && j > 0 && j<=Nquadrant && j+i < 2*Nquadrant) {gsl_matrix_int_set(profile,i,j,1);}
         if (i > 0 && i <= Nquadrant && j >= 3*Nquadrant && j-i < 3*Nquadrant) {gsl_matrix_int_set(profile,i,j,1);}
         if (i >= Nquadrant && i <= 2*Nquadrant && j >= 2*Nquadrant && j <= 3*Nquadrant && j-i > Nquadrant) {gsl_matrix_int_set(profile,i,j,1);}
         if (i >= 3*Nquadrant && i < 4*Nquadrant && j >= 2*Nquadrant && j <= 3*Nquadrant && i+j > 6*Nquadrant) {gsl_matrix_int_set(profile,i,j,1);}
      }
   }

   return true;
}

bool set_profile_cross(gsl_matrix_int * profile, int Nx, int Ny) {
   if (Nx%3 != 0 || Ny%3 != 0) {
      return false;
   }
   int Nxquadrant = Nx/3;
   int Nyquadrant = Ny/3;
   for (int i=0; i<Nx+1;i++) {
      for (int j=0; j<Ny+1;j++) {
         if (i > Nxquadrant && i<2*Nxquadrant && j>0 && j<3*Nyquadrant) {gsl_matrix_int_set(profile,i,j,1);}
         if (i > 0 && i < 3*Nxquadrant && j>Nyquadrant && j<2*Nyquadrant) {gsl_matrix_int_set(profile,i,j,1);}
      }
   }
   return true;
}

bool set_profile_square(gsl_matrix_int * profile, int Nx, int Ny) {
   for (int i=1; i<Nx;i++) {
      for (int j=1; j<Ny;j++) {
         gsl_matrix_int_set(profile,i,j,1);
      }
   }
   return true;
}

double calculate_area(gsl_matrix_int * profile, int Nx, int Ny) {
   //relative area occupied by profile
   double sum = 0.0;
   for (int i = 1; i < Nx;i++) {
      for (int j = 1; j < Ny;j++) {
         if (gsl_matrix_int_get(profile,i,j) == 1) {
            sum += 1.0;
         }
      }
   }

   return sum/((Nx-1)*(Ny-1));
}

double calculate_change_magnitude(gsl_matrix_int * mask, gsl_matrix * previous, gsl_matrix * current, int Nx, int Ny) {
   //computes per-element absolute change in velocity from one step to the next 
   double npoints = calculate_area(mask,Nx,Ny)*(Nx-1)*(Ny-1);
   double change = 0.0;
   for (int i = 0; i < Nx+1;i++) {
      for (int j = 0; j < Ny+1;j++) {
         change += fabs(gsl_matrix_get(current,i,j)-gsl_matrix_get(previous,i,j));
      }
   }
   return change/npoints;
}

int main(void) {

   int Nx=100, Ny=100; //numbers of "cells" along each axis. The actual number of points is larger by 1. Needs to be even!
   double dx = 1.0/Nx, dy = 1.0/Ny;
   double d = 2.0*(1.0/(dx*dx) + 1.0/(dy*dy));
   double rho = 1.0; //density, right-hand side of Poisson equation
   double eta = 1.0; //viscosity
   double magnitudeerror = 1e-8;
   double coefficienterror = 1e-8;
   uint64_t prev_time_value, time_value;   //for timekeeping
   uint64_t time_diff;

   //assign matrices
   gsl_matrix * matrikacurrent = gsl_matrix_alloc(Nx+1, Ny+1);
   gsl_matrix * matrikanext = gsl_matrix_alloc(Nx+1, Ny+1);
   gsl_matrix * temporary = gsl_matrix_alloc(Nx+1, Ny+1);

   gsl_matrix_int * profile = gsl_matrix_int_calloc(Nx+1,Ny+1);

   //bool flag = set_profile_halfmoon_noborder(profile,Nx,Ny);
   bool flag = set_profile_square(profile,Nx,Ny);
   //bool flag = set_profile_cross(profile,Nx,Ny);
   if (flag == false) {
      printf("\nInvalid grid size");
      return 0;
   }
   double area = calculate_area(profile,Nx,Ny);

   double sigmaestimate = Jacobi_spectral_radius(Nx, Ny, dx, dy, d);
   printf("estimated spectral radius: %f",sigmaestimate);
   //double omega = 1.0;  //if not using SOR
   double omega = optimum_omega_calc(sigmaestimate);  //for SOR
   printf("\nestimated omega value: %f", omega);

   /*
   //prints the mask to console - use for verifying 
   for (int j=0; j<Ny+1;j++) {
      for (int i=0; i<Nx+1;i++) {
         printf("%d ",gsl_matrix_int_get(profile,i,j));
      }
      printf("\n");
   }
   */


   //this block computes the weighing matrix for flux calculation via Simpson method 
   gsl_matrix * weights = gsl_matrix_alloc(Nx+1,Ny+1);
   gsl_vector * weightshelperx = gsl_vector_alloc(Nx+1);
   gsl_vector * weightshelpery = gsl_vector_alloc(Ny+1);   
   gsl_matrix_set_all(weights, 1.0);
   gsl_vector_set_all(weightshelperx, 1.0);
   gsl_vector_set_all(weightshelpery, 1.0);
   for (int k = 1; k < Nx; k++) {
      if (k%2 != 0) {gsl_vector_set(weightshelperx,k,4.0);}
      else if (k%2 == 0) {gsl_vector_set(weightshelperx,k,2.0);}
   }
   for (int l = 1; l < Ny; l++) {
      if (l%2 != 0) {gsl_vector_set(weightshelpery,l,4.0);}
      else if (l%2 == 0) {gsl_vector_set(weightshelpery,l,2.0);}
   }
   gsl_blas_dger(1.0, weightshelperx, weightshelpery, weights);   //computes outer product of vectors
   gsl_vector_free(weightshelperx);
   gsl_vector_free(weightshelpery);

   //filling of matrices
   gsl_matrix_set_all(matrikacurrent, 1.0);  
   gsl_matrix_set_all(matrikanext, 1.0);
   gsl_matrix_set_all(temporary,1.0);   
   //boundary condition requires stationary liquid -> border (and outside) elements are zero
   for (int j=0;j<Nx+1;j++) {
      for (int k=0;k<Ny+1;k++) {
         if (gsl_matrix_int_get(profile,j,k) != 1) {
            gsl_matrix_set(matrikacurrent, j, k, 0.0);
            gsl_matrix_set(matrikanext, j, k, 0.0);
            gsl_matrix_set(temporary,j,k,0.0);
         }         
      }
   }


   char filepath[256];
   snprintf(filepath, sizeof(filepath), "Rezultati/square_SOR_Nx_%d_Ny_%d_err_%.0e_matrix.dat", Nx,Ny,coefficienterror);
   FILE * out_profile = fopen(filepath, "wt");  //for printing velocity profile to a "matrix" data file
   snprintf(filepath, sizeof(filepath), "Rezultati/square_SOR_Nx_%d_Ny_%d_err_%.0e_stats.dat", Nx,Ny,coefficienterror);
   FILE * out_stats = fopen(filepath,"wt");  //for tracking of changes over iterations
   fprintf(out_stats, "#n_iterations   vector_change  C(Poiseuille)  C_change   omega\n");
   FILE * times_output = fopen("Rezultati/SORChebyshev_times.dat", "a+");

   double magchange, C_current, C_previous = Poiseuille_calculate_directSimpson(weights, matrikanext, Nx, Ny, dx, dy, eta, area);
   
/*
   //Jacobi, Gauss-Seidel or regular SOR 
   for (int i=0;i<1e5;i++) {
      //masked_Jacobi_iteration(profile,matrikacurrent, matrikanext, Nx, Ny, dx, dy, rho, d);
      //masked_GaussSeidel_iteration(profile,matrikacurrent, matrikanext, Nx, Ny, dx, dy, rho, d);
      //masked_GaussSeidel_SOR_iteration(profile, matrikacurrent, matrikanext, temporary, Nx, Ny, dx, dy, rho, d, omega);

      magchange = calculate_change_magnitude(profile, matrikacurrent, matrikanext, Nx, Ny);
      C_current = Poiseuille_calculate_directSimpson(weights, matrikanext, Nx, Ny, dx, dy, eta, area);

      if (magchange < magnitudeerror && fabs(C_current-C_previous) < coefficienterror) { //print final state on convergence
         printf("\nSequence converged at iteration %d.",i);
         fprintf(out_stats, "%d %.15f %.10f %.10f %f\n",i,magchange,C_current,fabs(C_current-C_previous),omega);
         break;
      }
      if (i % 100 == 0) {  //print state every 100 iterations, regardless of quality
         printf("\nFinished iteration %d.",i);
         fprintf(out_stats, "%d %.15f %.10f %.10f %f\n",i,magchange,C_current,fabs(C_current-C_previous),omega);
      }
      gsl_matrix_memcpy(matrikacurrent, matrikanext);
      C_previous = C_current;
   }
*/
   prev_time_value = get_posix_clock_time ();
   //Chebyshev-accelerated SOR
   bool odd;
   bool firstiteration = true;
   for (int i=0;i<1e5;i++) {
      gsl_matrix_memcpy(temporary,matrikanext);
      odd = true;
      masked_Chebyshev_semiiteration(profile,matrikacurrent, matrikanext, Nx, Ny, dx, dy, rho, d, omega, odd);
      omega = Chebyshev_update_omega(firstiteration, sigmaestimate, omega);
      if (i == 0) {firstiteration = false;}
      gsl_matrix_memcpy(matrikacurrent, matrikanext);
      odd = false;
      masked_Chebyshev_semiiteration(profile,matrikacurrent, matrikanext, Nx, Ny, dx, dy, rho, d, omega, odd);
      omega = Chebyshev_update_omega(firstiteration, sigmaestimate, omega);
      gsl_matrix_memcpy(matrikacurrent, matrikanext);
      
      magchange = calculate_change_magnitude(profile, temporary, matrikanext, Nx, Ny);
      C_current = Poiseuille_calculate_directSimpson(weights, matrikanext, Nx, Ny, dx, dy, eta, area);
      if (magchange < magnitudeerror && fabs(C_current-C_previous) < coefficienterror) { //print final state on convergence
         printf("\nSequence converged at iteration %d.",i);
         fprintf(out_stats, "%d %.15f %.10f %.10f %f\n",i,magchange,C_current,fabs(C_current-C_previous),omega);
         break;
      }      
      if (i % 5 == 0) {
         printf("\nFinished iteration %d.",i);
         fprintf(out_stats, "%d %.15f %.10f %.10f %f\n",i,magchange,C_current,fabs(C_current-C_previous),omega);
      }
      C_previous = C_current;
   }

   time_value = get_posix_clock_time ();
   time_diff = time_value - prev_time_value;
   printf("\nExecution time was %f milliseconds.",time_diff/1000.0);
   fprintf(times_output,"%d %d %.10f %f\n",Nx,Ny,magnitudeerror,time_diff/1000.0);
   
   //prints current profile to file
   for (int i=0;i<Nx+1;i++) {
      for (int j=0;j<Ny+1;j++) {
         fprintf(out_profile, " %.10f",gsl_matrix_get(matrikanext,i,j));
      }
      fprintf(out_profile,"\n");
   }

/*

   //this is for combing the profile for optimum alpha/omega with SOR, but it doesn't work quite right

   char filepath[256];
   snprintf(filepath, sizeof(filepath), "Rezultati/crescent_SOR_Nx_%d_Ny_%d_err_%.0e_alphavar_fine.dat", Nx,Ny,coefficienterror);
   FILE * out_alphavar = fopen(filepath,"wt");  //for tracking over iterations
   fprintf(out_alphavar, "#alpha omega n_iter_converged\n");
   double alpha,omega, magchange, C_current, C_previous;

   for (alpha = 10.0; alpha >= 0.5; alpha -= 0.01) {
      omega = 2.0/(1.0+alpha*M_PI/(Nx+1));  //estimate, for alpha search loop
      if (omega >= 2.0 || omega <= 0.0) {break;}
      //filling of matrices
      gsl_matrix_set_all(matrikacurrent, 1.0);  
      gsl_matrix_set_all(matrikanext, 1.0);
      gsl_matrix_set_all(temporary,1.0);   
      //boundary condition requires stationary liquid -> border (and outside) elements are zero
      for (int j=0;j<Nx+1;j++) {
         for (int k=0;k<Ny+1;k++) {
            if (gsl_matrix_int_get(profile,j,k) != 1) {
               gsl_matrix_set(matrikacurrent, j, k, 0.0);
               gsl_matrix_set(matrikanext, j, k, 0.0);
               gsl_matrix_set(temporary,j,k,0.0);
            }         
         }
      }

      C_previous = Poiseuille_calculate_directSimpson(weights, matrikanext, Nx, Ny, dx, dy, eta, area);
      for (int i=0;i<1e4;i++) {
         masked_GaussSeidel_SOR_iteration(profile, matrikacurrent, matrikanext, temporary, Nx, Ny, dx, dy, rho, d, omega);

         magchange = calculate_change_magnitude(profile, matrikacurrent, matrikanext, Nx, Ny);
         C_current = Poiseuille_calculate_directSimpson(weights, matrikanext, Nx, Ny, dx, dy, eta, area);

         if (magchange < magnitudeerror && fabs(C_current-C_previous) < coefficienterror) { //print final state on convergence
            printf("\nalpha = %f Sequence converged at iteration %d.",alpha,i);
            fprintf(out_alphavar, "%f %f %d %.10f\n",alpha,omega,i,C_current);
            break;
         }
         if (i % 10000 == 0) {
            printf("\nFinished iteration %d.",i);
         }
         gsl_matrix_memcpy(matrikacurrent, matrikanext);
         C_previous = C_current;
      }  
   }
*/
   gsl_matrix_free(matrikanext);
   gsl_matrix_free(matrikacurrent);
   gsl_matrix_free(temporary);
   gsl_matrix_int_free(profile);
   fclose(out_profile);
   fclose(out_stats);
   fclose(times_output);
   //fclose(out_alphavar);
   return 0;
}