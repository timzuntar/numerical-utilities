//Computation of oscillation eigenmodes of weighed membrane with Jacobi and QR algorithms 

#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include <math.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_eigen.h>
#include <gsl/gsl_linalg.h>

bool print_matrix(gsl_matrix * in, int Nx, int Ny) {
    for (int i=0; i<Nx;i++) {
        for (int j=0; j<Ny;j++) {
        printf("%.0f ",gsl_matrix_get(in,i,j));
        }
        printf("\n");
    }

    return true;
}

bool fprint_matrix(FILE * output, gsl_matrix * in, int Nx, int Ny) {
    for (int j=0; j<Ny;j++) {
        for (int i=0; i<Nx;i++) {
        fprintf(output, "%f ",gsl_matrix_get(in,i,j));
        }
        fprintf(output, "\n");
    }

    return true;
}

bool fprint_matrix_pm3d(FILE * output, gsl_matrix * in, int Nx, int Ny) {
    for (int j=0; j<Ny;j++) {
        for (int i=0; i<Nx;i++) {
        fprintf(output, "%f %f %f\n",(double)i/(Ny-1),j*M_PI/(Nx-1),gsl_matrix_get(in,i,j));
        }
        fprintf(output, "\n");
    }

    return true;
}

double k_eval(double omega, double rho, double thickness, double gamma) {
    double k2 = omega*omega*rho*thickness/gamma;
    return sqrt(k2);
}

bool compute_nabla_squared(gsl_matrix * A, int Nx, int Ny, double dx, double dy) {
    //matrix for wave equation. Each dimension "holds" Ny blocks of size Nx
    int dim = (Nx-1)*(Ny-1);

    gsl_matrix_set_all(A, 0.0); //initialize values

    for (int i=0; i<dim;i++) {
        for (int j=0; j<dim;j++) {
            if (i==j) {gsl_matrix_set(A, i,j, 2.0/(dx*dx) + 2.0/(dy*dy));}
            if (j==i+1 && j%(Nx-1)!=0) {gsl_matrix_set(A,i,j,-1.0/(dx*dx));}
            if (j==i-1 && i%(Nx-1)!=0) {gsl_matrix_set(A,i,j,-1.0/(dx*dx));}
            if (j==i+(Nx-1)) {gsl_matrix_set(A,i,j,-1.0/(dy*dy));}
            if (j==i-(Nx-1)) {gsl_matrix_set(A,i,j,-1.0/(dy*dy));}
        }
    }
    return true;
}

bool compute_nabla_squared_cylindrical(gsl_matrix * A, int Nphi, int Nr, double dphi, double dr) {
    //matrix for wave equation in cylindrical coordinates (r,phi). Each dimension "holds" Nr blocks of size Nphi
    int dim = (Nr-1)*(Nphi-1);

    gsl_matrix_set_all(A, 0.0); //initialize values

    for (int i=0; i<dim;i++) {
        for (int j=0; j<dim;j++) {
            if (i==j) {gsl_matrix_set(A, i,j, 2.0/(dr*dr) + 2.0/(dr*dr*dphi*dphi*((i%(Nphi-1))+1)*((i%(Nphi-1))+1)));}
            if (j==i+1 && j%(Nphi-1)!=0) {gsl_matrix_set(A,i,j,-1.0/(dr*dr*dphi*dphi*((i%(Nphi-1))+1)*((i%(Nphi-1))+1)));}
            if (j==i-1 && i%(Nphi-1)!=0) {gsl_matrix_set(A,i,j,-1.0/(dr*dr*dphi*dphi*((i%(Nphi-1))+1)*((i%(Nphi-1)+1))));}
            if (j==i+(Nphi-1)) {gsl_matrix_set(A,i,j,-1.0/(dr*dr)-0.5/(dr*dr*((i%(Nphi-1))+1)));}
            if (j==i-(Nphi-1)) {gsl_matrix_set(A,i,j,-1.0/(dr*dr)+0.5/(dr*dr*((i%(Nphi-1))+1)));}
        }
    }
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

   //we don't need the outer edges.
   for (int i=0; i<Nx-1;i++) {
      for (int j=0; j<Ny-1;j++) {
          gsl_matrix_set(profile,i,j,gsl_matrix_get(placeholder,i+1,j+1));
      }
   }

   gsl_matrix_free(placeholder);
   return true;
}

bool vectorize_matrix(gsl_matrix * state, gsl_vector * u, int Nx, int Ny) {
    for (int i=0;i<Ny-1;i++) {
        for (int j=0;j<Nx-1;j++) {
            gsl_vector_set(u,j+i*(Nx-1),gsl_matrix_get(state,i,j));
        }
    }
    return true;
}

bool reconstruct_matrix(gsl_matrix * state, gsl_vector * u, int Nx, int Ny) {
    for (int i=0;i<(Nx-1)*(Ny-1);i++) {
        gsl_matrix_set(state,i/(Nx-1),i%(Nx-1),gsl_vector_get(u,i));
    }

    return true;
}

bool diagonal_from_vector(gsl_matrix * B, gsl_matrix * profile, int Nx, int Ny) {
    for (int i=0;i<Ny-1;i++) {
        for (int j=0;j<Nx-1;j++) {
            gsl_matrix_set(B,j+i*(Nx-1),j+i*(Nx-1),gsl_matrix_get(profile,i,j));
        }
    }

    return true;
}

double Jacobi_step(gsl_matrix * C, gsl_matrix * G, gsl_matrix * GT, gsl_matrix * J, gsl_matrix * placeholder, int Nx, int Ny) {
    //finds the largest off-diagonal element and zeroes it out with a Givens rotation, then 
    double mag_offdiagonal = 0.0;
    int m,n;    //pivot row/column
    for (int i=0;i<(Ny-1)*(Nx-1);i++) {
        for (int j=0;j<(Nx-1)*(Ny-1);j++) {
            if (i!=j && (fabs(gsl_matrix_get(C,i,j))>mag_offdiagonal)) {
                mag_offdiagonal = fabs(gsl_matrix_get(C,i,j));
                m=i;
                n=j;
            }
        }
    }
    printf("\nm=%d n=%d",m,n);
    //compute rotation matrix
    double theta = 0.5*(M_PI/2 - atan((gsl_matrix_get(C,m,m)-gsl_matrix_get(C,n,n))/(2.0*gsl_matrix_get(C,m,n))));
    gsl_matrix_set_identity(G);
    gsl_matrix_set(G,m,m,cos(theta));
    gsl_matrix_set(G,n,n,cos(theta));
    gsl_matrix_set(G,m,n,-sin(theta));
    gsl_matrix_set(G,n,m,sin(theta));
    
    gsl_matrix_transpose_memcpy(GT, G); //transpose G
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, J, G, 0.0, placeholder);  //update full Jacobi matrix by right-multiplying with G-transpose
    gsl_matrix_memcpy(J,placeholder);
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, C, G, 0.0, placeholder);
    gsl_matrix_memcpy(C,placeholder);
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, GT, C, 0.0, placeholder);  //and update C
    gsl_matrix_memcpy(C,placeholder);

    return mag_offdiagonal;
}
/*
bool square_to_banded(gsl_matrix * A, gsl_matrix * Abanded, int Nx, int Ny) {
    if (Ny != Nx) {return false;}
    for (int c=0;c<Nx;c++) {
        for (int r=0;r<(Nx-1)*(Ny-1);r++) {
            if (r+c < (Nx-1)*(Ny-1)) {
                gsl_matrix_set(Abanded,r,c,gsl_matrix_get(A,r,r+c));
            }
        }
    }
    return true;
}

bool inverse_power_step(gsl_matrix * A, gsl_matrix * Abanded, gsl_vector * y, gsl_vector * x, int Nx, int Ny) {
    //matrix needs to be square for that. Banded decomposition for GSL 2.7+
    if (Ny != Nx) {return false;}
    int flag;
    square_to_banded(A,Abanded,Nx,Ny);
    flag = gsl_linalg_ldlt_band_decomp(Abanded);
    flag = gsl_linalg_ldlt_band_solve(Abanded,x,y); //the solution vector is y

    double ynorm = gsl_blas_dnrm2(y);
    gsl_vector_scale(y, 1.0/ynorm); //normalized y is now x for next step
    gsl_vector_memcpy(x,y); //overwrites approximation with next iteration

    return true;
}
*/
bool inverse_power_step_QR(gsl_matrix * A, gsl_vector * tau, gsl_vector * y, gsl_vector * x, int Nx, int Ny) {
    //matrix needs to be square for that. Banded decomposition for GSL 2.7+
    if (Ny != Nx) {return false;}
    int flag;
    flag = gsl_linalg_QR_decomp(A,tau);
    flag = gsl_linalg_QR_solve(A,tau,x,y);    //the solution vector is y

    double ynorm = gsl_blas_dnrm2(y);
    gsl_vector_scale(y, 1.0/ynorm); //normalized y is now x for next step
    gsl_vector_memcpy(x,y); //overwrites x with next iteration

    return true;
}

int main(void) {
    /*
    int Nx=10, Ny=10;
    double dx=1.0/Nx, dy=1.0/Ny;
    double beta = 1.0;  //relative density parameter; for beta=1 the membrane is homogeneous

    double mag_offdiag = 0.0;
    double epsilon = 1e-8;

    char filepath[256];
    
    gsl_matrix * uprofile = gsl_matrix_alloc(Ny-1,Nx-1);
    gsl_matrix * rhoprofile = gsl_matrix_alloc(Ny-1,Nx-1);
    gsl_vector * u = gsl_vector_alloc((Nx-1)*(Ny-1));
    gsl_vector * v = gsl_vector_alloc((Nx-1)*(Ny-1));
    gsl_vector * lambda = gsl_vector_alloc((Nx-1)*(Ny-1));
    gsl_matrix * A = gsl_matrix_alloc((Nx-1)*(Ny-1),(Nx-1)*(Ny-1));
    gsl_matrix * B = gsl_matrix_alloc((Nx-1)*(Ny-1),(Nx-1)*(Ny-1));
    gsl_matrix * C = gsl_matrix_alloc((Nx-1)*(Ny-1),(Nx-1)*(Ny-1));
    
    set_profile_Q(rhoprofile, Nx, Ny, beta);   //populate density and system matrices
    compute_nabla_squared(A, Nx, Ny, dx, dy);

    gsl_matrix_set_all(B,0.0);
    diagonal_from_vector(B, rhoprofile, Nx, Ny);    //rewrite density matrix in expanded form

    for (int i=0;i<(Nx-1)*(Ny-1);i++) {
        for (int j=0;j<(Nx-1)*(Ny-1);j++) {
            gsl_matrix_set(C,i,j,gsl_matrix_get(A,i,j)/sqrt(gsl_matrix_get(B,i,i)*gsl_matrix_get(B,j,j)));
        }
    }
    for (int k=0;k<(Nx-1)*(Ny-1);k++) {
        gsl_vector_set(v,k,sqrt(gsl_matrix_get(B,k,k))*gsl_vector_get(u,k));
    }

    //print_matrix(rhoprofile,Nx-1,Ny-1);
    //print_matrix(C,(Nx-1)*(Ny-1),(Ny-1)*(Nx-1));
    
    
    //Jacobi iteration

    gsl_matrix * G = gsl_matrix_alloc((Nx-1)*(Ny-1),(Nx-1)*(Ny-1));
    gsl_matrix * GT = gsl_matrix_alloc((Nx-1)*(Ny-1),(Nx-1)*(Ny-1));
    gsl_matrix * J = gsl_matrix_alloc((Nx-1)*(Ny-1),(Nx-1)*(Ny-1));
    gsl_matrix * placeholder = gsl_matrix_alloc((Nx-1)*(Ny-1),(Nx-1)*(Ny-1));

    gsl_matrix_set_identity(J);

    int x,reps_per_element = 100;
    for (x=0;x<reps_per_element*(Nx-1)*(Ny-1)*(Nx-1)*(Ny-1);x++) {
        mag_offdiag = Jacobi_step(C, G, GT, J, placeholder, Nx, Ny);
        printf("Step %d, max. off-diagonal value: %f\n", x, mag_offdiag);
        if (mag_offdiag < epsilon) {break;}
    }

    //print_matrix(C,(Nx-1)*(Ny-1),(Ny-1)*(Nx-1));

    for (int y=0;y<(Nx-1)*(Ny-1);y++) {
        gsl_vector_set(lambda,y,gsl_matrix_get(C,y,y));
    }
    gsl_eigen_symmv_sort(lambda, J, GSL_EIGEN_SORT_ABS_ASC);
    //we now have sorted eigenvalues and eigenvectors.
    for (int k=0;k<10;k++) {
        printf("%.2f ",gsl_vector_get(lambda,k)/(M_PI*M_PI));
    }

    snprintf(filepath, sizeof(filepath), "Rezultati/beta_05_Jacobi_N_%d_iter_%d_maxmag_%.3e_lambda.dat", Nx,x,mag_offdiag);
    FILE * out_lambda = fopen(filepath, "wt");  //for printing velocity profile to a "matrix" data file
    for (int k=0;k<(Nx-1)*(Ny-1);k++) {
        fprintf(out_lambda,"%f %f\n",gsl_vector_get(lambda,k), gsl_vector_get(lambda,k)/(M_PI*M_PI));
    }
    fclose(out_lambda);
    
    for (int column=0; column<0.1*(Nx-1)*(Ny-1);column++) {
        gsl_matrix_get_col(v, J, column);   //grabs an eigenvector from the matrix
        for (int k=0;k<(Nx-1)*(Ny-1);k++) {
            gsl_vector_set(v,k,gsl_vector_get(v,k)/sqrt(gsl_matrix_get(B,k,k)));
        }
        
        reconstruct_matrix(uprofile,v,Ny,Nx);
        snprintf(filepath, sizeof(filepath), "Rezultati/beta_05_Jacobi_N_%d_iter_%d_maxmag_%.3e_eigen_%d.dat", Nx,x,mag_offdiag,column);
        FILE * out_matrix = fopen(filepath, "wt");
        
        fprint_matrix(out_matrix,uprofile,Ny-1,Nx-1);
        
        fclose(out_matrix);
    }

    gsl_matrix_free(G);
    gsl_matrix_free(GT);
    gsl_matrix_free(J);
    gsl_matrix_free(placeholder);
    */
    /*
    //Inverse power method
    double sigma = 5*M_PI*M_PI;
    double lambda_result;
    gsl_matrix * Areduced = gsl_matrix_alloc((Nx-1)*(Ny-1),(Nx-1)*(Ny-1));
    gsl_matrix * identity = gsl_matrix_alloc((Nx-1)*(Ny-1),(Nx-1)*(Ny-1));
    gsl_vector * yi = gsl_vector_alloc((Nx-1)*(Ny-1));
    gsl_vector * tau = gsl_vector_alloc((Nx-1)*(Ny-1));
    for (int y=0;y<(Nx-1)*(Ny-1);y++) {
        gsl_vector_set(yi,y,(double)rand()/RAND_MAX-0.5);
        //printf("%.3f",gsl_vector_get(yi,y));
    }
    gsl_matrix_memcpy(Areduced,C);
    gsl_matrix_set_identity(identity);
    gsl_matrix_scale(identity,sigma);
    gsl_matrix_sub(Areduced,identity);

    int reps_per_element = 1;
    for (int x=0;x<reps_per_element*(Nx-1)*(Ny-1)*(Nx-1)*(Ny-1);x++) {
        inverse_power_step_QR(Areduced, tau, v, yi, Nx, Ny);
    }
    //compute eigenvalues at the end
    gsl_blas_dgemv(CblasNoTrans, 1.0, C, yi, 0.0, lambda);
    gsl_blas_ddot(v,lambda,&lambda_result);
    printf("\nsigma=%.2f,lambda=%.5f",sigma,lambda_result/(M_PI*M_PI));

    reconstruct_matrix(uprofile,v,Ny,Nx);
    snprintf(filepath, sizeof(filepath), "Rezultati/homogeneous_inversepot_N_%d_reps_%d_sigma_%.2f.dat", Nx,reps_per_element,sigma);
    FILE * out_matrix = fopen(filepath, "wt");
    fprint_matrix(out_matrix,uprofile,Ny-1,Nx-1);
    fclose(out_matrix);

    gsl_vector_free(yi);
    gsl_vector_free(tau);

    gsl_matrix_free(uprofile);
    gsl_matrix_free(rhoprofile);
    gsl_vector_free(u);
    gsl_vector_free(v);
    gsl_vector_free(lambda);
    gsl_matrix_free(A);
    gsl_matrix_free(B);
    gsl_matrix_free(C);
    */
    /*
    //QR method
    gsl_vector *eval = gsl_vector_alloc((Nx-1)*(Ny-1));
    gsl_matrix *evec = gsl_matrix_alloc((Nx-1)*(Ny-1),(Nx-1)*(Ny-1));
    gsl_eigen_symmv_workspace * workspace = gsl_eigen_symmv_alloc((Nx-1)*(Ny-1));
    gsl_eigen_symmv (C, eval, evec, workspace);
    gsl_eigen_symmv_free (workspace);
    gsl_eigen_symmv_sort (eval,evec,GSL_EIGEN_SORT_ABS_ASC);

    snprintf(filepath, sizeof(filepath), "Rezultati/beta_10_QR_N_%d_lambda.dat", Nx);
    FILE * out_lambda = fopen(filepath, "wt");  //for printing velocity profile to a "matrix" data file
    for (int k=0;k<(Nx-1)*(Ny-1);k++) {
        fprintf(out_lambda,"%f %f\n",gsl_vector_get(eval,k), gsl_vector_get(eval,k)/(M_PI*M_PI));
    }
    fclose(out_lambda);

    for (int column=0; column<0.1*(Nx-1)*(Ny-1);column++) {
        gsl_matrix_get_col(v, evec, column);   //grabs an eigenvector from the matrix
        for (int k=0;k<(Nx-1)*(Ny-1);k++) {
            gsl_vector_set(v,k,gsl_vector_get(v,k)/sqrt(gsl_matrix_get(B,k,k)));
        }
        
        reconstruct_matrix(uprofile,v,Ny,Nx);
        snprintf(filepath, sizeof(filepath), "Rezultati/beta_10_QR_N_%d_eigen_%d.dat", Nx,column);
        FILE * out_matrix = fopen(filepath, "wt");
        
        fprint_matrix(out_matrix,uprofile,Ny-1,Nx-1);
        
        fclose(out_matrix);
    }

    gsl_matrix_free(evec);
    gsl_vector_free(eval);
    */
    
    /////////////////////////////////////
    //Cylindrical geometry
    int Nr=30, Nphi=30;
    double dr=1.0/Nr, dphi=M_PI/Nphi;
    char filepath[256];

    gsl_matrix * C = gsl_matrix_alloc((Nr-1)*(Nphi-1),(Nr-1)*(Nphi-1));
    compute_nabla_squared_cylindrical(C, Nphi, Nr, dphi, dr);

    /*
    //for Jacobi:
    gsl_vector * lambda = gsl_vector_alloc((Nr-1)*(Nphi-1));
    gsl_vector * v = gsl_vector_alloc((Nr-1)*(Nphi-1));
    gsl_matrix * uprofile = gsl_matrix_alloc((Nr-1),(Nphi-1));
    gsl_matrix * G = gsl_matrix_alloc((Nr-1)*(Nphi-1),(Nr-1)*(Nphi-1));
    gsl_matrix * GT = gsl_matrix_alloc((Nr-1)*(Nphi-1),(Nr-1)*(Nphi-1));
    gsl_matrix * J = gsl_matrix_alloc((Nr-1)*(Nphi-1),(Nr-1)*(Nphi-1));
    gsl_matrix * placeholder = gsl_matrix_alloc((Nr-1)*(Nphi-1),(Nr-1)*(Nphi-1));

    gsl_matrix_set_identity(J);
    int x,reps_per_element = 3;
    double mag_offdiag = 0.0;
    double epsilon = 1e-10;
    for (x=0;x<reps_per_element*(Nr-1)*(Nr-1)*(Nphi-1)*(Nphi-1);x++) {
        mag_offdiag = Jacobi_step(C, G, GT, J, placeholder, Nr, Nphi);
        printf("Step %d, max. off-diagonal value: %f\n", x, mag_offdiag);
        if (mag_offdiag < epsilon) {break;}
    }
    for (int y=0;y<(Nr-1)*(Nphi-1);y++) {
        gsl_vector_set(lambda,y,gsl_matrix_get(C,y,y));
    }
    gsl_eigen_symmv_sort(lambda, J, GSL_EIGEN_SORT_ABS_ASC);
    //we now have sorted eigenvalues and eigenvectors.
    for (int k=0;k<10;k++) {
        printf("%.2f ",gsl_vector_get(lambda,k));
    }
    snprintf(filepath, sizeof(filepath), "Rezultati/cylinder_Jacobi_N_%d_iter_%d_maxmag_%.3e_lambda.dat", Nr,x,mag_offdiag);
    FILE * out_lambda = fopen(filepath, "wt");  //for printing velocity profile to a "matrix" data file
    for (int k=0;k<(Nr-1)*(Nphi-1);k++) {
        fprintf(out_lambda,"%f %f\n",gsl_vector_get(lambda,k), sqrt(gsl_vector_get(lambda,k)));
    }
    fclose(out_lambda);
    for (int column=0; column<0.1*(Nr-1)*(Nphi-1);column++) {
        gsl_matrix_get_col(v, J, column);   //grabs an eigenvector from the matrix
        
        reconstruct_matrix(uprofile,v,Nr,Nphi);
        snprintf(filepath, sizeof(filepath), "Rezultati/cylinder_Jacobi_N_%d_iter_%d_maxmag_%.3e_eigen_%d.dat", Nr,x,mag_offdiag,column);
        FILE * out_matrix = fopen(filepath, "wt");
        
        fprint_matrix(out_matrix,uprofile,Nr-1,Nphi-1);
        
        fclose(out_matrix);
    }

    gsl_matrix_free(C);
    gsl_matrix_free(uprofile);
    gsl_matrix_free(G);
    gsl_matrix_free(GT);
    gsl_matrix_free(J);
    gsl_matrix_free(placeholder);
    gsl_vector_free(lambda);
    gsl_vector_free(v);
    */
   
    //QR
    gsl_vector * lambda = gsl_vector_alloc((Nr-1)*(Nphi-1));
    gsl_vector_complex * v = gsl_vector_complex_alloc((Nr-1)*(Nphi-1));
    gsl_matrix * uprofile = gsl_matrix_alloc(Nr-1,Nphi-1);
    gsl_matrix * B = gsl_matrix_alloc((Nr-1)*(Nphi-1),(Nr-1)*(Nphi-1));
    gsl_vector_complex *eval = gsl_vector_complex_alloc((Nr-1)*(Nphi-1));
    gsl_matrix_complex *evec = gsl_matrix_complex_alloc((Nr-1)*(Nphi-1),(Nr-1)*(Nphi-1));
    gsl_eigen_nonsymmv_workspace * workspace = gsl_eigen_nonsymmv_alloc((Nr-1)*(Nphi-1));
    gsl_eigen_nonsymmv (C, eval, evec, workspace);
    gsl_eigen_nonsymmv_free (workspace);
    gsl_eigen_nonsymmv_sort (eval,evec,GSL_EIGEN_SORT_ABS_ASC);

    snprintf(filepath, sizeof(filepath), "Rezultati/cylinder_QR_N_%d_lambda.dat", Nr);
    FILE * out_lambda = fopen(filepath, "wt");  //for printing velocity profile to a "matrix" data file
    for (int k=0;k<(Nr-1)*(Nphi-1);k++) {
        fprintf(out_lambda,"%f %f\n",GSL_REAL(gsl_vector_complex_get(eval,k)), sqrt(fabs(GSL_REAL(gsl_vector_complex_get(eval,k)))));
    }
    fclose(out_lambda);

    //for (int column=0; column<0.1*(Nr-1)*(Nphi-1);column++) {
    for (int column=0; column<25;column++) {
        gsl_matrix_complex_get_col(v, evec, column);   //grabs an eigenvector from the matrix
        for (int i=0;i<(Nr-1)*(Nphi-1);i++) {
            gsl_vector_set(lambda,i,GSL_REAL(gsl_vector_complex_get(v,i)));
        }

        reconstruct_matrix(uprofile,lambda,Nphi,Nr);
        snprintf(filepath, sizeof(filepath), "Rezultati/cylinder_QR_N_%d_eigen_%d.dat", Nr,column);
        FILE * out_matrix = fopen(filepath, "wt");
        
        fprint_matrix_pm3d(out_matrix,uprofile,Nphi-1,Nr-1);
        
        fclose(out_matrix);
    }
    gsl_matrix_complex_free(evec);
    gsl_vector_complex_free(eval);
    
    return 0;
}