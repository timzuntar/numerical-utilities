#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>
#include <sys/types.h>
#include <sys/stat.h>

#include <fftw3.h>  //for solving Poisson's equation using FFT

#include <gsl/gsl_vector.h> //more efficient matrix operations
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_complex.h>
#include <gsl/gsl_complex_math.h>

double timestep_eval(gsl_matrix * u, gsl_matrix * v, int Nx, int Ny, double dx, double dy){
    //adjusts the timestep to comply with Courant's condition (with some padding, hence the 2x factor)

    double umax = 0.0, vmax = 0.0;
    double element_u,element_v;

    for (int i = 0; i < Ny; i++) {
        for (int j = 0; j < Nx; j++) {
            element_u = fabs(gsl_matrix_get(u,i,j));
            element_v = fabs(gsl_matrix_get(v,i,j));
            if (element_u > umax) {umax = element_u;}
            if (element_v > vmax) {vmax = element_v;}
        }
    }
    double tdnewu = 0.4*dx/umax;
    double tdnewv = 0.4*dy/vmax;

    printf("\nvmax = %f, umax = %f",vmax,umax);

    if (tdnewv < tdnewu) {return tdnewv/2.0;}
    else {return tdnewu/2.0;}
}

bool vorticity_eval(gsl_matrix * u, gsl_matrix * v, gsl_matrix * psi, gsl_matrix * vorticity, int Nx, int Ny, double dx, double dy) {
    //calculates z-component of vorticity (which is the only one in a 2D system)
    double dudy, dvdx;

    for (int i = 1; i < Ny-1; i++) {
        for (int j = 1; j < Nx-1; j++) {
            dudy = (gsl_matrix_get(u,i,j+1) - gsl_matrix_get(u,i,j-1) )/(2.0*dy);  //symmetric differences
            dvdx = (gsl_matrix_get(v,i+1,j) - gsl_matrix_get(u,i-1,j) )/(2.0*dx);
            gsl_matrix_set(vorticity, i,j,dudy-dvdx);
        }
    }
    //Now for the edges.
    for (int j = 1; j < Ny-1; j++) {
        gsl_matrix_set(vorticity, j, 0, 2.0*gsl_matrix_get(psi, j, 1)/(dx*dx)); //left
        gsl_matrix_set(vorticity, j, Nx-1, 2.0*gsl_matrix_get(psi, j, Nx-2)/(dx*dx));  //right
    }
    for (int i = 0; i < Nx; i++) {
        gsl_matrix_set(vorticity, Ny-1, i, 2.0*gsl_matrix_get(psi, Ny-2, i)/(dy*dy)); //bottom
        gsl_matrix_set(vorticity, 0, i, 2.0*(gsl_matrix_get(psi, 1, i)-1.0*dx)/(dy*dy)); //top (lid)
    }
    return true;
}

bool vorticity_step(gsl_matrix * u, gsl_matrix * v, gsl_matrix * psi, gsl_matrix * vorticity, gsl_matrix * vorticity_next, int Nx, int Ny, double dx, double dy, double dt, double Re) {
    //calculates vorticity at the next timepoint in the interior

    double uxi, vxi, laplacian1, laplacian2;    //separate components to increase readability

    for (int i = 1; i < Ny-1; i++) {
        for (int j = 1; j < Nx-1; j++) {
            uxi = (gsl_matrix_get(u,i,j+1)*gsl_matrix_get(vorticity,i,j+1) - gsl_matrix_get(vorticity,i,j-1)*gsl_matrix_get(u,i,j-1))/dx;
            vxi = (gsl_matrix_get(v,i+1,j)*gsl_matrix_get(vorticity,i+1,j) - gsl_matrix_get(vorticity,i-1,j)*gsl_matrix_get(v,i-1,j))/dy;
            laplacian1 = (gsl_matrix_get(vorticity,i,j+1)+gsl_matrix_get(vorticity,i,j-1)-2.0*gsl_matrix_get(vorticity,i,j))/(dx*dx);
            laplacian2 = (gsl_matrix_get(vorticity,i+1,j)+gsl_matrix_get(vorticity,i-1,j)-2.0*gsl_matrix_get(vorticity,i,j))/(dy*dy);

            gsl_matrix_set(vorticity_next,i,j,gsl_matrix_get(vorticity,i,j) - (uxi+vxi)*dt*0.5 + dt*(laplacian1+laplacian2)/Re);
        }
    }
    //don't update edges!
    return true;
}

bool vorticity_step_edges(gsl_matrix * psi, gsl_matrix * vorticity_next, int Nx, int Ny, double dx, double dy) {
    //calculates boundary vorticity
    for (int j = 1; j < Ny-1; j++) {
        gsl_matrix_set(vorticity_next, j, 0, 2.0*gsl_matrix_get(psi, j, 1)/(dx*dx)); //left
        gsl_matrix_set(vorticity_next, j, Nx-1, 2.0*gsl_matrix_get(psi, j, Nx-2)/(dx*dx));  //right
    }
    for (int i = 0; i < Nx; i++) {
        gsl_matrix_set(vorticity_next, Ny-1, i, 2.0*gsl_matrix_get(psi, Ny-2, i)/(dy*dy)); //bottom
        gsl_matrix_set(vorticity_next, 0, i, 2.0*(gsl_matrix_get(psi, 1, i)-1.0*dx)/(dy*dy)); //top (lid)
    }
    return true;
}

bool psi_force_update_edges(gsl_matrix * psi, int Nx, int Ny) {
    //manually sets the edge values of psi to 0...just in case
    for (int i = 0; i < Ny; i++) {
        gsl_matrix_set(psi,i,Nx-1,0.0);   //right
        gsl_matrix_set(psi,i,0,0.0);
    }
    for (int j = 0; j < Nx; j++) {
        gsl_matrix_set(psi,Ny-1,j,0.0);
        gsl_matrix_set(psi,0,j,0.0);
    }
    return true;
}

bool velocity_eval(gsl_matrix * u, gsl_matrix * v, gsl_matrix * psi, int Nx, int Ny, double dx, double dy) {
    //calculates velocities from field
    for (int i = 1; i < Ny-1; i++) {
        for (int j = 1; j < Nx-1; j++) {
            gsl_matrix_set(u,i,j,(gsl_matrix_get(psi,i+1,j)-gsl_matrix_get(psi,i-1,j))/(2.0*dy) );  //again, symmetric differences
            gsl_matrix_set(v,i,j,(gsl_matrix_get(psi,i,j-1)-gsl_matrix_get(psi,i,j+1))/(2.0*dx) );
        }
    }
    //separately for edges, to take care of boundary conditions:
    for (int i = 1; i < Ny-1; i++) {
        gsl_matrix_set(u,i,Nx-1,0.0);   //right
        gsl_matrix_set(v,i,Nx-1,0.0);

        gsl_matrix_set(u,i,0,0.0);  //left
        gsl_matrix_set(v,i,0,0.0);
    }
    for (int j = 0; j < Nx; j++) {
        gsl_matrix_set(u,Ny-1,j,0.0);   //bottom
        gsl_matrix_set(v,Ny-1,j,0.0);

        gsl_matrix_set(u,0,j,1.0);  //lid
        gsl_matrix_set(v,0,j,0.0);
    }

    return true;
}

bool odd_extension(gsl_matrix * psi, gsl_matrix_complex * poisson_in, int Nx, int Ny) {
    //extends the matrix; needed for FFT operations
    for (int i = 1; i < Ny-1; i++) {
        for (int j = 1; j < Nx-1; j++) {
            gsl_matrix_complex_set(poisson_in, i-1, j+Nx-3, gsl_complex_rect(gsl_matrix_get(psi,i,j),0.0)); //right upper quadrant
            gsl_matrix_complex_set(poisson_in, i-1, j-1, gsl_complex_rect(-gsl_matrix_get(psi,i,Nx-j-1),0.0));    //left upper
            gsl_matrix_complex_set(poisson_in, i+Ny-3, j-1, gsl_complex_rect(gsl_matrix_get(psi,Ny-i-1,Nx-j-1),0.0)); //left lower
            gsl_matrix_complex_set(poisson_in, i+Ny-3, j+Nx-3, gsl_complex_rect(-gsl_matrix_get(psi,Ny-i-1,j),0.0)); //right lower
        }
    }
    return true;
}

bool poisson_solver(gsl_matrix * psi, gsl_matrix * vorticity, gsl_matrix_complex * poisson_psi, gsl_matrix_complex * poisson_vorticity, int Nx, int Ny, double dx) {
    //solves the poisson equation (laplacian of psi = vorticity)
    gsl_complex value = gsl_complex_rect(0.0,0.0);

    odd_extension(psi, poisson_psi, Nx, Ny);
    odd_extension(vorticity, poisson_vorticity, Nx, Ny);

	fftw_complex* in_psi = (fftw_complex*) poisson_psi->data;
	fftw_complex* in_vort = (fftw_complex*) poisson_vorticity->data;

    fftw_plan psiforward = fftw_plan_dft_2d(2*(Ny-2), 2*(Nx-2), in_psi, in_psi, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_plan vortforward = fftw_plan_dft_2d(2*(Ny-2), 2*(Nx-2), in_vort, in_vort, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_plan psibackward = fftw_plan_dft_2d(2*(Ny-2), 2*(Nx-2), in_psi, in_psi, FFTW_BACKWARD, FFTW_ESTIMATE);

    fftw_execute(psiforward);
    fftw_execute(vortforward);

    for (int i = 0; i < 2*(Ny-2); i++) {
        for (int j = 0; j < 2*(Nx-2); j++) {
            //adjust coefficients
            //value = gsl_complex_mul_real(gsl_matrix_complex_get(poisson_vorticity,i,j), 0.5*dx*dx/(cos(2*M_PI*i/Ny) + cos(2*M_PI*j/Nx)-2.0) );
            value = gsl_complex_mul_real(gsl_matrix_complex_get(poisson_vorticity,i,j), 0.5*dx*dx/(cos(M_PI*i/(Ny-2)) + cos(M_PI*j/(Nx-2))-1.999999999999) );
            gsl_matrix_complex_set(poisson_psi,i,j,value);
        }
    }
    gsl_matrix_complex_scale(poisson_psi, gsl_complex_rect(1.0/(4*(Nx-2)*(Ny-2)),0));

    fftw_execute(psibackward);  //after the reverse transform, poisson_psi contains the coefficients we seek

    for (int i = 0; i < Ny-2; i++) {
        for (int j = 0; j < Nx-2; j++) {
            gsl_matrix_set(psi, i+1,j+1, GSL_REAL(gsl_matrix_complex_get(poisson_psi,i,j+Nx-2)));
            //gsl_matrix_set(psi, i+1,j+1, GSL_REAL(gsl_matrix_complex_get(poisson_psi,i,j)));
        }
    }  

    fftw_destroy_plan(psiforward);  //cleanup
    fftw_destroy_plan(psibackward);
    fftw_destroy_plan(vortforward);

    return true;
}

double vorticity_change(gsl_matrix * vorticity, gsl_matrix * vorticity_next, int Nx, int Ny) {
    //calculates sum of per-element changes to vorticity from one time step to the next 
    double discrepancy = 0.0;

    for (int i = 0; i < Ny; i++) {
        for (int j = 0; j < Nx; j++) {
            discrepancy += fabs(gsl_matrix_get(vorticity_next,i,j)-gsl_matrix_get(vorticity,i,j));
        }
    }
    return discrepancy;
}

double force_on_lid(gsl_matrix * vorticity, double Nx, double dx, double Re) {
    double force = 0.0;

    force += (0.5*dx/Re)*(gsl_matrix_get(vorticity,0,0) + gsl_matrix_get(vorticity,0,Nx-1));

    for (int j = 1; j < Nx-1; j++) {
        force += (dx/Re)*gsl_matrix_get(vorticity, 0, j);
    }

    return force;
}

bool force_vec_output(gsl_matrix * vorticity, gsl_vector * force, double Nx, double dx, double Re) {
    gsl_vector_set(force, 0, (0.5*dx/Re)*(gsl_matrix_get(vorticity,0,0)));
    gsl_vector_set(force, Nx-1,(0.5*dx/Re)*(gsl_matrix_get(vorticity,0,Nx-1)));

    for (int j = 1; j < Nx-1; j++) {
        gsl_vector_set(force,j,(dx/Re)*gsl_matrix_get(vorticity, 0, j));
    }
    return true;
}


int main(void) {
    double dt;
    //double Re = 1.0; //let's try laminar flow for a start. PROGRAM DOESN'T WORK FOR RE < ~150!
    double Re = 250.0;
    double t = 0.0;

    int Nx = 258;  //total number of points, not intervals! Size of vector for FFT is Nx-2, so try to stick to 2^n + 2
    int Ny = 258;
    double dx = 1.0/((double)Nx-1);
    double dy = 1.0/((double)Ny-1);

    gsl_matrix * u = gsl_matrix_alloc (Ny, Nx); //holds velocity components in x-direction
    gsl_matrix * v = gsl_matrix_alloc (Ny, Nx); //holds velocity components in y-direction
    gsl_matrix * psi = gsl_matrix_alloc (Ny, Nx);   //potential field
    gsl_matrix * vorticity = gsl_matrix_alloc (Ny, Nx);
    gsl_matrix * vorticity_next = gsl_matrix_alloc (Ny, Nx);
    gsl_matrix_complex * poisson_psi = gsl_matrix_complex_alloc (2*(Ny-2), 2*(Nx-2));    //we'll need these two for the FFT
    gsl_matrix_complex * poisson_vorticity = gsl_matrix_complex_alloc (2*(Ny-2), 2*(Nx-2));
    gsl_vector * f = gsl_vector_alloc(Nx);

    //directory and file handling. Terribly written, but screw it.
    char bufferdir[100], buffer[100];
    snprintf(bufferdir, 100, "Rezultati/N%03d_Re%05d", Nx, (int)Re);
    strcpy(buffer, bufferdir);
    strcat(buffer, "_psi");
    mkdir(buffer, 0700);
    strcpy(buffer, bufferdir);
    strcat(buffer, "_vort");
    mkdir(buffer, 0700);
    strcpy(buffer, bufferdir);
    strcat(buffer, "_absvel");
    mkdir(buffer, 0700);
    strcpy(buffer, bufferdir);
    strcat(buffer, "_animation");
    mkdir(buffer, 0700);

    strcpy(buffer, bufferdir);
    strcat(buffer, "_data.dat");
    FILE * scalars = fopen(buffer, "wt");

    strcpy(buffer, bufferdir);
    strcat(buffer, "_force.dat");
    FILE * force_vec = fopen(buffer, "wt");

    FILE * out_psi;
    FILE * out_vort;
    FILE * out_absvel;
    FILE * out_anim;

    //initialisation steps

    double tmax = 100.0;  //maximum time of simulation
    int maxsteps = 100001;   //max number of simulation steps
    double fcurrent;    //for holding the force value
    double vortchange;  //for holding changes in vorticity
    double vort_treshold = 5.0; //halt when changes in vorticity sum fall below this level

    gsl_matrix_set_zero(u); //liquid is at rest at t=0...
    gsl_matrix_set_zero(v);
    for (int j = 0; j < Nx; j++) {
        gsl_matrix_set(u,0,j,1.0);    //...except for points touching the lid, which moves towards positive side with unit velocity
    }

    dt = timestep_eval(u, v, Nx, Ny, dx, dy);   //initialize timestep

    gsl_matrix_set_zero(psi);   //up to additive constant, all values can be set to 0

    vorticity_eval(u,v,psi,vorticity,Nx,Ny,dx,dy);    //initialise vorticity matrix

    fcurrent = force_on_lid(vorticity, Nx, dx, Re); //force at t=0
    printf("\nt = %f, F = %.20f",t, fcurrent);


    fprintf(scalars,"%f %f %.10f %f\n",t, 0.0, fcurrent, 0.0);

    //initialisation complete
    int n = 100; //output matrices every n steps
    int n2 = 100; //
    
    for (int m = 0; m < maxsteps; m++) {
        //evolve interior cells in time
        vorticity_step(u, v, psi, vorticity, vorticity_next, Nx, Ny, dx, dy, dt, Re);
        //step one: get new psi from vorticity
        poisson_solver(psi, vorticity_next, poisson_psi, poisson_vorticity, Nx, Ny, dx);

        //update vorticity at edges and force 0 at edges of psi
        vorticity_step_edges(psi, vorticity_next, Nx, Ny, dx, dy);
        psi_force_update_edges(psi, Nx, Ny);
        
        //evaluate changes, then overwrite
        vortchange = vorticity_change(vorticity, vorticity_next, Nx, Ny);
        if ((vortchange < vort_treshold) && (t > 1.0)) {
            printf("\nVorticity converged; program halted.");
            break;
        }

        printf(" dS = %f",vortchange);
        gsl_matrix_memcpy(vorticity, vorticity_next);
        //everything is now updated!

        t += dt;
        if (t > tmax) {break;}

        //calculate force on lid, output relevant data
        fcurrent = force_on_lid(vorticity, Nx, dx, Re);
        printf("\nt = %f, F = %.20f",t, fcurrent);
        fprintf(scalars,"%f %f %.10f %f\n",t, dt, fcurrent, vortchange);

        //every now and then, calculate and output whole force vector. First column contains times.
        if (m%n == 0) {
            force_vec_output(vorticity, f, Nx, dx, Re);
            fprintf(force_vec, "%f", t);
            for (int j = 0; j < Nx; j++) {
                fprintf(force_vec, " %.10f", gsl_vector_get(f,j));
            }
            fprintf(force_vec, "\n");
        }
        
        //also print out field, vorticity and absolute velocity matrices every n steps
        if (m%n == 0) {
            snprintf(buffer, 100, "Rezultati/N%03d_Re%05d_psi/psi_%06.3f.dat", Nx, (int)Re, t);
            out_psi = fopen(buffer,"w");
            sprintf(buffer, "Rezultati/N%03d_Re%05d_vort/vort_%06.3f.dat", Nx, (int)Re, t);
            out_vort = fopen(buffer,"w");
            sprintf(buffer, "Rezultati/N%03d_Re%05d_absvel/absvel_%06.3f.dat", Nx, (int)Re, t);
            out_absvel = fopen(buffer,"w");

            for (int i = 0; i < Ny; i++) {
                for (int j = 0; j < Nx; j++) {
                    fprintf(out_psi,"%f ", gsl_matrix_get(psi,i,j));
                    fprintf(out_vort,"%f ", gsl_matrix_get(vorticity,i,j));
                    fprintf(out_absvel,"%f ", sqrt(gsl_matrix_get(u,i,j)*gsl_matrix_get(u,i,j) + gsl_matrix_get(v,i,j)*gsl_matrix_get(v,i,j)));
                }
                fprintf(out_psi,"\n");
                fprintf(out_vort,"\n");
                fprintf(out_absvel,"\n");
            }

            fclose(out_psi);
            fclose(out_vort);
            fclose(out_absvel);
        }
        
        if (m%n2 == 0) {
            snprintf(buffer, 100, "Rezultati/N%03d_Re%05d_animation/%06d.dat", Nx, (int)Re, m/n2);
            out_anim = fopen(buffer,"w");

            for (int j = 0; j < Nx; j++) {
                for (int i = 0; i < Ny; i++) {
                    fprintf(out_anim, "%f %f %f %f %f\n", (j+0.5)*dx, (i+0.5)*dy, gsl_matrix_get(vorticity,i,j), gsl_matrix_get(u,i,j), gsl_matrix_get(v,i,j));
                }
                fprintf(out_anim,"\n");
            }
            fclose(out_anim);
        }

        //step two: get new velocities
        velocity_eval(u, v, psi, Nx, Ny, dx, dy);
        //step three: calculate new dt and proceed to next iteration
        dt = timestep_eval(u, v, Nx, Ny, dx, dy);
    }
    
    fclose(scalars);
    fclose(force_vec);

    return 0;
}