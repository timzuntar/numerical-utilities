//computes the ubiquitous transition in 2D Ising model via Metropolis algorithm

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_matrix.h>
#include <sys/time.h>

double dE (gsl_matrix_int* F, int n, int i, int j, double J, double H) {
  int left,right,up,down;

  if (i == 0) {left = gsl_matrix_int_get(F,n-1,j);}
  else {left = gsl_matrix_int_get(F,i-1,j);}

  if (i == n-1) {right = gsl_matrix_int_get(F,0,j);}
  else {right = gsl_matrix_int_get(F,i+1,j);}

  if (j == 0) {up = gsl_matrix_int_get(F,i,n-1);}
  else {up = gsl_matrix_int_get(F,i,j-1);}

  if (j == n-1) {down = gsl_matrix_int_get(F,i,0);}
  else {down = gsl_matrix_int_get(F,i,j+1);}  

  return 2.0*J*gsl_matrix_int_get(F,i,j)*(left+right+up+down) + 2.0*H*gsl_matrix_int_get(F,i,j);
}

void samp (gsl_matrix_int* F, int n, double * s, double * e, double J, double H ) {
  int i,j;
  int esum;double Hamiltonian;

  *s=0.0;
  *e=0.0;

  for (i=0;i<n;i++) {
    for (j=0;j<n;j++) {
      *s+=gsl_matrix_int_get(F,i,j);
      esum = 0;

      if (i == 0) {esum += gsl_matrix_int_get(F,n-1,j);}
      else {esum += gsl_matrix_int_get(F,i-1,j);}
      if (i == n-1) {esum += gsl_matrix_int_get(F,0,j);}
      else {esum += gsl_matrix_int_get(F,i+1,j);}
      if (j == 0) {esum += gsl_matrix_int_get(F,i,n-1);}
      else {esum += gsl_matrix_int_get(F,i,j-1);}
      if (j == n-1) {esum += gsl_matrix_int_get(F,i,0);}
      else {esum += gsl_matrix_int_get(F,i,j+1);}

      Hamiltonian = 0.0 - 0.5*J*(gsl_matrix_int_get(F,i,j))*esum+gsl_matrix_int_get(F,i,j)*H;

      *e+=Hamiltonian;
    }
  }
  *s/=(double)(n*n);
  *e/=(double)(n*n);  //mean values of spin and energy
}

void space_corr (gsl_matrix_int* F, gsl_vector * corrs, int n, int maxcorrlen) {
  int i,j;
  int zero,offset;
  int dist;

  for (dist=0;dist<maxcorrlen;dist++) {
    for (i=0;i<n;i++) {
      for (j=0;j<n;j++) {
        zero = gsl_matrix_int_get(F,i,j);

        if (dist+i>=n) {offset = gsl_matrix_int_get(F,(i+dist)%n,j);}
        else {offset = gsl_matrix_int_get(F,i+dist,j);}

        gsl_vector_set(corrs,dist,gsl_vector_get(corrs,dist)+zero*offset);
      }
    }
  }
}

void init (gsl_matrix_int* F, int n, gsl_rng * r ) {
  int i,j;
  for (i=0;i<n;i++) {
    for (j=0;j<n;j++) {
      gsl_matrix_int_set(F,i,j,2*(int)gsl_rng_uniform_int(r,2)-1);
    }
  }
}

void correl(double Ce[],double Cm[],double E,double M,
            double esave[],double msave[],int pass,int tsave)
   {
      int tdiff,index0,index;
      index0 = ((pass - 1) % tsave) + 1;
      if (pass > tsave) 
        {
         index = index0;
         for (tdiff=tsave; tdiff >= 1; tdiff--)
           {
            Ce[tdiff-1] = Ce[tdiff-1] + E*esave[index-1];
            Cm[tdiff-1] = Cm[tdiff-1] + M*msave[index-1];
            index = index + 1;
            if(index > tsave) index = 1;
            }
         }
      esave[index0-1] = E;
      msave[index0-1] = M;
    }

int main (void) {

   int n = 50;
   int N = n*n;
   int maxcorrlen = n/2;

   double T = 1.5;   //kT/J originally, just kT here
   double J = 1.0;
   double H = 1.0;   //for now
   int nCycles = 100000; //number of flips at each temperature is nCycles*n*n
   int fSamp = 1;
   int nSamp,tdiff;
   double de;
   double b,x;
   int i,j,a,c;
   double Ce[50],Cm[50],esave[101],msave[101];
   int tsave = 40;

  gsl_matrix_int * matrika = gsl_matrix_int_alloc (n, n);
  gsl_vector * corr = gsl_vector_alloc(maxcorrlen);

   //FILE * out_e = fopen("Podatki/2_N10J1H0_104draws.dat", "wt");
   FILE * out_2d = fopen("Podatki/2_N50H1_T15_array.dat","wt");
   FILE * out_timecorr = fopen("Podatki/2_N50H1_T15_timecorr.dat","wt");
   FILE * out_spacecorr = fopen("Podatki/2_N50H1_T15_spacecorr.dat","wt");

   double s,e;
   double cumul[7];
   gsl_rng * random = gsl_rng_alloc(gsl_rng_mt19937);

   init(matrika,n,random);    //populates the grid with randomly chosen spins

//////////////////////////////////////
  for(int state=0;state<1;state++) {  //was 10000
    if (T < 1.0) {break;}

    cumul[0] = 0.0; //spin sum
    cumul[1] = 0.0; //energy sum
    cumul[2] = 0.0; //sum of s^2
    cumul[3] = 0.0; // sum of E^2
    cumul[4] = 0.0; //abs. magnitude of spin
    cumul[5] = 0.0; //susceptibility
    cumul[6] = 0.0; //heat capacity

    gsl_vector_set_zero(corr); //initialise, or forget the previous correlations
    for (int l=0;l<50;l++) {
      Ce[l] = 0;
      Cm[l] = 0;
    }
    for (int l=0;l<101;l++) {
      esave[l] = 0;
      msave[l] = 0;
    }

    nSamp=0;

    for (c=0;c<nCycles;c++) {
      for (a=0;a<N;a++) {     //a_max x isn't necessarily equal to N
         i=gsl_rng_uniform_int(random,n);
         j=gsl_rng_uniform_int(random,n);  //choose i-th, j-th spin

         de = dE(matrika,n,i,j,J,H);  //what would the energy difference be?

        if (de < 0) {
          gsl_matrix_int_set(matrika,i,j,gsl_matrix_int_get(matrika,i,j)*-1);
        }
        else {
          b = exp(-de/T);          //this is functionally de/T
          x = gsl_rng_uniform(random);  //equivalent to chi from earlier method
          if (x<b) {gsl_matrix_int_set(matrika,i,j,gsl_matrix_int_get(matrika,i,j)*-1);} //accept the new configuration
        }
      }
      if (c%fSamp == 0 && c > nCycles*0.9) {  //sampling only takes place in second half to avoid burn-in
         samp(matrika,n,&s,&e,J,H); //sampling of system
         cumul[0] += s;
         cumul[1] += e;
         cumul[2] += s*s;
         cumul[3] += e*e;
         cumul[4] += fabs(s);

         correl(Ce,Cm,e*N,s*N,esave,msave,nSamp,tsave);
         space_corr (matrika, corr, n, maxcorrlen);

         nSamp+=1;
      }
    }

    //calculation of susceptibility and specific heat capacity
    cumul[5] = fabs(cumul[2]/(double)nSamp - cumul[0]*cumul[0]/(double)(nSamp*nSamp));
    cumul[6] = fabs(cumul[3]/(double)nSamp - cumul[1]*cumul[1]/(double)(nSamp*nSamp));
    for (int d=1;d<maxcorrlen;d++) {
      gsl_vector_set(corr,d,gsl_vector_get(corr,d)/gsl_vector_get(corr,0)); //initialise, or forget the previous correlations
      }
      gsl_vector_set(corr,0,gsl_vector_get(corr,0)/gsl_vector_get(corr,0));

    printf( "T= %f, Average M: %.10f\n", T, cumul[0]/(double)nSamp);
      //printf("\n    t       Ce(t)  Cm(t)       \n\n");
      for (tdiff = 0;tdiff < tsave;tdiff++)
        {
        //correlation is 1 at 0 and 0 at infinity
          Ce[tdiff] = (Ce[tdiff]/(double)(nSamp-tsave) - N*N*cumul[1]*cumul[1]/((double)nSamp*nSamp));
          Cm[tdiff] = (Cm[tdiff]/(double)(nSamp-tsave) - N*N*cumul[0]*cumul[0]/((double)nSamp*nSamp));
          fprintf(out_timecorr,"%6d %10.5f %10.5f\n",tdiff,Ce[tdiff]/Ce[0],Cm[tdiff]/Cm[0]);
        }

      for (int d=0;d<maxcorrlen;d++) {
        fprintf(out_spacecorr,"\n%d %f",d,gsl_vector_get(corr,d));
      }
    //printf("# The average energy per spin is %.10f\n", cumul[1]/(double)nSamp);
    //fprintf(out_e, "%f  %d %.15f %.15f %.15f %.15f %.15f\n", T, (state+1)*nCycles, fabs(cumul[0]/(double)nSamp), cumul[1]/(double)nSamp, cumul[4]/(double)nSamp, cumul[5]/T, cumul[6]/(T*T));
    
    for (int i=0;i<n;i++) {
      for (int j=0;j<n;j++) {
        fprintf(out_2d,"%d ",gsl_matrix_int_get(matrika,i,j));
      }
      fprintf(out_2d,"\n");
    }

    if (T>2.0 && T<2.6) {T/= 1.01;}
    else {T /= 1.05;}
  }
 //////////////////////////////////
   //fclose(out_e);
   fclose(out_2d);
   fclose(out_timecorr);
   fclose(out_spacecorr);
   return 0;
}