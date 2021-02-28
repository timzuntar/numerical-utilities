#Code to (not very successfully) determine ionization energies and eigenfunctions of helium atom with Hartree-Fock algorithm

import math 
import numpy
from scipy import optimize
from scipy import integrate

def starting_approx(x,Z):
    Z_prime = Z-5.0/16.0
    return 2.0*math.sqrt(Z_prime)*Z_prime*x*math.exp(-x*Z_prime)

def Numerov_expansion(x,xindex,phi_tabulated,Z,epsilon):
    a1 = 1.0
    a2 = -Z
    a3 = (2.0*Z**2-epsilon-phi_tabulated[xindex])/6.0
    a4 = (-Z**3 - Z*(phi_tabulated[xindex]+epsilon))/18.0

    return a1*x+a2*x**2+a3*x**3+a4*x**4

def Numerov_coefficient(x,xindex,epsilon,phi_tabulated,Z):
    return Z*2.0/x + 2.0*phi_tabulated[xindex] + epsilon

def tabulate_phi(R_values,dx):
    phi_tabulated = []
    for i in range(1,len(R_values)):
        phi_tabulated.append(poisson_integration(i*dx,i,R_values,dx))
    return phi_tabulated

def poisson_integration(x,xindex,R_values,dx):
    numsamples = len(R_values)
    if (xindex == numsamples-1):
        R_inner = R_tabulated
        for i in range(numsamples):
            R_inner[i] *= R_inner[i]
        integral = numpy.trapz(R_inner,x=None,dx=dx)
        return -(1.0/x)*integral
    else:
        R_inner = R_values[:xindex+1]
        R_outer = R_values[xindex:]
        for i in range(len(R_inner)):
            R_inner[i] *= R_inner[i]
        for j in range(len(R_outer)):
            R_outer[j] *= R_outer[j]/(dx*(j+xindex))
        
        term1 = (1.0/x)*numpy.trapz(R_inner,x=None,dx=dx)
        term2 = numpy.trapz(R_outer,x=None,dx=dx)
        return -term1-term2

def Numerovstep(x,xindex,minus2,minus1,dx,phi_tabulated,Z,epsilon):
    kplus = epsilon + 2.0*Z/x + 2.0*phi_tabulated[xindex]
    Rplus = (2.0*(1.0-5.0*(dx**2)*(minus1[0])/12.0)*minus1[1] - (1.0 + (dx**2)*(minus2[0])/12.0)*minus2[1])/(1.0 + (dx**2)*(kplus)/12.0)

    return [kplus,Rplus]

def normalisationconstant(R_tabulated,dx):
    const = integrate.simps(numpy.square(R_tabulated),None,dx=dx, even='first')
    return 1.0/const

def energy_calc(R_tabulated,phi_tabulated,Z,dx):
    derivative_term = []
    Z_term = [0.0]
    potential_term = [0.0]
    for i in range(1,len(R_tabulated)):
        derivative_term.append(((R_tabulated[i]-R_tabulated[i-1])/dx)**2)
        Z_term.append(2.0*Z*(R_tabulated[i]**2)/(i*dx))
        potential_term.append(phi_tabulated[i-1]*(R_tabulated[i]**2))
    t1 = integrate.simps(derivative_term,None,dx=dx, even='first')
    t2 = integrate.simps(Z_term,None,dx=dx, even='first')
    t3 = integrate.simps(potential_term,None,dx=dx, even='first')
    return 2*13.6058*(t1-t2+t3)

def trajectory_integrate(epsilon, dRdx, startx, dx, phi_tabulated, Z, numpts):
    vecminus2 = [Numerov_coefficient(startx,0,epsilon,phi_tabulated,Z),Numerov_expansion(startx,0,phi_tabulated,Z,epsilon)]
    vecminus1 = [Numerov_coefficient(startx+dx,1,epsilon,phi_tabulated,Z),Numerov_expansion(startx+dx,0,phi_tabulated,Z,epsilon)]
    ynumerov = [0,vecminus2[1],vecminus1[1]]
    x = startx+dx

    count = 0
    Rmax = 0.0
    C = 1e-3
    intlength = int(numpts/20) #number of steps function must stay below epsilon to qualify as "converged"

    for i in range(2,numpts):
        x += dx
        vecnumerov = Numerovstep(x,i,vecminus2,vecminus1,dx,phi_tabulated,Z,epsilon)

        if (numpy.sign(vecnumerov[1]) != numpy.sign(vecminus1[1])):
            count = 0

        if (abs(vecnumerov[1]) > Rmax and i < numpts*0.5):
            Rmax = abs(vecnumerov[1])
            eps = Rmax*C

        if (abs(vecnumerov[1]) < eps):  #if function value dips below eps, count how long it stays there
            count += 1

        if (abs(vecnumerov[1]) > eps):  #reset if it increases again
            count = 0

        ynumerov.append(vecnumerov[1])
        """
        if (count >= intlength):
            return ynumerov
        """
        vecminus2 = vecminus1
        vecminus1 = vecnumerov

    return ynumerov

def trajectory_evaluate(epsilon, dRdx, startx, dx, phi_tabulated, Z, numpts):
    #computes one step of the minimization algorithm
    vecminus2 = [Numerov_coefficient(startx,0,epsilon,phi_tabulated,Z),Numerov_expansion(startx,0,phi_tabulated,Z,epsilon)]
    vecminus1 = [Numerov_coefficient(startx+dx,1,epsilon,phi_tabulated,Z),Numerov_expansion(startx+dx,0,phi_tabulated,Z,epsilon)]
    ynumerov = [0,vecminus2[1],vecminus1[1]]
    x = startx+dx

    count = 0
    Rmax = 0.0
    C = 1e-3
    intlength = int(numpts/20) #number of steps function must stay below epsilon to qualify as "converged"

    Av = 100.0    #term due to mismatch between predicted and actual number of zeros
    Ar = 1.0    #term to penalize very large or small maximum values
    Ad = 100.0    #term to penalize fast changes in value past some point

    maxinit = abs(vecminus1[1]-vecminus2[1])
    zeros = 1
    penalty = 0.0  #weighing constant

    for i in range(2,numpts):
        x += dx
        vecnumerov = Numerovstep(x,i,vecminus2,vecminus1,dx,phi_tabulated,Z,epsilon)

        if (x < 1.0):
            if (abs(vecnumerov[1]-vecminus1[1])>maxinit):
                maxinit = abs(vecnumerov[1]-vecminus1[1])

        if (numpy.sign(vecnumerov[1]) != numpy.sign(vecminus1[1])):
            count = 0
            zeros += 1

        if (abs(vecnumerov[1]) > Rmax and i < numpts*0.5):
            Rmax = abs(vecnumerov[1])
            eps = Rmax*C

        if (abs(vecnumerov[1]) < eps):  #if function value dips below eps, count how long it stays there
            count += 1

        if (abs(vecnumerov[1]) > eps):  #reset if it increases again
            count = 0

        ynumerov.append(vecnumerov[1])
        """
        if (count >= intlength):
            finalchange = abs(vecnumerov[1]-vecminus2[1])
            penaltyAv = Av*abs(zeros - 1)
            penaltyAr = Ar*abs(Rmax-abs(vecnumerov[1]))/Rmax
            penaltyAd = Ad*finalchange/maxinit

            print("Good fit reached, penalty elements: %.0f %.10f %.10f" % (penaltyAv,penaltyAr,penaltyAd))
            penalty = penaltyAv+penaltyAr+penaltyAd
            return penalty
        """
        vecminus2 = vecminus1
        vecminus1 = vecnumerov

    finalchange = abs(vecnumerov[1]-vecminus2[1])
    penaltyAv = Av*abs(zeros - 1)
    penaltyAr = Ar*abs(Rmax-abs(vecnumerov[1]))/Rmax
    penaltyAd = Ad*finalchange/maxinit

    #print("Max. integration time exceeded, penalty: %.0f %.10f %.10f" % (penaltyAv,penaltyAr,penaltyAd))
    penalty = penaltyAv+penaltyAr+penaltyAd+1
    return penalty

"""
#TEST
dx = 1e-2
dRdx = 1.0
Z = 2
numpts = int(10/dx)
epsilon = -1.80757

R_tabulated = [0.0]
for i in range(1,numpts+1):
    R_tabulated.append(starting_approx(i*dx,Z))
print("Initial approximation consists of %d points." % (len(R_tabulated)))
phi_tabulated = tabulate_phi(R_tabulated,dx)
R_new = trajectory_integrate(epsilon, dRdx, dx, dx, phi_tabulated, Z, numpts)
R_new = numpy.multiply(R_new,math.sqrt(normalisationconstant(R_new,dx)))    #renormalise
print("Trajectory consists of %d points." % (len(R_new)))
trajectory = open("Rezultati/testtrajectory_initapprox.dat","w+")
for k in range(len(R_new)):
    trajectory.write("%.10f %.15f\n" % (k*dx, R_new[k]))

trajectory.close()
"""


dx = 5e-3
dRdx = 1.0
Z = 1.2
numpts = int(10/dx)
Ebounds = (-20.0,0.0)
convergence_criterium = 1e-5

multipliers = open("Rezultati/multipliers_dx_%.1e_Z_%.2f_initialguess_Zprime.dat" % (dx,Z),"w+")

u = 0
epsilon_past = 0.0   #we don't really know the multiplier for the initial approximation
multipliers.write("%d %.15f\n" % (u,epsilon_past))

R_tabulated = [0.0]
for i in range(1,numpts+1):
    R_tabulated.append(starting_approx(i*dx,Z))

norm = math.sqrt(normalisationconstant(R_tabulated,dx))
for a in range(len(R_tabulated)):
    R_tabulated[a] *= norm
phi_tabulated = tabulate_phi(R_tabulated,dx)
trajectory = open("Rezultati/trajectory_dx_%.1e_Z_%.2f_Zprime_%d.dat" % (dx,Z,0),"w+")
for k in range(1,len(R_tabulated)):
    trajectory.write("%.10f %.15f %.15f\n" % (k*dx, R_tabulated[k],phi_tabulated[k-1]))
trajectory.close()

u += 1
sol = optimize.minimize_scalar(trajectory_evaluate,args=(dRdx, dx, dx, phi_tabulated, Z, numpts), bounds=Ebounds, method="bounded", options={'disp': True,'xatol': 1e-10,'maxiter': 500})
print(sol.x)

epsilon_current = sol.x
multipliers.write("%d %.15f\n" % (u,epsilon_current))

while (abs(epsilon_current-epsilon_past)>convergence_criterium or u < 5):
    #now we get our next iteration of R and potential. The function is integrated and renormalised again
    R_tabulated = trajectory_integrate(sol.x, dRdx, dx, dx, phi_tabulated, Z, numpts)
    norm = math.sqrt(normalisationconstant(R_tabulated,dx))
    for a in range(len(R_tabulated)):
        R_tabulated[a] *= norm
    phi_tabulated = tabulate_phi(R_tabulated,dx)

    trajectory = open("Rezultati/trajectory_dx_%.1e_Z_%.2f_Zprime_%d.dat" % (dx,Z,u),"w+")
    for k in range(1,len(R_tabulated)):
        trajectory.write("%.10f %.15f %.15f\n" % (k*dx, R_tabulated[k],phi_tabulated[k-1]))
    trajectory.close()

    #Ebounds = (2*epsilon_current,0.5*epsilon_current)
    sol = optimize.minimize_scalar(trajectory_evaluate,args=(dRdx, dx, dx, phi_tabulated, Z, numpts), bounds=Ebounds, method="bounded", options={'disp': True,'xatol': 1e-10,'maxiter': 500})
    print("Solution after %dth iteration is %.15f." % (u,sol.x))

    u += 1
    epsilon_past = epsilon_current
    epsilon_current = sol.x
    multipliers.write("%d %.15f\n" % (u,sol.x))

energy = open("Rezultati/energies.dat","a+")
t1 = energy_calc(R_tabulated,phi_tabulated,Z,dx)
energy.write("%f %.2f %.15f %.15f\n" % (dx,Z,t1,epsilon_current*13.6058))
energy.close()

multipliers.close()

"""
#VARIATION OF Z
Z = 1.09
umax = 300
dx = 5e-3
dRdx = 1.0
convergence_criterium = 1e-4
energy = open("Rezultati/Zvar_energies.dat","a+")

z=1
while (Z > 1.04):
    u = 0
    epsilon_past = 0.0   #we don't really know the multiplier for the initial approximation
    Ebounds = (-20.0,0.0)
    numpts = int(10/dx)

    R_tabulated = [0.0]
    for i in range(1,numpts+1):
        R_tabulated.append(starting_approx(i*dx,Z))

    norm = math.sqrt(normalisationconstant(R_tabulated,dx))
    for a in range(len(R_tabulated)):
        R_tabulated[a] *= norm
    phi_tabulated = tabulate_phi(R_tabulated,dx)
    trajectory = open("Rezultati/trajectory_dx_%.1e_Z_%.2f_Zprime_%d.dat" % (dx,Z,0),"w+")
    for k in range(1,len(R_tabulated)):
        trajectory.write("%.10f %.15f %.15f\n" % (k*dx, R_tabulated[k],phi_tabulated[k-1]))
    trajectory.close()

    u += 1
    sol = optimize.minimize_scalar(trajectory_evaluate,args=(dRdx, dx, dx, phi_tabulated, Z, numpts), bounds=Ebounds, method="bounded", options={'disp': True,'xatol': 1e-10,'maxiter': 500})
    print(sol.x)

    epsilon_current = sol.x

    while (abs(epsilon_current-epsilon_past)>convergence_criterium or u < 5):
        if (u > umax):
            break
        #now we get our next iteration of R and potential. The function is integrated and renormalised again
        R_tabulated = trajectory_integrate(sol.x, dRdx, dx, dx, phi_tabulated, Z, numpts)
        norm = math.sqrt(normalisationconstant(R_tabulated,dx))
        for a in range(len(R_tabulated)):
            R_tabulated[a] *= norm
        phi_tabulated = tabulate_phi(R_tabulated,dx)

        sol = optimize.minimize_scalar(trajectory_evaluate,args=(dRdx, dx, dx, phi_tabulated, Z, numpts), bounds=Ebounds, method="bounded", options={'disp': True,'xatol': 1e-10,'maxiter': 500})
        print("Solution after %dth iteration is %.15f." % (u,sol.x))

        u += 1
        epsilon_past = epsilon_current
        epsilon_current = sol.x
    if (u > umax):
        break
    trajectory = open("Rezultati/trajectory_dx_%.1e_Z_%.2f_Zprime_%d.dat" % (dx,Z,u),"w+")
    for k in range(1,len(R_tabulated)):
        trajectory.write("%.10f %.15f %.15f\n" % (k*dx, R_tabulated[k],phi_tabulated[k-1]))
    trajectory.close()

    
    t1 = energy_calc(R_tabulated,phi_tabulated,Z,dx)
    energy.write("%f %.2f %d %.15f %.15f\n" % (dx,Z,u,t1,epsilon_current*13.6058))
    Z -= 0.01
    z += 1

energy.close()
"""