#Simple script for calculation of analytical SIMS depth profile signal
#and adherence to experimental data. Written for EFPov course in August 2020.

import numpy as np
from scipy import optimize
import math
import os

def calculate_deviation(measured_norm, analytical_norm):
    #returns goodness-of-fit (epsilon) according to scheme by Liu et al.
    #cutoff is fraction of signal maximum, e.g. 10^-3, below which data is discarded as noise.
    cutoff = 1e-3

    n = 0
    maximum = np.max(measured_norm)
    summa = 0.0

    for index,point in enumerate(measured_norm):
        if (point/maximum > cutoff):
            n += 1
            summa += (analytical_norm[index] - measured_norm[index])**2
        
    return math.sqrt(summa/(n*maximum))

def MRI_evaluate_delta(par_w,par_lambda,par_sigma,par_z0,position):
    #evaluates analytical MRI resolution function for an ideal delta-layer according to Liu et al.
    #very low values of lambda are handled separately, script is otherwise prone to crashing.

    lambda_as_0_cutoff = 1e-3
    z = position - par_z0
    
    if (par_lambda < lambda_as_0_cutoff):
        term1 = 1.0/(2*par_w)
        term1 *= np.exp((-z-par_w)/par_w + (par_sigma**2)/(2*(par_w**2)))
        term1 *= 1 - math.erf(((-z-par_w)/par_sigma + par_sigma/par_w)/math.sqrt(2))
        term2 = 0.0
    else:
        term1 = (1-math.exp(-par_w/par_lambda))/(2*par_w)
        term1 *= np.exp((-z-par_w)/par_w + (par_sigma**2)/(2*(par_w**2)))
        term1 *= 1 - math.erf(((-z-par_w)/par_sigma + par_sigma/par_w)/math.sqrt(2))
        term2 = np.exp(z/par_lambda + (par_sigma**2)/(2*(par_lambda**2)))/(2*par_lambda)
        term2 *= 1 + math.erf(((-z-par_w)/par_sigma - par_sigma/par_w)/math.sqrt(2))

    return term1 + term2

def MRI_evaluate_finite_thickness(par_w,par_lambda,par_sigma,par_z1,par_z2,position):
    #evaluates analytical MRI resolution function for a homogenous layer spanning from depth z1 to z2 according to Gautier et al.
    #very low values of lambda are handled separately, script seems prone to crashing otherwise.
    lambda_as_0_cutoff = 1e-3
    z1 = position - par_z1
    z2 = position - par_z2

    if (par_lambda < lambda_as_0_cutoff):
        term1 = 0.5*(math.erf((z1 + par_w)/(math.sqrt(2)*par_sigma))-math.erf((z2 + par_w)/(math.sqrt(2)*par_sigma)))
        term2 = 0.0
        term3 = 0.5*np.exp((par_sigma**2)/(2*(par_w**2)))
        term3 *= np.exp((par_w-z2)/par_w)*(1.0+math.erf((z2+par_w)/(par_sigma*math.sqrt(2)) - par_sigma/(par_w*math.sqrt(2)))) - np.exp((par_w-z1)/par_w)*(1.0+math.erf((z1+par_w)/(par_sigma*math.sqrt(2)) - par_sigma/(par_w*math.sqrt(2))))

    else:
        term1 = 0.5*(math.erf((z1 + par_w)/(math.sqrt(2)*par_sigma))-math.erf((z2 + par_w)/(math.sqrt(2)*par_sigma)))
        term2 = 0.5*np.exp((par_sigma**2)/(2*(par_lambda**2)))
        term2 *= (np.exp(z1/par_lambda)*(1.0-math.erf((z1+par_w)/(par_sigma*math.sqrt(2)) + par_sigma/(par_lambda*math.sqrt(2)))) - np.exp(z2/par_lambda)*(1.0-math.erf((z2+par_w)/(par_sigma*math.sqrt(2))+par_sigma/(par_lambda*math.sqrt(2)))))
        term3 = 0.5*np.exp((par_sigma**2)/(2*(par_w**2)))*(1.0-np.exp(-par_w/par_lambda))
        term3 *= np.exp((-par_w-z2)/par_w)*(1.0+math.erf((z2+par_w)/(par_sigma*math.sqrt(2)) - par_sigma/(par_w*math.sqrt(2)))) - np.exp((-par_w-z1)/par_w)*(1.0+math.erf((z1+par_w)/(par_sigma*math.sqrt(2)) - par_sigma/(par_w*math.sqrt(2))))

    return term1 + term2 + term3

def calculate_profile(params,positions,profile_type):
    #returns the analytical MRI profile for a buried layer of either infinitesimal or finite thickness
    #for layer edge calculation, a width of 50 nm is assumed (should be more than enough)

    analytical_norm = np.empty(len(positions))
    if (profile_type == "delta" and len(params)==4):
        for pos_index, position in enumerate(positions):
            analytical_norm[pos_index] = MRI_evaluate_delta(*params,position)
    elif (profile_type == "edge"):
        dummy_z0 = params[3] - 50.0
        for pos_index, position in enumerate(positions):
            analytical_norm[pos_index] = MRI_evaluate_finite_thickness(params[0],params[1],params[2],dummy_z0,params[3],position)
    elif (profile_type == "finite"):
        for pos_index, position in enumerate(positions):
            analytical_norm[pos_index] = MRI_evaluate_finite_thickness(*params,position)      

    #normalizes calculated profile again, just in case
    maximum = np.max(analytical_norm)
    analytical_norm = np.divide(analytical_norm,maximum)
    
    return analytical_norm

def MRI_wrapper(params, positions, measured_norm, profile_type):
    #handler for minimization routine. Returns goodness-of-fit.
    #separate from calculate_profile so it can be used for returning the entire profile

    analytical_norm = calculate_profile(params,positions,profile_type)
    epsilon = calculate_deviation(measured_norm,analytical_norm)

    print("Current parameter values:")
    print(*params)
    print("\nepsilon = %f" % (epsilon))

    return epsilon

###################
#SCRIPT STARTS HERE
###################

#name of file containing measurement data (needs to be in folder "script_input"!)
file_name = "ZJ7Cs05"
file_extension = ".dat"
input_name = "script_input/" + file_name + file_extension

#note: columns start with 0!
depthcolumn = 2 #column of depth data. 
signalcolumn = 7    #column of chosen signal data (in our case 7 were Sr+ ions)
#script assumes signal to be already normalized
ignore = 100    #number of starting data points to ignore (e.g. to avoid fitting surface effects)
positions, measured = np.loadtxt(input_name,usecols=(depthcolumn,signalcolumn),skiprows=ignore,unpack=True)  

if not os.path.exists("script_output"):
    os.makedirs("script_output")

#set profile type and initial parameter values
#"delta" - delta layer at depth par_z0
#"edge" - semi-infinite layer ending at depth par_zend
#"finite" - layer of finite thickness from par_zstart to par_zend
profile_type = "delta"  #possible values: "delta", "edge", "finite"

#initial parameter values.
par_w = 1.5
par_lambda = 0.1
par_sigma = 0.5
par_z0 = 20.4   #for "delta" profile
par_zstart = 19.0 #for "finite" profile
par_zend = 21.0 #for "edge" and "finite" profiles

#minimum and maximum possible values for each parameter. Adjust accordingly.
#unused with Nelder-Mead method. 
bounds_w = (0.1,10.0)
bounds_lambda = (0.0,5.0)
bounds_sigma = (0.1,5.0)
bounds_z0 = (19.0,21.0)
bounds_zstart = (15.0,29.0)
bounds_zend = (16.0,30.0)

#set parameter and bound vectors
if (profile_type == "delta"):
    params = [par_w,par_lambda,par_sigma,par_z0]
    param_bounds = (bounds_w,bounds_lambda,bounds_sigma,bounds_z0)
elif (profile_type == "edge"):
    params = [par_w,par_lambda,par_sigma,par_zend]
    param_bounds = (bounds_w,bounds_lambda,bounds_sigma,bounds_zend)
elif (profile_type == "finite"):
    params = [par_w,par_lambda,par_sigma,par_zstart,par_zend]
    param_bounds = (bounds_w,bounds_lambda,bounds_sigma,bounds_zstart,bounds_zend)

#run minimization routine. method="Nelder-Mead" ignores bounds. method="Powell" respects bounds, but is less stable for bad initial guesses.
#if Powell's method returns nonsense results regardless of parameter choice, try Nelder-Mead or adjust parameter values
min_method = "Nelder-Mead"  #other good option is "Nelder-Mead"
optimization_result = optimize.minimize(MRI_wrapper,params,args=(positions,measured,profile_type),method=min_method,bounds=param_bounds,options={"disp": True})
if (optimization_result.success == True):
    print("Algorithm has converged after %d steps. Final parameter values:\n" % (optimization_result.nfev))
    print(optimization_result.x)
else:
    print("No convergence after %d steps.\nMake sure correct profile type is selected or adjust initial values." % (optimization_result.nfev))

#print to files
output_string = "script_output/" + file_name + "_" + profile_type + "_fit.dat"
profile_output = open(output_string,"wt")
param_output = open("script_output/fitdata.dat","a")
analytical_norm = calculate_profile(optimization_result.x,positions,profile_type)

#prints depth and measured data to columns 1 and 2, best fit to column 3 
for element in range(len(analytical_norm)):
    profile_output.write("%f %f %f\n" % (positions[element],measured[element],analytical_norm[element]))

#prints file name, source column, profile type, final parameter values and (in last column) goodness of fit
param_output.write("%s column_%d %s" % (file_name,signalcolumn,profile_type))
for i in range(len(optimization_result.x)):
    param_output.write(" %f" % optimization_result.x[i])
param_output.write(" %f\n" % (optimization_result.fun))

profile_output.close()
param_output.close()