import math
import os
from cv2 import imread 
from numba import njit
import numpy as np
from scipy import interpolate,optimize

num_cpus = 3

def Timestep(C_current,cmin,dt0):
    cmin_current = C_current.min()
    if (cmin_current > cmin):
        return dt0*(1-cmin)/(1-cmin_current)
    else:
        return dt0

def RasterToConcentration(C_current,raster,Nx,Ny,c0):
    #a value of 0 -> no adsorption, 255 -> equilibrium adsorbed concentration
    for i in range(0,Nx):
        for j in range(0,Ny):
            C_current[i,j] = c0*(raster[j,i]/255.0)
    return C_current

@njit
def AdsorptionRateModifier(C_current,C_next,V_current,Nx,Ny,hz,dt,c0,Rprime,ke,a):
    for i in range(0,Nx):
        for j in range(0,Ny):
            if (i==0): left = c0
            else: left = C_current[i-1,j]
            if (i==Nx-1): right = c0
            else: right = C_current[i+1,j]
            if (j==0): backward = c0
            else: backward = C_current[i,j-1]
            if (j==Ny-1): forward = c0
            else: forward = C_current[i,j+1]
            neighborsum = left + right + forward + backward
            dC = dt*V_current[i,j,0]*Rprime*(c0-C_current[i,j])*(1 + ke*(C_current[i,j] + a*neighborsum)/(1+4*a))
            if (dC/hz >= V_current[i,j,0]):
                C_next[i,j] = C_current[i,j] + V_current[i,j,0]*hz
                V_current[i,j,0] = 0.0
                #print("Caution!")
            else:
                C_next[i,j] = C_current[i,j] + dC
                V_current[i,j,0] -= dC/hz
    return C_next,V_current

@njit
def IterateTimeStep(V_current,V_next,D,Nx,Ny,Nz,dt,hlateral,hz,cb,c0):
    pathological = False
    for i in range(0,Nx):
        for j in range(0,Ny):
            for k in range(0,Nz):
                if (i==0): left = cb
                else: left = V_current[i-1,j,k]
                if (i==Nx-1): right = cb
                else: right = V_current[i+1,j,k]
                if (j==0): backward = cb
                else: backward = V_current[i,j-1,k]
                if (j==Ny-1): forward = cb
                else: forward = V_current[i,j+1,k]
                if (k==Nz-1): bottom = cb
                else: bottom = V_current[i,j,k+1]
                if (k==0):
                    V_next[i,j,k] = V_current[i,j,k] + dt * D * ((left+right+forward+backward - 4*V_current[i,j,k])/(hlateral*hlateral) + (bottom - V_current[i,j,k])/(hz*hz))
                else:
                    top = V_current[i,j,k-1]
                    V_next[i,j,k] = V_current[i,j,k] + dt * D * ((left+right+forward+backward - 4*V_current[i,j,k])/(hlateral*hlateral) + (top+bottom - 2*V_current[i,j,k])/(hz*hz))
                if(V_next[i,j,k] < 0.0):
                    #print("Negative concentration value encountered! %e -> %e" % (V_current[i,j,k],V_next[i,j,k]))
                    pathological = True
                    return V_current, pathological
    return V_next, pathological

@njit
def IterateTimeStepParallelised(V_current,D,Nzstart,Nzstop,Nx,Ny,Nz,dt,hlateral,hz,cb,c0):
    V_next_slice = np.empty((Nx,Ny,Nzstop-Nzstart),dtype="float64")
    pathological = False
    for i in range(0,Nx):
        for j in range(0,Ny):
            for k in range(Nzstart,Nzstop):
                if (i==0): left = cb
                else: left = V_current[i-1,j,k]
                if (i==Nx-1): right = cb
                else: right = V_current[i+1,j,k]
                if (j==0): backward = cb
                else: backward = V_current[i,j-1,k]
                if (j==Ny-1): forward = cb
                else: forward = V_current[i,j+1,k]
                if (k==Nz-1): bottom = cb
                else: bottom = V_current[i,j,k+1]
                if (k==0):
                    V_next_slice[i,j,k-Nzstart] = V_current[i,j,k] + dt * D * ((left+right+forward+backward - 4*V_current[i,j,k])/(hlateral*hlateral) + (bottom - V_current[i,j,k])/(hz*hz))
                else:
                    top = V_current[i,j,k-1]
                    V_next_slice[i,j,k-Nzstart] = V_current[i,j,k] + dt * D * ((left+right+forward+backward - 4*V_current[i,j,k])/(hlateral*hlateral) + (top+bottom - 2*V_current[i,j,k])/(hz*hz))
                if(V_next_slice[i,j,k-Nzstart] < 0.0):
                    #print("Negative concentration value encountered! %e -> %e" % (V_current[i,j,k],V_next_slice[i,j,k-Nzstart]))
                    pathological = True
                    return V_current, pathological
    return V_next_slice, pathological

def ExtractROIAverage(C_current,t,simulation_concs,simulation_times,ROI_bounds):
    sum = 0.0
    for i in range(ROI_bounds[0],ROI_bounds[2]+1):
        for j in range(ROI_bounds[1],ROI_bounds[3]+1):
            sum += C_current[i,j]
    sum /= math.fabs(ROI_bounds[2]+1-ROI_bounds[0])*(math.fabs(ROI_bounds[3]+1-ROI_bounds[1]))
    simulation_concs.append(sum)
    simulation_times.append(t)

def CompareSignals(measurement_times,measurement_concs,simulation_times,simulation_concs,init_conc,c0,time_delay,fulloutput):
    #Intensity is assumed to vary with square of concentration.
    measurement_times += time_delay #move measurement data to proper delay
    simulation_concs = (simulation_concs-init_conc)/(c0-init_conc)  #initial normalisation
    signal = np.square(np.array(simulation_concs))  #concentrations to signal

    func_sim = interpolate.interp1d(simulation_times,signal,fill_value="extrapolate")
    interpolated_concs = func_sim(measurement_times)
    start_conc = func_sim(time_delay)   #find simulated concentration at ROI definition
    measurement_concs = measurement_concs*(1.0-start_conc) + start_conc    #squash measurement to proper signal range

    N = len(measurement_times)
    MSD = 0.0
    for n in range(0,N):
        MSD += (interpolated_concs[n] - measurement_concs[n])**2
    MSD /= N
    if (fulloutput == False):
        return 1e6*MSD
    else:
        return 1e6*MSD,signal,simulation_times,measurement_concs,measurement_times

def SingleRun(x,V_next,C_next,raster,measurement_times,measurement_concs,Nx,Ny,Nz,c0,cb,cmin,hlateral,hz,dt0,time_delay,ROI_bounds,a,fulloutput,D):
    ke = x[0]*1e4
    Rprime = x[1]
    #Rprime = x[0]  #1-parameter minimization
    #D = x[2]*1e-12 #3-parameter minimization
    D *= 1e-12
    t = 0.0

    V_current = np.full((Nx,Ny,Nz),cb)
    C_current = np.empty((Nx,Ny))
    C_current = RasterToConcentration(C_current,raster,Nx,Ny,c0)
    simulation_concs = []
    simulation_times = []
    ExtractROIAverage(C_current,t,simulation_concs,simulation_times,ROI_bounds)
    init_conc = simulation_concs[0]
    printcounter = 0

    while(C_current.min() <= cmin):
        if (t > 10.0/cb):
            print("Run terminated at t = %f. Max. time exceeded." % t)
            if (fulloutput == True):
                #return CompareSignals(measurement_times,measurement_concs,simulation_times,simulation_concs,init_conc,c0,time_delay), simulation_concs, simulation_times
                return CompareSignals(measurement_times,measurement_concs,simulation_times,simulation_concs,init_conc,c0,time_delay,True)
            else:
                return CompareSignals(measurement_times,measurement_concs,simulation_times,simulation_concs,init_conc,c0,time_delay,False)
        dt = Timestep(C_current,cmin,dt0)
        C_next,V_current = AdsorptionRateModifier(C_current,C_next,V_current,Nx,Ny,hz,dt,c0,Rprime,ke,a)

        V_next,pathological = IterateTimeStep(V_current,V_next,D,Nx,Ny,Nz,dt,hlateral,hz,cb,c0)

        if (pathological == True):
            print("Run terminated at t = %f. Negative concentration value encountered." % t)
            if (fulloutput == True):
                return 1e6,[],[]
            else:
                return 1e6
        
        if (printcounter >= 200):    
            print(t,C_current.min())
            printcounter = 0
        

        np.copyto(C_current, C_next)
        np.copyto(V_current, V_next)
        t += dt
        printcounter += 1
        ExtractROIAverage(C_current,t,simulation_concs,simulation_times,ROI_bounds)

    print("Run finished at t = %f." % t)
    if (fulloutput == True):
        #return CompareSignals(measurement_times,measurement_concs,simulation_times,simulation_concs,init_conc,c0,time_delay), (simulation_concs-init_conc)*c0/(c0-init_conc), simulation_times
        return CompareSignals(measurement_times,measurement_concs,simulation_times,simulation_concs,init_conc,c0,time_delay,True)
    else:
        return CompareSignals(measurement_times,measurement_concs,simulation_times,simulation_concs,init_conc,c0,time_delay,False)

@njit(parallel=True)
def IterateTimeStepMain(V_next,V_current,D,zslice_height,Nx,Ny,Nz,dt,hlateral,hz,cb,c0,num_cpus):
    timestep = []
    for cpu in range(num_cpus):
        Nzstart = zslice_height*cpu
        if (cpu == num_cpus-1):
            Nzstop = Nz
        else:
            Nzstop = zslice_height*(cpu+1)
        timestep.append(IterateTimeStepParallelised(V_current,D,Nzstart,Nzstop,Nx,Ny,Nz,dt,hlateral,hz,cb,c0))
    for cpu in range(num_cpus):
        Nzstart = zslice_height*cpu
        if (cpu == num_cpus-1):
            Nzstop = Nz
        else:
            Nzstop = zslice_height*(cpu+1)

        pathological = timestep[cpu][1]
        if (pathological == True):
            return V_next,pathological
        V_next[:,:,Nzstart:Nzstop] = timestep[cpu][0]  
    return V_next,pathological

def SingleRunParallelised(x,V_next,C_next,raster,measurement_times,measurement_concs,Nx,Ny,Nz,c0,cb,cmin,hlateral,hz,dt0,ROI_bounds,a,num_cpus,fulloutput):
    #DEPRECATED!
    ke = x[0]*1e4
    Rprime = x[1]
    D = x[2]*1e-12
    t = 0.0

    zslice_height = Nz//num_cpus

    V_current = np.full((Nx,Ny,Nz),cb)
    C_current = np.empty((Nx,Ny))
    C_current = RasterToConcentration(C_current,raster,Nx,Ny,c0)
    simulation_concs = []
    simulation_times = []
    ExtractROIAverage(C_current,t,simulation_concs,simulation_times,ROI_bounds)
    init_conc = simulation_concs[0]
    printcounter = 0

    while(C_current.min() <= cmin):
        if (t > 1000.0):
            if (fulloutput == True):
                return 1e6*CompareSignals(measurement_times,measurement_concs,simulation_times,(simulation_concs-init_conc)*c0/(c0-init_conc),c0), simulation_concs, simulation_times
            else:
                return 1e6*CompareSignals(measurement_times,measurement_concs,simulation_times,(simulation_concs-init_conc)*c0/(c0-init_conc),c0)

        dt = Timestep(C_current,cmin,dt0)
        C_next,V_current = AdsorptionRateModifier(C_current,C_next,V_current,Nx,Ny,hz,dt,c0,Rprime,ke,a)

        V_next,pathological = IterateTimeStepMain(V_next,V_current,D,zslice_height,Nx,Ny,Nz,dt,hlateral,hz,cb,c0,num_cpus)
        if (pathological == True):
            if (fulloutput == True):
                return 1e6, [], []
            else:
                return 1e6

        if (printcounter >= 200):    
            print(t,C_current.min())
            printcounter = 0

        np.copyto(C_current, C_next)
        np.copyto(V_current, V_next)
        t += dt
        printcounter += 1
        ExtractROIAverage(C_current,t,simulation_concs,simulation_times,ROI_bounds)

    if (fulloutput == True):
        return 1e6*CompareSignals(measurement_times,measurement_concs,simulation_times,simulation_concs,c0), simulation_concs, simulation_times
    else:
        return 1e6*CompareSignals(measurement_times,measurement_concs,simulation_times,simulation_concs,c0)

#the average size of unadsorbed areas is around 100 microns
hlateral = 4.4e-6 #resolution of images is around 1.1 micron/pixel. 4x reduced resolution -> 4.4 microns/pixel
hz = 4.4e-6 #cube grid
c0 = 0.81e-6 #around 1 mg/m^2

inputs = np.genfromtxt("sample_inputtable.dat",dtype=("|S31", "|S39", float, float, float, float, float, float, int, int, int, int, float))
print(inputs.shape)
if len(inputs.shape) == 0:
    inputs = np.array([inputs])

for i in range(np.shape(inputs)[0]):
    dt0 = inputs[i][2]
    time_delay = inputs[i][12]

    #Initial parameter choice
    ke = inputs[i][5]
    #ke = 0.0
    Rprime = inputs[i][6]
    a = 0.0 #neighbour weighing parameter. A value of 0.25 weighs the sum of neighbors equally to the central cell
    #D = inputs[i][7] #the actual value is D*1e-12, but such a discrepancy in values messes with the minimization algorithm
    D = 7.0

    #initguess = [ke,Rprime,D]  #3-parameter model
    #parameterbounds = ((0.0,1e+3),(0.1,3e1),(5.0,60.0))
    initguess = [ke,Rprime] #2-parameter
    parameterbounds = ((0.0,5e+3),(0.01,1e3))
    #initguess = [Rprime]   #1-parameter
    #parameterbounds = ((0.01,1e2),)

    cb = inputs[i][4]    #boundary concentration (in kg/m^3)
    cmin = inputs[i][3]*c0    #stop iterating when every surface cell surpasses this concentration 

    raster = imread("input-rasters/" + inputs[i][1].decode("utf-8"),0)
    Ny,Nx = raster.shape
    Nz = int(min(Nx,Ny)*hlateral/hz)

    ROI_bounds = (inputs[i][8],inputs[i][9],inputs[i][10],inputs[i][11])

    print("Full size of array is %d, %d, %d. Starting simulation..." % (Nx,Ny,Nz))

    out = np.loadtxt("input-signals/" + inputs[i][0].decode("utf-8"))
    out = out[np.where(out[range(out.shape[0]),1] < inputs[0][3]**2 - 0.001)]
    measurement_times = out[:,0]
    measurement_concs = out[:,1]
    #measurement points with c > cmin need to be removed, otherwise interpolation may fail

    V_next = np.empty((Nx,Ny,Nz),dtype="float64")
    C_next = np.empty((Nx,Ny), dtype="float64")

    #if the next 4 lines are commented out, minimization is not performed
    #instead, the simulation is carried out at parameter values included in the input table
    #this is useful if you wish to tweak something by hand
    """
    godobject = (V_next,C_next,raster,measurement_times,measurement_concs,Nx,Ny,Nz,c0,cb,cmin,hlateral,hz,dt0,time_delay,ROI_bounds,a,False,D)
    optimized = optimize.minimize(SingleRun,initguess,bounds=parameterbounds,args=godobject,method="L-BFGS-B",options={"disp": True})
    print(optimized.x)
    initguess = optimized.x
    """

    #choose one of these based on the number of parameters you are fitting for
    #MSD, simulation_concs, simulation_times = SingleRun(initguess,V_next,C_next,raster,measurement_times,measurement_concs,Nx,Ny,Nz,c0,cb,cmin,hlateral,hz,dt0,time_delay,ROI_bounds,a,True,D)
    MSD, simulation_concs, simulation_times, measurement_concs,measurement_times = SingleRun(initguess,V_next,C_next,raster,measurement_times,measurement_concs,Nx,Ny,Nz,c0,cb,cmin,hlateral,hz,dt0,time_delay,ROI_bounds,a,True,D)
    #MSD, simulation_concs, simulation_times = SingleRunParallelised(initguess,V_next,C_next,raster,measurement_times,measurement_concs,Nx,Ny,Nz,c0,cb,cmin,hlateral,hz,dt0,ROI_bounds,a,num_cpus,True)

    print("Mean squared difference = %f" % (MSD))
    signal = np.array(simulation_concs)
    signal /= c0
    #same concept here...
    #outstring = "output-data/" + os.path.splitext(inputs[i][0].decode("utf-8"))[0] + "ke_%.3e_R_%.3e_D_%.3e_dt0_%.2f.txt" % (initguess[0]*1e4,initguess[1],initguess[2]*1e-12,dt0)
    outstring = "output-data/" + os.path.splitext(inputs[i][0].decode("utf-8"))[0] + "ke_%.3e_R_%.3e_D_%.3e_dt0_%.2f.txt" % (initguess[0]*1e4,initguess[1],D*1e-12,dt0)
    #outstring = "output-data/" + os.path.splitext(inputs[i][0].decode("utf-8"))[0] + "ke_%.3e_R_%.3e_D_%.3e_dt0_%.2f.txt" % (ke,initguess[0],D*1e-12,dt0)
    #...and here.
    #np.savetxt(outstring,np.transpose(np.vstack((simulation_times,np.square(signal)))))
    np.savetxt(outstring,np.transpose(np.vstack((simulation_times,simulation_concs))))
    #np.savetxt(outstring+"b",np.transpose(np.vstack((measurement_times,measurement_concs))))
    
    
    results = open("output-data/parameter_output_fixedD.txt","a")
    #results.write(os.path.splitext(inputs[i][0].decode("utf-8"))[0] + " MSD %.3f dt %.3f cb %.3e hlateral %.2e c0 %.3e ke %.5e R %.5e D %.5e\n" % (MSD,dt0,cb,hlateral,c0,initguess[0]*1e4,initguess[1],initguess[2]*1e-12))
    results.write(os.path.splitext(inputs[i][0].decode("utf-8"))[0] + " MSD %.3f dt %.3f cb %.3e hlateral %.2e c0 %.3e ke %.5e R %.5e D %.5e\n" % (MSD,dt0,cb,hlateral,c0,initguess[0]*1e4,initguess[1],D*1e-12))
    #results.write(os.path.splitext(inputs[i][0].decode("utf-8"))[0] + " MSD %.3f dt %.3f cb %.3e hlateral %.2e c0 %.3e ke %.5e R %.5e D %.5e\n" % (MSD,dt0,cb,hlateral,c0,ke,initguess[0],D*1e-12))
    results.close()