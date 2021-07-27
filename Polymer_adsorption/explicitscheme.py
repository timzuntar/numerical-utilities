import math
from cv2 import imread 
import numpy as np
from scipy import interpolate,optimize

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

def AdsorptionRateModifier(C_current,C_next,V_current,hz,dt,c0,Rprime,ke,a):
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
                C_next = C_current + V_current[i,j,0]*hz
                V_current[i,j,0] = 0.0
                print("Caution!")
            else:
                C_next[i,j] = C_current[i,j] + dC
                V_current[i,j,0] -= dC/hz
    return C_next,V_current

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
                    print("Negative concentration value encountered! %e -> %e" % (V_current[i,j,k],V_next[i,j,k]))
                    pathological = True
                    return V_current, pathological
    return V_next, pathological

def ExtractROIAverage(C_current,t,simulation_concs,simulation_times,ROI_bounds):
    sum = 0.0
    for i in range(ROI_bounds[0],ROI_bounds[2]+1):
        for j in range(ROI_bounds[1],ROI_bounds[3]+1):
            sum += C_current[i,j]
    sum /= math.fabs(ROI_bounds[2]+1-ROI_bounds[0])*(math.fabs(ROI_bounds[3]+1-ROI_bounds[1]))
    simulation_concs.append(sum)
    simulation_times.append(t)

def CompareSignals(measurement_times,measurement_concs,simulation_times,simulation_concs,c0):
    #Intensity is assumed to vary with square of concentration. Therefore 
    signal = np.array(simulation_concs)
    signal /= c0
    func_sim = interpolate.interp1d(simulation_times,np.square(signal),fill_value="extrapolate")
    interpolated_concs = func_sim(measurement_times)
    N = len(measurement_times)
    MSD = 0.0
    for n in range(0,N):
        MSD += (interpolated_concs[n] - measurement_concs[n])**2
    MSD /= N
    return MSD

def SingleRun(x,V_next,C_next,raster,measurement_times,measurement_concs,Nx,Ny,Nz,c0,cb,cmin,hlateral,hz,dt0,ROI_bounds):
    ke = x[0]
    Rprime = x[1]
    a = x[2]
    D = x[3]
    t = 0.0

    V_current = np.full((Nx,Ny,Nz),cb)
    C_current = np.empty((Nx,Ny))
    C_current = RasterToConcentration(C_current,raster,Nx,Ny,c0)
    simulation_concs = []
    simulation_times = []
    ExtractROIAverage(C_current,t,simulation_concs,simulation_times,ROI_bounds)

    while(C_current.min() <= cmin):
        dt = Timestep(C_current,cmin,dt0)
        C_next,V_current = AdsorptionRateModifier(C_current,C_next,V_current,hz,dt,c0,Rprime,ke,a)
        V_next,pathological = IterateTimeStep(V_current,V_next,D,Nx,Ny,Nz,dt,hlateral,hz,cb,c0)
        if (pathological == True):
            return 1e6
        print(t,C_current.min())
        np.copyto(C_current, C_next)
        np.copyto(V_current, V_next)
        t += dt
        ExtractROIAverage(C_current,t,simulation_concs,simulation_times,ROI_bounds)
    
    return 1e6*CompareSignals(measurement_times,measurement_concs,simulation_times,simulation_concs,c0)

#The approach is roughly as follows: sample/air interfaces are presented as a 2D grid, the bulk as a 3D grid;
#each cell has its own adsorbed concentration ranging from 0 to 1 (saturated).
#A cell which borders cells with high surface concentration is more likely to adsorb additional molecules, so in the event of a rupture the film
#regrowth is not entirely homogeneous. This is represented by a modified Kisliuk isotherm, also considering neighboring cells with a weighing parameter.
#Stopping condition for the simulation is concentrations in each surface cell surpassing a specified minimum limit.
#We define a ROI for space-averaging of data; a single curve is obtained.
#The curve is then fitted against experimental data. Ideally, the ROIs should be as similar as possible.
#The model is 4-parametric. If it does not converge, the diffusion constant can be fixed to a realistic value.

#the average unadsorbed area is a fraction of a millimeter across- one micron resolution should be fine enough to capture details,
#but short enough that computation time remains bearable
hlateral = 4e-6 #resolution of ellipsometry images is around 1 micron/pixel. 4x reduced resolution -> 4 microns/pixel
hz = 4e-6
c0 = 1e-6 #1 mg/square meter

dt0 = 0.1

#Initial parameter choice
#ke = 1.0/c0
ke = 2e+1
Rprime = 2e+1
#a = 0.25
a = 0.0
D = 1.1e-11   #rough approximation

initguess = [ke,Rprime,a,D]
parameterbounds = ((0.0,None),(0.0,None),(0.0,10.0),(1e-12,1e-10))

cb = 3e-2    #boundary concentration (in kg/L)
cmin = 0.99*c0    #stop iterating when every surface cell surpasses this concentration 

raster = imread("examplerasterROI1.png",0)
Ny,Nx = raster.shape
Nz = int(min(Nx,Ny)*hlateral/hz)
ROI_bounds = (53,31,59,36)  #ROI1
#ROI_bounds = (30,29,50,42)    #ROI0
print("Full size of array is %d, %d, %d" % (Nx,Ny,Nz))

out = np.loadtxt("data/exampleinputROI1.txt")
out = out[np.where(out[range(out.shape[0]),1] < 0.9)]
measurement_times = out[:,0]
measurement_concs = out[:,1]
#measurement points with c > cmin need to be removed, otherwise interpolation will fail

V_next = np.empty((Nx,Ny,Nz))
C_next = np.empty((Nx,Ny))

godobject = (V_next,C_next,raster,measurement_times,measurement_concs,Nx,Ny,Nz,c0,cb,cmin,hlateral,hz,dt0,ROI_bounds)

optimized = optimize.minimize(SingleRun,initguess,bounds=parameterbounds,args=godobject,method="L-BFGS-B",options={"maxfun": 30, "maxiter": 10, "disp": True})
print(optimized.x)

ke = optimized.x[0]
Rprime = optimized.x[1]
a = optimized.x[2]
D = optimized.x[3]

cmin = 0.999*c0
t = 0.0

V_current = np.full((Nx,Ny,Nz),cb)
C_current = np.empty((Nx,Ny))
C_current = RasterToConcentration(C_current,raster,Nx,Ny,c0)
simulation_concs = []
simulation_times = []
ExtractROIAverage(C_current,t,simulation_concs,simulation_times,ROI_bounds)

while(C_current.min() <= cmin):
    dt = Timestep(C_current,cmin,dt0)
    C_next,V_current = AdsorptionRateModifier(C_current,C_next,V_current,hz,dt,c0,Rprime,ke,a)
    V_next = IterateTimeStep(V_current,V_next,D,Nx,Ny,Nz,dt,hlateral,hz,cb,c0)
    print(t,C_current.min()/c0, V_current.min())
    np.copyto(C_current, C_next)
    np.copyto(V_current, V_next)
    t += dt
    ExtractROIAverage(C_current,t,simulation_concs,simulation_times,ROI_bounds)

signal = np.array(simulation_concs)
signal /= c0
outstring = "data/ROI1_fixed_extratime_%dpar_ke_%.3e_R_%.3e_a_%.3e_D_%.3e_dt0_%.2f.txt" % (len(initguess),ke,Rprime,a,D,dt0)
np.savetxt(outstring,np.transpose(np.vstack((simulation_times,np.square(signal)))))

results = open("data/parameter_output.txt","a")
results.write("exampleinputROI1 dt %.3f cb %.3e hlateral %.2e c0 %.3e ke %.5e R %.5e a %.5e D %.5e\n" % (dt0,cb,hlateral,c0,ke,Rprime,a,D))
results.close()