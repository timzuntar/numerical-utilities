import cmath
import math
import numpy as np
import random
from scipy import constants, special

def IntensityGradient(vector, I0, z0, w0, onedim=False):
    #computes the field intensity and gradients of a Gaussian beam with waist at (0,0,0)
    #and propagating along the positive z-axis.

    temp1 = (1 + (vector[2]/z0)**2)
    w2 = (w0**2) * temp1
    r2 = vector[0]**2 + vector[1]**2
    temp2 = math.exp(-2 * r2 / w2)
    I = I0 * temp2 * (w0**2) / w2

    if (onedim == True):
        gradx = 0.0
        grady = 0.0
        gradz = I0 * temp2 * (1/(z0**2 * temp1**2)) * ((4*r2*vector[2])/w2 - 2*vector[2])
        
    else:
        temp3 = -4 * I0 * (w0**2 / w2**2) * temp2

        gradx = vector[0] * temp3
        grady = vector[1] * temp3
        gradz = I0 * temp2 * (1/(z0**2 * temp1**2)) * ((4*r2*vector[2])/w2 - 2*vector[2])

    return I, [gradx,grady,gradz]

def RayleighLimitEval(vector, I0, z0, w0, n0, n1, wavelength, radius, onedim=False):
    #evaluates gradient and scattering forces of a homogeneous spherical particle in said Gaussian beam,
    #in the Rayleigh approximation

    k = 2 * math.pi / wavelength
    refractive_term = ((n1/n0)**2 - 1) / ((n1/n0)**2 + 2)
    I, gradient = IntensityGradient(vector, I0, z0, w0, onedim)
    gradient_term = refractive_term * 2 * math.pi * n1 * (radius**3) / constants.c
    scattering_term = refractive_term**2 * 8 * math.pi * n1 * (k**4) * (radius**6) / (3 * constants.c)
    
    Fgrad = [-gradient_term * gradient[0], -gradient_term * gradient[1], -gradient_term * gradient[2]]
    Fscat = [0, 0, -scattering_term * I]

    return Fgrad, Fscat

def UpdatePosition(vector, I0, z0, w0, n0, n1, wavelength, radius, rhomedium, rhoparticle, D, Brownconst, t, dt, onedim=False):
    #propagates equation of motion one timestep for Rayleigh approximation. Time parameter could be used to implement trap movement or switching
    
    Fgrad, Fscat = RayleighLimitEval(vector, I0, z0, w0, n0, n1, wavelength, radius, onedim)
    #assuming gravity acts along z-axis
    Fg = [0, 0, 4 * math.pi * constants.g * (radius**3) * (rhomedium - rhoparticle)]
    noise = [random.normalvariate(0.0,1.0),random.normalvariate(0.0,1.0),random.normalvariate(0.0,1.0)]
    F = [Fgrad[0] + Fscat[0] + Fg[0], Fgrad[1] + Fscat[1] + Fg[1], Fgrad[2] + Fscat[2] + Fg[2]]

    dr = [-dt*F[0]/D + noise[0]*Brownconst, -dt*F[1]/D + noise[1]*Brownconst, -dt*F[2]/D + noise[2]*Brownconst]

    return F, dr

def TrappingEfficiency(Fz, n1, I0, w0):
    #ratio of exerted force to beam momentum

    P0 = math.pi * I0 * w0**2 / 2
    return constants.c * Fz / (n1 * P0)

def ShapeCoeffs(l, zoffset, w0, n0, wavelength):
    #Lorenz-Mie coefficients of a modified localized approximation of a Gaussian beam as per Lock, 2004
    #Only 1 distinct value in this edge case - function returns two values for future compatibility

    s_i = wavelength / (n0 * 2 * math.pi * w0)
    gl = cmath.exp(-1j * n0 * zoffset * 2 * math.pi / wavelength)
    gl *= cmath.exp(-(s_i**2) * (l+2) * (l-1) / (1 - 1j * 2 * s_i * zoffset / w0))
    gl /= (1 - 1j * 2 * s_i * zoffset / w0)

    return gl, gl

def ScatteringAmplitudes(l, n0, n1, wavelength, radius):
    #Mie partial wave scattering amplitudes for a homogeneous sphere as given by Lock, 2004

    X = n0 * radius * 2 * math.pi / wavelength
    Y = n1 * X / n0
    al = X * special.spherical_jn(l,X) * RBPsiDv(l,Y) - (n1/n0) * RBPsiDv(l,X) * Y * special.spherical_jn(l,Y)
    al /= RBZeta(l,X)*RBPsiDv(l,Y) - (n1/n0) * RBZetaDv(l,X) * Y * special.spherical_jn(l,Y)
    bl = (n1/n0) * X * special.spherical_jn(l,X) * RBPsiDv(l,Y) - RBPsiDv(l,X) * Y * special.spherical_jn(l,Y)
    bl /= (n1/n0) * RBZeta(l,X) * RBPsiDv(l,Y) - RBZetaDv(l,X) * Y * special.spherical_jn(l,Y)

    return al, bl

def RBZeta(l, X):
    #helper function to keep line length manageable, evaluates Ricatti-Bessel zeta function
    return X*(special.spherical_jn(l,X) + 1j * special.spherical_yn(l,X))

def RBPsiDv(l,X):
    #helper function to keep line length manageable, evaluates derivative of Ricatti-Bessel psi function
    return special.spherical_jn(l,X) + X * special.spherical_jn(l,X,derivative=True)

def RBZetaDv(l, X):
    #helper function to keep line length manageable, evaluates derivative of Ricatti-Bessel zeta function
    temp1 = special.spherical_jn(l,X) + 1j * special.spherical_yn(l,X)
    temp2 = X * (special.spherical_jn(l,X,derivative=True) + 1j * special.spherical_yn(l,X,derivative=True))
    return temp1 + temp2

def LorenzMieTerm(l, zoffset, w0, n0, n1, wavelength, radius):
    #Actually computes the coefficient sum (again, per Lock 2004)

    gl, hl = ShapeCoeffs(l,zoffset,w0,n0,wavelength)
    glplus, hlplus = ShapeCoeffs(l+1,zoffset,w0,n0,wavelength)
    al, bl = ScatteringAmplitudes(l,n0,n1,wavelength,radius)
    alplus, blplus = ScatteringAmplitudes(l+1,n0,n1,wavelength,radius)
    Ul = al + np.conj(alplus) - 2 * al * np.conj(alplus)
    Vl = bl + np.conj(blplus) - 2 * bl * np.conj(blplus)
    Wl = al + np.conj(bl) - 2 * al * np.conj(bl)

    temp1 = gl * np.conj(glplus) * Ul + np.conj(gl) * glplus * np.conj(Ul) + hl * np.conj(hlplus) * Vl + np.conj(hl) * hlplus * np.conj(Vl)
    temp1 *= l * (l + 2) / (l + 1)
    temp2 = gl * np.conj(hl) * Wl + np.conj(gl) * hl * np.conj(Wl)
    temp2 *= (2*l + 1)/(l * (l + 1))
    return temp1 + temp2

def LorenzMieEval(I0, zoffset, w0, n0, n1, wavelength, radius, lmax):
    #computes optical force in z-direction for a homogeneous sphere positioned in the axis of a near-Gaussian beam.
    #Truncation at l=10 should give a decent approximation

    coeffsum = 0.0
    for l in range(1,lmax+1):
        coeffsum += LorenzMieTerm(l,zoffset,w0, n0, n1, wavelength, radius)

    return [0,0,np.real(I0 * wavelength**2 / (4 * math.pi * (constants.c)**2 * constants.mu_0) * coeffsum) * (constants.c * constants.mu_0 / n0)]

def UpdatePositionLM(vector, I0, z0, w0, n0, n1, wavelength, radius, rhomedium, rhoparticle, D, Brownconst, t, dt, lmax):
    #propagates equation of motion one timestep for LM axial model.

    Fopt = LorenzMieEval(I0, vector[2], w0, n0, n1, wavelength, radius, lmax)
    #assuming gravity acts along z-axis
    Fg = 4 * math.pi * constants.g * (radius**3) * (rhomedium - rhoparticle)
    noise = random.normalvariate(0.0,1.0)

    dr = [0, 0, -dt*(Fopt[2] + Fg)/D + noise*Brownconst]

    return Fopt, dr

def SingleRayForce(theta, n0, n1, dP):
    #forces that a single ray (of power dP) exerts on the sphere. Position of Z and Y axes is relative to the incoming ray.
    #The ray is assumed to be circularily polarized, therefore the effective Fresnel reflection coefficient is the average
    #of values for both polarizations

    r = math.asin(math.sin(theta) * n0 / n1)
    Rs = (math.fabs((n0 * math.cos(theta) - n1 * math.cos(r))/(n0 * math.cos(theta) + n1 * math.cos(r))))**2
    Rp = (math.fabs((n0 * math.cos(r) - n1 * math.cos(theta))/(n0 * math.cos(r) + n1 * math.cos(theta))))**2
    R = (Rs + Rp)/2.0
    T = 1.0-R

    numerator = 1 + R**2 + 2 * R * math.cos(2*r)
    Fz = 1 + R * math.cos(2*theta) - (T**2) * (math.cos(2*theta - 2*r) + R * math.cos(2*theta)) / numerator
    Fy = - R * math.sin(2*theta) + (T**2) * (math.sin(2*theta - 2*r) + R * math.sin(2*theta)) / numerator

    return Fz * n1 * dP / constants.c, Fy * n1 * dP / constants.c

def AnnulusCoords(zoffset, radius, w0, phi):
    #computes the geometry of an xy-plane annulus for incoming rays around angle theta (ray optics approximation,
    #beam waist is approximated with a point)

    if (zoffset < 0):    #particle located before the beam waist
        u = math.fabs(math.fabs(zoffset) * math.cos(phi) + math.sqrt(zoffset**2 * (math.cos(phi)**2 - 1) + radius**2))
    elif (zoffset >= 0):  #particle beyond beam waist
        u = math.fabs(math.fabs(zoffset) * math.cos(phi) - math.sqrt(zoffset**2 * (math.cos(phi)**2 - 1) + radius**2))
    if (u > math.fabs(zoffset) + radius or u < math.fabs(zoffset) - radius):
        print("\nUh-oh")
    
    z = math.cos(phi) * u
    if (zoffset < radius):
        if ((zoffset + math.cos(phi)*u)/radius <= -1):
            vartheta = -math.pi
        else:
            vartheta = math.acos((zoffset + math.cos(phi)*u)/radius)
    elif (zoffset >= radius):
        if ((zoffset - math.cos(phi)*u)/radius >= 1):
            vartheta = math.pi
        else:
            vartheta = math.acos((zoffset - math.cos(phi)*u)/radius)
    if (zoffset < 0):
        theta = vartheta - phi
    elif (zoffset >= 0 and zoffset < radius):
        theta = phi - vartheta
    elif (zoffset >= radius):
        theta = vartheta + phi

    return u, theta, vartheta, z

def AnnulusPowerGauss(philower, phiupper, I0, z0, w0, z):
    #Fraction of beam power contained in annulus between the two angles at distance z from waist (for Gaussian beam)

    r1 = z / (math.pi/2 - math.atan(philower))
    r2 = z / (math.pi/2 - math.atan(phiupper))
    w2 = w0**2 * (1 + (z/z0)**2)

    return (math.pi * I0 * w0 / 2) * (math.exp(- 2 * (r1**2) / w2) - math.exp(- 2 * (r2**2) / w2))

def AnnulusPower(philower, phiupper, I0, z0, w0, z):
    #Fraction of beam power contained in annulus between the two angles at distance z from waist (for ray optics approx.)

    r1 = z / (math.pi/2 - math.atan(philower))
    r2 = z / (math.pi/2 - math.atan(phiupper))
    w2 = w0**2 * ((z/z0)**2)

    return (math.pi * I0 * w0 / 2) * (math.exp(- 2 * (r1**2) / w2) - math.exp(- 2 * (r2**2) / w2))


def RayOpticsEval(I0, zoffset, w0, n0, n1, wavelength, radius, nphi):
    #computes optical force in z-direction for a homogeneous sphere positioned in the axis of a Gaussian beam
    #in the ray optics approximation. Final parameter determines fineness of angular discretization.

    divangle = wavelength / (math.pi * w0)
    if (math.fabs(zoffset) / radius <= 1.0):
        phimax = 4*divangle
    else:
        phimax = min(4*divangle, math.asin(radius / math.fabs(zoffset)))
    #write down central angles of each ray annulus (relative to waist)
    philist = []
    for i in range(nphi):
        philist.append((i+0.5) * phimax/nphi)

    P = 0.0
    Fopt = 0.0
    Fztotal = 0.0
    Fytotal = 0.0
    for t in range(nphi):
        u, theta, vartheta, zannulus = AnnulusCoords(zoffset, radius, w0, philist[t])
        dP = AnnulusPower(philist[t] - 0.5 * phimax/nphi , philist[t] + 0.5 * phimax/nphi, I0, z0, w0, zannulus)
        P += dP
        Fz, Fy = SingleRayForce(theta, n0, n1, dP)
        if (zoffset < 0):
            Fopt += Fz * math.cos(philist[t]) - Fy * math.sin(philist[t])
            Fztotal += Fz * math.cos(philist[t])
            Fytotal -= Fy * math.sin(philist[t])            
        elif (zoffset >= 0):
            Fopt += Fz * math.cos(philist[t]) + Fy * math.sin(philist[t])
            Fztotal += Fz * math.cos(philist[t])
            Fytotal += Fy * math.sin(philist[t])

    return [Fztotal,Fytotal,Fopt]

def UpdatePositionRayOptics(vector, I0, z0, w0, n0, n1, wavelength, radius, rhomedium, rhoparticle, D, Brownconst, t, dt, nphi):
    #propagates equation of motion one timestep for axial model of ray optics.

    Fopt = RayOpticsEval(I0, vector[2], w0, n0, n1, wavelength, radius, nphi)
    #assuming gravity acts along z-axis
    Fg = 4 * math.pi * constants.g * (radius**3) * (rhomedium - rhoparticle)
    noise = random.normalvariate(0.0,1.0)

    dr = [0, 0, -dt*(Fopt[2] + Fg)/D + noise*Brownconst]

    return Fopt, dr
    
def zForceSlice(zstart,zend,numz,I0,z0,w0,n0,n1,wavelength,radius,lmax,nphi):
    #computes the z-axis force at equally spaced points in chosen interval.
    #Skips positions too close to beam waist to avoid accidental zero division

    data = np.empty((numz,5))
    dz = (zend-zstart)/numz
    ipathologic = []
    for i in range(numz):
        z = zstart + i*dz
        if (math.fabs(z) < 1e-9):
            ipathologic.append(i)
            continue
        else:
            data[i,0] = z
            RayleighFgrad, RayleighFscat = RayleighLimitEval([0,0,z],I0,z0,w0,n0,n1,wavelength,radius,onedim=True)
            LMF = LorenzMieEval(I0,z,w0,n0,n1,wavelength,radius,lmax)
            RayF = RayOpticsEval(I0,z,w0,n0,n1,wavelength,radius,nphi)
            data[i,1] = RayleighFgrad[2]
            data[i,2] = RayleighFscat[2]
            data[i,3] = RayF[0]
            data[i,4] = RayF[1]
        
    return np.delete(data, ipathologic, axis=0)

def zPosSlice(zstart,zend,numz,I0,z0,wlist,n0,n1,wavelength,radius,lmax,nphi,folder="Rezultati/",method="Rayleigh"):
    #calculates axial force profile for all beam waist values from list and approximate locations of zeros 
    #writes everything to file

    dz = (zend-zstart)/numz
    
    if (method=="Rayleigh"):
        f = open(folder+"zeros_Rayleigh_a_%.2f_um_n0_%.2f_n1_%.2f.dat" % (radius*1e6,n0,n1),"w+")
    elif (method=="LM"):
        f = open(folder+"zeros_LM_a_%.2f_um_n0_%.2f_n1_%.2f.dat" % (radius*1e6,n0,n1),"w+")
    elif (method=="Ray"):
        f = open(folder+"zeros_Ray_a_%.2f_um_n0_%.2f_n1_%.2f.dat" % (radius*1e6,n0,n1),"w+")
    else:
        return False
    
    for w0 in wlist:
        z0 = math.pi * (w0**2) / wavelength

        if (method=="Rayleigh"):
            RayleighFgrad, RayleighFscat = RayleighLimitEval([0,0,zstart],I0,z0,w0,n0,n1,wavelength,radius,onedim=True)
            Fprev = RayleighFgrad+RayleighFscat
        elif (method=="LM"):
            Fprev = LorenzMieEval(I0,zstart,w0,n0,n1,wavelength,radius,lmax)
        elif (method=="Ray"):
            Fprev = RayOpticsEval(I0,zstart,w0,n0,n1,wavelength,radius,nphi)

        maxF = Fprev[2]
        minF = Fprev[2]
        f.write("%.5e %.5e " % (radius,w0))
        ipathologic = []
        for i in range(1,numz):
            z = zstart + i*dz
            if (math.fabs(z) < 1e-9):
                ipathologic.append(i)
                continue
            else:
                if (method=="Rayleigh"):
                    RayleighFgrad, RayleighFscat = RayleighLimitEval([0,0,z],I0,z0,w0,n0,n1,wavelength,radius,onedim=True)
                    Fcurrent = RayleighFgrad+RayleighFscat
                elif (method=="LM"):
                    Fcurrent = LorenzMieEval(I0,z,w0,n0,n1,wavelength,radius,lmax)
                elif (method=="Ray"):
                    Fcurrent = RayOpticsEval(I0,z,w0,n0,n1,wavelength,radius,nphi)
                
                if (Fcurrent[2] > maxF):
                    maxF = Fcurrent[2]
                if (Fcurrent[2] < minF):
                    minF = Fcurrent[2]
                if ((Fprev[2] > 0 and Fcurrent[2] < 0) or (Fprev[2] < 0 and Fcurrent[2] > 0)):
                    f.write("%f " % ((2*z-dz)/2))
                Fprev = Fcurrent
        
        f.write("%.10e %.10e\n" % (maxF,minF))

    f.close()
    return True


#material parameters
radius = 5e-6         #particle diameter is 100 nm
eta = 8.9e-4            #dynamic viscosity of water at 25C
T = 298
n0 = 1.33               #refractive index of surrounding medium
n1 = 1.5                #refractive index of particle
rhomedium = 1           #water
rhoparticle = 2.65      #silica

#beam parameters
wavelength = 1064e-9    #standard infrared Nd:YAG
w0 = 0.1e-6             #beam waist of 1 micron diameter
z0 = math.pi * (w0**2) / wavelength
P = 0.1                 #100 mW
I0 = 2*P/(math.pi * w0**2)
#I0 = 3e12               #approximately 80 mW power for chosen w0
#I0 = 0.0                #test code validity without beam

#simulation parameters
dt = 1e-4
tmax = 1.0
lmax = 10
nphi = 100
random.seed(1)          #if we wish for Brownian motion to be repeatable

D = 6 * math.pi * eta * radius
Brownconst = math.sqrt(2 * constants.Boltzmann * T * dt / D)
momentum = n1 * constants.h / wavelength


""" r0 = [1e-6,0.0,2e-6]
F = [0.0,0.0,0.0]

multipliers = open("Rezultati/Rayleigh_beam_test_dt_%.2e.dat" % (dt),"w+")
multipliers.write(f"# radius {radius:.3e} eta {eta:.2e} T {T:d} n0 {n0:.2f} n1 {n1:.2f} rhomedium {rhomedium:.2f} rhoparticle {rhoparticle:.2f}\n")
multipliers.write(f"# lambda {wavelength:.4e} w0 {w0:.2e} z0 {z0:.3e} I0 {I0:.2e}\n\n")

t = 0.0
r = r0

while(t < tmax):
    multipliers.write(f"{t:.5f} {r[0]:.12f} {r[1]:.12f} {r[2]:.12f} {F[0]:.6e} {F[1]:.6e} {F[2]:.6e}\n")
    F, dr = UpdatePosition(r, I0, z0, w0, n0, n1, wavelength, radius, rhomedium, rhoparticle, D, Brownconst, t, dt)
    r = np.add(r,dr)
    t += dt

multipliers.close() """


""" multipliers = open("Rezultati/LM_beam_test_dt_%.2e.dat" % (dt),"w+")
multipliers.write(f"# radius {radius:.3e} eta {eta:.2e} T {T:d} n0 {n0:.2f} n1 {n1:.2f} rhomedium {rhomedium:.2f} rhoparticle {rhoparticle:.2f}\n")
multipliers.write(f"# lambda {wavelength:.4e} w0 {w0:.2e} z0 {z0:.3e} I0 {I0:.2e}\n\n")

t = 0.0
r = [0,0,0]
tmax = 0.5
F = [0.0,0.0,0.0]

while(t < tmax):
    multipliers.write(f"{t:.5f} {r[2]:.12f} {F[2]:.6e}\n")
    F, dr = UpdatePositionLM(r, I0, z0, w0, n0, n1, wavelength, radius, rhomedium, rhoparticle, D, Brownconst, t, dt, lmax)
    r = np.add(r,dr)
    t += dt
    print("%.4f" % (t))

multipliers.close() """


""" multipliers = open("Rezultati/Rayoptics_beam_test_dt_%.2e.dat" % (dt),"w+")
multipliers.write(f"# radius {radius:.3e} eta {eta:.2e} T {T:d} n0 {n0:.2f} n1 {n1:.2f} rhomedium {rhomedium:.2f} rhoparticle {rhoparticle:.2f}\n")
multipliers.write(f"# lambda {wavelength:.4e} w0 {w0:.2e} z0 {z0:.3e} I0 {I0:.2e}\n\n")

t = 0.0
r = [0,0,1e-6]
tmax = 0.5
F = [0.0,0.0,0.0]

while(t < tmax):
    multipliers.write(f"{t:.5f} {r[2]:.12f} {F[2]:.6e}\n")
    F, dr = UpdatePositionRayOptics(r, I0, z0, w0, n0, n1, wavelength, radius, rhomedium, rhoparticle, D, Brownconst, t, dt, nphi)
    r = np.add(r,dr)
    t += dt
    print("%.4f" % (t))

multipliers.close() """


#Here we generate force profiles along the z-axis in a sample trap for all 3 approaches.
zstart = -50e-6
zend = 50e-6
numz = 2001

forces = zForceSlice(zstart,zend,numz,I0,z0,w0,n0,n1,wavelength,radius,lmax,nphi)
fileslice = open("Rezultati/zslice_comparison_a_%.2f_um_waist_%.2f_um.dat" % (radius*1e6,w0*1e6),"w+")
fileslice.write(f"# radius {radius:.3e} I0 {I0:.5f} n0 {n0:.2f} n1 {n1:.2f} LM: lmax {lmax:d} Ray: nphi {nphi:d}\n")
np.savetxt(fileslice, forces)
fileslice.close()

""" filesingleray = open("Rezultati/single_ray_n1_%.2f_n0_%.2f.dat" % (n1,n0),"w+")
for i in range(0,181):
    Fz,Fy = SingleRayForce(math.radians(i/2),n0,n1,1.0)
    filesingleray.write("%f %.10e %.10e\n" % (i/2,Fz,Fy))
filesingleray.close() """

""" wlist = []
for w in range(1,1000):
    wlist.append(w*0.01*1e-6)
zstart = -5e-6
zend = 5e-6
numz = 10000
zPosSlice(zstart,zend,numz,I0,z0,wlist,n0,n1,wavelength,radius,lmax,nphi,"Rezultati/","LM") """

""" #Sanity check of annulus coordinate function
zstart = -1.4e-6
zend = 1.5e-6
numz = 2000
dz = (zend-zstart)/numz
divangle = wavelength / (math.pi * w0)
print(divangle)
fileannulustest = open("Rezultati/raytest.dat","w+")
fileannulustest.write(f"# radius {radius:.4e} w0 {w0:.2e}\n\n")
for i in range(numz):
    u,theta,vartheta,z = AnnulusCoords(zstart+i*dz, radius, w0, divangle)
    fileannulustest.write("%e %e %e %e %e\n" % (zstart+i*dz,u,theta,vartheta,z))
fileannulustest.close() """