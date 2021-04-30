import cmath
import math
import numpy as np
import random
from scipy import constants, special, fft, optimize, integrate, signal

def IntensityGradient(vector, I0, z0, w0, onedim=False):
    #computes the field intensity and gradients of a Gaussian beam with waist at (0,0,0)
    #and propagating along the positive z-axis direction.

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

    k = 2 * n0 * math.pi / wavelength
    refractive_term = ((n1/n0)**2 - 1) / ((n1/n0)**2 + 2)
    I, gradient = IntensityGradient(vector, I0, z0, w0, onedim)
    gradient_term = refractive_term * 2 * math.pi * n0 * (radius**3) / constants.c
    scattering_term = refractive_term**2 * 8 * math.pi * n0 * (k**4) * (radius**6) / (3 * constants.c)
    
    Fgrad = [-gradient_term * gradient[0], -gradient_term * gradient[1], -gradient_term * gradient[2]]
    Fscat = [0, 0, scattering_term * I]

    return Fgrad, Fscat

def UpdatePosition(vector, I0, z0, w0, n0, n1, wavelength, radius, rhomedium, rhoparticle, D, Brownconst, t, dt, onedim=False):
    #propagates equation of motion one timestep for Rayleigh approximation. Time parameter could be used to implement trap movement or switching
    
    Fgrad, Fscat = RayleighLimitEval(vector, I0, z0, w0, n0, n1, wavelength, radius, onedim)
    #assuming gravity acts along z-axis
    Fg = [0, 0, 4 * math.pi * constants.g * (radius**3) * (rhomedium - rhoparticle)]
    noise = [random.normalvariate(0.0,1.0),random.normalvariate(0.0,1.0),random.normalvariate(0.0,1.0)]
    if (onedim == False):
        F = [Fgrad[0] + Fscat[0] + Fg[0], Fgrad[1] + Fscat[1] + Fg[1], Fgrad[2] + Fscat[2] + Fg[2]]
        dr = [-dt*F[0]/D + noise[0]*Brownconst, -dt*F[1]/D + noise[1]*Brownconst, -dt*F[2]/D + noise[2]*Brownconst]
    else:
        F = [0,0,Fgrad[2] + Fscat[2] + Fg[2]]
        dr = [0,0,-dt*F[2]/D + noise[2]*Brownconst]

    return F, dr

def TrappingEfficiency(Fz, n1, I0, w0):
    #ratio of exerted force to beam momentum

    P0 = math.pi * I0 * w0**2 / 2
    return constants.c * Fz / (n1 * P0)

def ShapeCoeffs(l, zoffset, w0, n0, wavelength):
    #Lorenz-Mie coefficients of a modified localized approximation of a Gaussian beam as per Lock, 2004
    #Only 1 distinct value in this edge case - function returns two values for future compatibility

    s_i = wavelength / (n0 * 2 * math.pi * w0)
    gl = np.exp(np.complex(0, -n0 * zoffset * 2 * math.pi / wavelength))
    gl *= np.exp(-(s_i**2) * (l+2) * (l-1) / np.complex(1, -2 * s_i * zoffset / w0))
    gl /= np.complex(1,-2 * s_i * zoffset / w0)

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

def PrecomputeAmplitudes(lmax, n0, n1, wavelength, radius):
    coeffs = np.zeros((lmax+1,2),dtype=np.complex)
    for l in range(1,lmax+2):
        coeffs[l-1,0],coeffs[l-1,1] = ScatteringAmplitudes(l,n0,n1,wavelength,radius)
    return coeffs

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

def LorenzMieTerm(l, zoffset, w0, n0, n1, wavelength, radius, precomputed=None):
    #Actually computes the coefficient sum (again, per Lock 2004)

    gl, hl = ShapeCoeffs(l,zoffset,w0,n0,wavelength)
    glplus, hlplus = ShapeCoeffs(l+1,zoffset,w0,n0,wavelength)
    if precomputed is not None:
        al = precomputed[l-1,0]
        bl = precomputed[l-1,1]
        alplus = precomputed[l,0]
        blplus = precomputed[l,1]
    else:
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

def LorenzMieEval(I0, zoffset, w0, n0, n1, wavelength, radius, lmax, precomputed=None):
    #computes optical force in z-direction for a homogeneous sphere positioned in the axis of a near-Gaussian beam.
    #Truncation at l=10 should give a decent approximation

    coeffsum = 0.0
    for l in range(1,lmax+1):
        coeffsum += LorenzMieTerm(l,zoffset,w0, n0, n1, wavelength, radius,precomputed)

    #testing of Rayleigh approx. equivalency
    #gl, hl = ShapeCoeffs(1,zoffset,w0,n0,wavelength)
    #glplus, hlplus = ShapeCoeffs(2,zoffset,w0,n0,wavelength)
    #al, bl = ScatteringAmplitudes(1,n0,n1,wavelength,radius)
    #X = n0 * radius * 2 * math.pi / wavelength
    #m = n1/n0
    #s_i = wavelength / (n0 * 2 * math.pi * w0)
    #al = np.complex(0,-2.0/3) * (X**3) * ((m**2-1)/(m**2+2))
    #al -= np.complex(0,2.0/5) * (X**5) * (m**2-1) * (m**2-2) / ((m**2+2)**2)
    #al += np.complex((4.0/9) * (X**6) * (((m**2-1)/(m**2+2))**2),0)

    #coeffsum = 1.5 * (gl * np.conj(glplus) * al + np.conj(gl) * glplus * np.conj(al) + gl * np.conj(hl) * al + np.conj(gl) * hl * np.conj(al))

    return [0,0,np.real(I0 * (wavelength**2 / (4 * math.pi * (constants.c)**2 * constants.mu_0)) * coeffsum) * (constants.c * constants.mu_0 / n0)]

def UpdatePositionLM(vector, I0, z0, w0, n0, n1, wavelength, radius, rhomedium, rhoparticle, D, Brownconst, t, dt, lmax, precomputed=None):
    #propagates equation of motion one timestep for LM axial model.

    Fopt = LorenzMieEval(I0, vector[2], w0, n0, n1, wavelength, radius, lmax,precomputed)
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
    #Check for errors?

    r1 = z / (math.pi/2 - math.atan(philower))
    r2 = z / (math.pi/2 - math.atan(phiupper))
    w2 = w0**2 * (1 + (z/z0)**2)

    return (math.pi * I0 * (w0**2) / 2) * (math.exp(- 2 * (r1**2) / w2) - math.exp(- 2 * (r2**2) / w2))

def AnnulusPower(philower, phiupper, I0, z0, w0, z):
    #Fraction of beam power contained in annulus between the two angles at distance z from waist (for ray optics approx.)

    #r1 = z / (math.pi/2 - math.atan(philower))
    #r2 = z / (math.pi/2 - math.atan(phiupper))
    r1 = z * math.tan(philower)
    r2 = z * math.tan(phiupper)
    w2 = w0**2 * ((z/z0)**2)

    return (math.pi * I0 * (w0**2) / 2) * (math.exp(- 2 * (r1**2) / w2) - math.exp(- 2 * (r2**2) / w2))


def RayOpticsEval(I0, zoffset, w0, n0, n1, wavelength, radius, nphi):
    #computes optical force in z-direction for a homogeneous sphere positioned in the axis of a Gaussian beam
    #in the ray optics approximation. Final parameter determines fineness of angular discretization.

    divangle = wavelength / (math.pi * w0)
    if (math.fabs(zoffset) / radius <= 1.0):
        phimax = min(4*divangle,math.pi/2)
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
    #computes the z-axis force and its integral at equally spaced points in chosen interval.
    #Skips positions too close to beam waist to avoid accidental zero division

    data = np.empty((numz,8))
    dz = (zend-zstart)/numz
    ipathologic = []
    coeffs = PrecomputeAmplitudes(lmax,n0,n1,wavelength,radius)

    for i in range(numz):
        z = zstart + i*dz
        if (math.fabs(z) < 1e-10):
            ipathologic.append(i)
            continue
        else:
            data[i,0] = z
            RayleighFgrad, RayleighFscat = RayleighLimitEval([0,0,z],I0,z0,w0,n0,n1,wavelength,radius,onedim=True)
            LMF = LorenzMieEval(I0,z,w0,n0,n1,wavelength,radius,lmax,precomputed=coeffs)
            RayF = RayOpticsEval(I0,z,w0,n0,n1,wavelength,radius,nphi)
            data[i,1] = RayleighFgrad[2]
            data[i,2] = RayleighFscat[2]
            data[i,3] = LMF[2]
            data[i,4] = RayF[2]

    data = np.delete(data, ipathologic, axis=0)
    newlength = np.shape(data)[0]
    data[0,5] = 0.0
    data[0,6] = 0.0
    data[0,7] = 0.0
    for i in range (1,newlength):
        data[i,5] = integrate.simps(np.add(data[:i,1],data[:i,2]), data[:i,0])
        data[i,6] = integrate.simps(data[:i,3], data[:i,0])
        data[i,7] = integrate.simps(data[:i,4], data[:i,0])

    return data

def zParSlice(zstart,zend,numz,T,I0,z0,wlist,n0,n1,wavelength,radius,lmax,nphi,folder="Rezultati/",method="Rayleigh"):
    #calculates axial force profiles and approximate locations of local minima 
    #writes everything to file

    dz = (zend-zstart)/numz
    
    if (method=="Rayleigh"):
        f = open(folder+"zeros_Rayleigh_a_%.2f_um_n0_%.2f_n1_%.2f.dat" % (radius*1e6,n0,n1),"a+")
        fwells = open(folder+"welldepths_Rayleigh_a_%.2f_um_n0_%.2f_n1_%.2f.dat" % (radius*1e6,n0,n1),"a+")
    elif (method=="LM"):
        f = open(folder+"zeros_LM_a_%.2f_um_n0_%.2f_n1_%.2f.dat" % (radius*1e6,n0,n1),"a+")
        fwells = open(folder+"welldepths_LM_a_%.2f_um_n0_%.2f_n1_%.2f.dat" % (radius*1e6,n0,n1),"a+")
        #only valid for variation of beam waist, not radius!
        coeffs = PrecomputeAmplitudes(lmax,n0,n1,wavelength,radius)
    elif (method=="Ray"):
        f = open(folder+"zeros_Ray_a_%.2f_um_n0_%.2f_n1_%.2f.dat" % (radius*1e6,n0,n1),"a+")
        fwells = open(folder+"welldepths_Ray_a_%.2f_um_n0_%.2f_n1_%.2f.dat" % (radius*1e6,n0,n1),"a+")
    else:
        return False
    
    #compute axial forces for every value of parameter
    for w0 in wlist:
        z0 = math.pi * (w0**2) * n0 / wavelength

        data = np.empty((numz,3))
        ipathologic = []

        for i in range(numz):
            z = zstart + i*dz
            if (math.fabs(z) < 1e-10):
                ipathologic.append(i)
                continue
            else:
                if (method=="Rayleigh"):
                    RayleighFgrad, RayleighFscat = RayleighLimitEval([0,0,z],I0,z0,w0,n0,n1,wavelength,radius,onedim=True)
                    Fcurrent = RayleighFgrad[2]+RayleighFscat[2]
                elif (method=="LM"):
                    Ftemp = LorenzMieEval(I0,z,w0,n0,n1,wavelength,radius,lmax,precomputed=coeffs)
                    Fcurrent = Ftemp[2]
                elif (method=="Ray"):
                    Ftemp = RayOpticsEval(I0,z,w0,n0,n1,wavelength,radius,nphi)
                    Fcurrent = Ftemp[2]
                data[i,0] = z
                data[i,1] = Fcurrent

        data = np.delete(data, ipathologic, axis=0)
        newlength = np.shape(data)[0]
        data[0,2] = 0.0
        for i in range (1,newlength):
            data[i,2] = integrate.simps(data[:i,1], data[:i,0])

        Fmax = np.max(data[:,1])
        Fmin = np.min(data[:,1])
        minlist = signal.argrelextrema(data[:,2], np.less, order=3)
        lowerbound_min = 1.1
        maxlist = signal.argrelextrema(data[:,2], np.greater, order=3)
        lowerbound_max = 1.1
        if any(map(len, maxlist)):
            for element in maxlist:
                if (data[element[0],2] < lowerbound_max):
                    lowerbound_max = data[element[0],2]
                    lowerbound_max_pos = data[element[0],0]

        f.write("%.5e %.5e %.8e %.8e" % (radius,w0, Fmax, Fmin))
        if any(map(len, minlist)):
            for element in minlist:
                f.write(" %.6e %.12e" % (data[element[0],0],data[element[0],2]))
                print("w0 = %e found %d local minima, %d maxima." % (w0, len(minlist), len(maxlist)))
                if (data[element[0],2] < lowerbound_min):
                    lowerbound_min = data[element[0],2]
                    lowerbound_min_pos = data[element[0],0]
        else:
            print("w0 = %e found 0 local minima." % (w0))

        if (lowerbound_max < 1 and lowerbound_min < lowerbound_max):
            welldepth = (lowerbound_max - lowerbound_min)/4.11433402e-21    #give well depth in units of kT
            fwells.write("%.5e %.5e %.6e %.6e %.6e %.6e %.6e\n" % (radius,w0, lowerbound_min_pos, lowerbound_max_pos, lowerbound_min, lowerbound_max, welldepth))
        else:
            fwells.write("%.5e %.5e 0.0 0.0 0.0 0.0 0.0\n" % (radius,w0))

        f.write("\n")

    f.close()
    fwells.close()
    return True

def zRadiusSlice(zstart,zend,numz,T,I0,z0,w0,n0,n1,wavelength,radiuslist,lmax,nphi,folder="Rezultati/",method="Rayleigh"):
    #calculates axial force profiles and approximate locations of local minima 
    #writes everything to file

    dz = (zend-zstart)/numz
    
    if (method=="Rayleigh"):
        f = open(folder+"zeros_Rayleigh_w_%.2f_um_n0_%.2f_n1_%.2f.dat" % (w0*1e6,n0,n1),"a+")
        fwells = open(folder+"welldepths_Rayleigh_w_%.2f_um_n0_%.2f_n1_%.2f.dat" % (w0*1e6,n0,n1),"a+")
    elif (method=="LM"):
        f = open(folder+"zeros_LM_w_%.2f_um_n0_%.2f_n1_%.2f.dat" % (w0*1e6,n0,n1),"a+")
        fwells = open(folder+"welldepths_LM_w_%.2f_um_n0_%.2f_n1_%.2f.dat" % (w0*1e6,n0,n1),"a+")
        #only valid for variation of beam waist, not radius!
        #coeffs = PrecomputeAmplitudes(lmax,n0,n1,wavelength,radius)
    elif (method=="Ray"):
        f = open(folder+"zeros_Ray_w_%.2f_um_n0_%.2f_n1_%.2f.dat" % (w0*1e6,n0,n1),"a+")
        fwells = open(folder+"welldepths_Ray_w_%.2f_um_n0_%.2f_n1_%.2f.dat" % (w0*1e6,n0,n1),"a+")
    else:
        return False
    
    #compute axial forces for every value of parameter
    for radius in radiuslist:
        z0 = math.pi * (w0**2) * n0 / wavelength
        data = np.empty((numz,3))
        ipathologic = []

        for i in range(numz):
            z = zstart + i*dz
            if (math.fabs(z) < 1e-10):
                ipathologic.append(i)
                continue
            else:
                if (method=="Rayleigh"):
                    RayleighFgrad, RayleighFscat = RayleighLimitEval([0,0,z],I0,z0,w0,n0,n1,wavelength,radius,onedim=True)
                    Fcurrent = RayleighFgrad[2]+RayleighFscat[2]
                elif (method=="LM"):
                    Ftemp = LorenzMieEval(I0,z,w0,n0,n1,wavelength,radius,lmax)
                    Fcurrent = Ftemp[2]
                elif (method=="Ray"):
                    Ftemp = RayOpticsEval(I0,z,w0,n0,n1,wavelength,radius,nphi)
                    Fcurrent = Ftemp[2]
                data[i,0] = z
                data[i,1] = Fcurrent

        data = np.delete(data, ipathologic, axis=0)
        newlength = np.shape(data)[0]
        data[0,2] = 0.0
        for i in range (1,newlength):
            data[i,2] = integrate.simps(data[:i,1], data[:i,0])

        Fmax = np.max(data[:,1])
        Fmin = np.min(data[:,1])
        minlist = signal.argrelextrema(data[:,2], np.less, order=3)
        lowerbound_min = 1.1
        maxlist = signal.argrelextrema(data[:,2], np.greater, order=3)
        lowerbound_max = 1.1
        if any(map(len, maxlist)):
            for element in maxlist:
                if (data[element[0],2] < lowerbound_max):
                    lowerbound_max = data[element[0],2]
                    lowerbound_max_pos = data[element[0],0]

        f.write("%.5e %.5e %.8e %.8e" % (radius,w0, Fmax, Fmin))
        if any(map(len, minlist)):
            for element in minlist:
                f.write(" %.6e %.12e" % (data[element[0],0],data[element[0],2]))
                print("a = %e found %d local minima, %d maxima." % (radius, len(minlist), len(maxlist)))
                if (data[element[0],2] < lowerbound_min):
                    lowerbound_min = data[element[0],2]
                    lowerbound_min_pos = data[element[0],0]
        else:
            print("a = %e found 0 local minima." % (radius))

        if (lowerbound_max < 1 and lowerbound_min < lowerbound_max):
            welldepth = (lowerbound_max - lowerbound_min)/4.11433402e-21    #give well depth in units of kT
            fwells.write("%.5e %.5e %.6e %.6e %.6e %.6e %.6e\n" % (radius,w0, lowerbound_min_pos, lowerbound_max_pos, lowerbound_min, lowerbound_max, welldepth))
        else:
            fwells.write("%.5e %.5e 0.0 0.0 0.0 0.0 0.0\n" % (radius,w0))

        f.write("\n")

    f.close()
    fwells.close()
    return True

def simFT(dt,zlist,T):
    n = len(zlist)
    zFT = np.real(fft.rfft(zlist))
    zFT = np.square(zFT)
    freqs = fft.rfftfreq(n,dt)

    popt,_ = optimize.curve_fit(transferfunction,freqs,zFT,[100.0,zFT[0]],method="lm")
    trapcoeff = 2.0*constants.Boltzmann*T*popt[0]/(math.pi*popt[1])

    return freqs,zFT,popt[0],popt[1],trapcoeff

def SimAveraging(dt,n,zaveraged,T):
    freqs = fft.rfftfreq(n,dt)
    popt,_ = optimize.curve_fit(transferfunction,freqs,zaveraged,[50.0,zaveraged[0]],method="lm")
    trapcoeff = 2.0*constants.Boltzmann*T*popt[0]/(math.pi*popt[1])

    return freqs,popt[0],popt[1],trapcoeff

def transferfunction(omega,omega0,Sconst):
    return Sconst/(omega**2 + omega0**2)

def GaussianE(x,z0, w0, n0, n1, wavelength):
    #returns electric field strength (divided by E0) for Gaussian beam and the weakly confined beam approximation formula by Lock (2004)
    
    s_i = wavelength / (n0 * 2 * math.pi * w0)
    D = 1.0/np.complex(1,2*s_i*x/w0)
    E_approx = D*np.exp(np.complex(0,n0*2*math.pi*x/wavelength))
    eta = math.atan(x/z0)
    w2 = (w0**2) * (1 + (x/z0)**2)
    E_exact = (w0/math.sqrt(w2)) * np.exp(np.complex(0,2*math.pi*n0*x/wavelength)) * np.exp(np.complex(0,eta))
    
    #return np.real(E_exact), np.imag(E_exact) ,np.real(E_approx), np.imag(E_approx)
    return np.real(E_exact * np.conj(E_exact)), np.imag(E_exact) ,np.real(E_approx * np.conj(E_approx)), np.imag(E_approx)


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
w0 = 0.5e-6             #beam waist radius
z0 = math.pi * (w0**2) * n0 / wavelength
#P = 0.0                #test code validity without beam
P = 0.1                 #100 mW
I0 = 2*P/(math.pi * (w0**2))             

#simulation parameters
dt = 1e-4
tmax = 1.0
FourierNreps = 10
lmax = 100
nphi = 75
#random.seed(1)          #if we wish for Brownian motion to be repeatable

D = 6 * math.pi * eta * radius
Brownconst = math.sqrt(2 * constants.Boltzmann * T * dt / D)
momentum = n1 * constants.h / wavelength

#Simulation and their Fourier transforms
r0 = [0.0,0.0,0e-6] #particle starts out in beam waist
F = [0.0,0.0,0.0]

#Rayleigh
""" multipliers = open("Rezultati/Rayleigh_P1e-2_dt_%.2e.dat" % (dt),"w+")
multipliers.write(f"# radius {radius:.3e} eta {eta:.2e} T {T:d} n0 {n0:.2f} n1 {n1:.2f} rhomedium {rhomedium:.2f} rhoparticle {rhoparticle:.2f}\n")
multipliers.write(f"# lambda {wavelength:.4e} w0 {w0:.2e} z0 {z0:.3e} I0 {I0:.2e}\n\n")

t = 0.0
r = r0

while(t < tmax):
    multipliers.write(f"{t:.5f} {r[0]:.12f} {r[1]:.12f} {r[2]:.12f} {F[0]:.6e} {F[1]:.6e} {F[2]:.6e}\n")
    F, dr = UpdatePosition(r, I0, z0, w0, n0, n1, wavelength, radius, rhomedium, rhoparticle, D, Brownconst, t, dt,onedim=True)
    r = np.add(r,dr)
    t += dt
multipliers.close() """

#LM
""" multipliers = open("Rezultati/LM_500mm_P1e-1_dt_%.2e.dat" % (dt),"w+")
multipliers.write(f"# radius {radius:.3e} eta {eta:.2e} T {T:d} n0 {n0:.2f} n1 {n1:.2f} rhomedium {rhomedium:.2f} rhoparticle {rhoparticle:.2f}\n")
multipliers.write(f"# lambda {wavelength:.4e} w0 {w0:.2e} z0 {z0:.3e} I0 {I0:.2e}\n\n")

coeffs = PrecomputeAmplitudes(lmax,n0,n1,wavelength,radius)
t = 0.0
r = r0

while(t < tmax):
    multipliers.write(f"{t:.5f} {r[2]:.12f} {F[2]:.6e}\n")
    F, dr = UpdatePositionLM(r, I0, z0, w0, n0, n1, wavelength, radius, rhomedium, rhoparticle, D, Brownconst, t, dt, lmax, precomputed=coeffs)
    r = np.add(r,dr)
    t += dt
multipliers.close() """

#ray optics approx.
""" multipliers = open("Rezultati/Rayoptics_a5um_P1e-1_dt_%.2e.dat" % (dt),"w+")
multipliers.write(f"# radius {radius:.3e} eta {eta:.2e} T {T:d} n0 {n0:.2f} n1 {n1:.2f} rhomedium {rhomedium:.2f} rhoparticle {rhoparticle:.2f}\n")
multipliers.write(f"# lambda {wavelength:.4e} w0 {w0:.2e} z0 {z0:.3e} I0 {I0:.2e}\n\n")

t = 0.0
r = r0

while(t < tmax):
    multipliers.write(f"{t:.5f} {r[2]:.12f} {F[2]:.6e}\n")
    F, dr = UpdatePositionRayOptics(r, I0, z0, w0, n0, n1, wavelength, radius, rhomedium, rhoparticle, D, Brownconst, t, dt, nphi)
    r = np.add(r,dr)
    t += dt
    #print("%.4f" % (t))

multipliers.close() """

#FT for all approaches
""" t = 0.0
r = r0
zpositions = []
coeffs = PrecomputeAmplitudes(lmax,n0,n1,wavelength,radius) #LM only
#1st run is separate
while(t < tmax):
    #F, dr = UpdatePosition(r, I0, z0, w0, n0, n1, wavelength, radius, rhomedium, rhoparticle, D, Brownconst, t, dt,onedim=True)
    F, dr = UpdatePositionLM(r, I0, z0, w0, n0, n1, wavelength, radius, rhomedium, rhoparticle, D, Brownconst, t, dt, lmax, precomputed=coeffs)
    #F, dr = UpdatePositionRayOptics(r, I0, z0, w0, n0, n1, wavelength, radius, rhomedium, rhoparticle, D, Brownconst, t, dt, nphi)
    r = np.add(r,dr)
    if (t > 0.2*tmax):  #"burn-in" to make sure particle starts out near equilibrium position
        zpositions.append(r[2])
    t += dt

n = len(zpositions)
print("n = %d" % (n))
zFT = np.real(fft.rfft(zpositions))
zFT = np.square(zFT)

#remaining runs
for i in range(1,FourierNreps):
    print("Simulating repetition %d" % (i))
    t = 0.0
    r = r0
    zrep = []
    while(t < tmax):
        #F, dr = UpdatePosition(r, I0, z0, w0, n0, n1, wavelength, radius, rhomedium, rhoparticle, D, Brownconst, t, dt,onedim=True)
        F, dr = UpdatePositionLM(r, I0, z0, w0, n0, n1, wavelength, radius, rhomedium, rhoparticle, D, Brownconst, t, dt, lmax, precomputed=coeffs)
        #F, dr = UpdatePositionRayOptics(r, I0, z0, w0, n0, n1, wavelength, radius, rhomedium, rhoparticle, D, Brownconst, t, dt, nphi)
        r = np.add(r,dr)
        if (t > 0.2*tmax):
            zrep.append(r[2])
        t += dt
    FTpart = np.real(fft.rfft(zpositions))
    FTpart = np.square(FTpart)
    zFT = np.add(zFT,FTpart) """

""" zFT = np.true_divide(zFT,FourierNreps)
freqs,omega0,Sconst,trapcoeff = SimAveraging(dt,n,zFT,T)
print("omega_0 = %f, k_z = %e" % (omega0,trapcoeff))

fourierresults = open("Rezultati/rayoptics_FT.dat","a+")
fourierresults.write("%e %f %d %f %e\n" % (dt, tmax, FourierNreps, math.fabs(omega0), math.fabs(trapcoeff)))
fourierresults.close()

fourier = open("Rezultati/rayoptics_dt_%.2e_1dim_FT.dat" % (dt),"w+")
for i in range(len(freqs)):
    fourier.write("%f %.10e %.10e\n" % (freqs[i],zFT[i],transferfunction(freqs[i],omega0,Sconst)))
fourier.close() """

""" zFT = np.true_divide(zFT,FourierNreps)
freqs = fft.rfftfreq(n,dt)
fourier = open("Rezultati/LM_P1e0_dt_%.2e_radius_%.2e_w0_%.2e_1dim_FT.dat" % (dt,radius,w0),"w+")
for i in range(len(freqs)):
    fourier.write("%f %.10e\n" % (freqs[i],zFT[i]))
fourier.close() """

#fit transforms post facto (doesn' work for some reason?)
""" data = np.loadtxt("Rezultati/rayoptics_P1e-6_dt_1.00e-04_1dim_FT_single.dat")
zFT = data[0:8000,1]
freqs = data[0:8000,0]
print(np.shape(zFT),np.shape(freqs))
n = len(zFT)
popt,_ = optimize.curve_fit(transferfunction,freqs,zFT,[20.0,1.2e-6],method="lm")
trapcoeff = 2.0*constants.Boltzmann*T*popt[0]/(math.pi*popt[1])
print("omega_0 = %f, k_z = %e" % (popt[0],trapcoeff))

fourier = open("Rezultati/rayoptics_dt_%.2e_1dim_FT.dat" % (dt),"w+")
for i in range(len(freqs)):
    fourier.write("%f %.10e %.10e\n" % (freqs[i],zFT[i],transferfunction(freqs[i],popt[0],popt[1])))
fourier.close() """

""" popt = [7.75918,1.47463e-08]
popterror = [0.04648,0.07376] #relative error
trapcoeff = 2.0*constants.Boltzmann*T*popt[0]/(math.pi*popt[1])
uncertainty = trapcoeff * (popterror[0]+popterror[1])
print("%.6e %.6e" % (trapcoeff,uncertainty)) """

#Here we generate force profiles along the z-axis in a sample trap for all 3 approaches.
""" zstart = -200e-6
zend = 200e-6
numz = 15001

forces = zForceSlice(zstart,zend,numz,I0,z0,w0,n0,n1,wavelength,radius,lmax,nphi)
fileslice = open("Rezultati/test_zslice_comparison_a_%.2f_um_waist_%.2f_um.dat" % (radius*1e6,w0*1e6),"w+")
fileslice.write(f"# radius {radius:.3e} I0 {I0:.5f} n0 {n0:.2f} n1 {n1:.2f} LM: lmax {lmax:d} Ray: nphi {nphi:d}\n")
np.savetxt(fileslice, forces)
fileslice.close() """

#checks locations of equilibria for a range of waist diameters
wlist = []
for w in range(50,60):
    wlist.append(w*0.1*1e-6)
zstart = -300e-6
zend = 300e-6
numz = 30001
zParSlice(zstart,zend,numz,T,I0,z0,wlist,n0,n1,wavelength,radius,lmax,nphi,"Rezultati/","Ray")

#...and particle sizes
""" alist = []
for a in range(1,2):
    alist.append(a*0.1*1e-6)
zstart = -50e-6
zend = 50e-6
numz = 30001
zRadiusSlice(zstart,zend,numz,T,I0,z0,w0,n0,n1,wavelength,alist,lmax,nphi,"Rezultati/","LM") """

###################################
#testing functions below this point
###################################

#Make sure fields of both analytic Gaussian beam and "approximation" agree 
""" fileEfield = open("Rezultati/Gaussianbeamfieldcomp.dat","w+")
for i in range(-5000,5001):
    Eexactreal,Eexactimag,Eapproxreal,Eapproximag = GaussianE(i*1e-8,z0, w0, n0, n1, wavelength)
    fileEfield.write("%.12f %.10e %.10e %.10e %.10e\n" % (i*1e-8,Eexactreal,Eexactimag,Eapproxreal,Eapproximag))
fileEfield.close() """

#Force components of single incoming ray
""" filesingleray = open("Rezultati/single_ray_n1_%.2f_n0_%.2f.dat" % (n1,n0),"w+")
for i in range(0,361):
    Fz,Fy = SingleRayForce(math.radians(i/4),n0,n1,1.0)
    filesingleray.write("%f %.10e %.10e\n" % (i/2,Fz,Fy))
filesingleray.close() """

#Check of field gradient for Gaussian beam
""" zstart = -10e-6
zend = 10e-6
numz = 2000
r = 1e-6
dz = (zend-zstart)/numz
filegradtest = open("Rezultati/beam_intensity_gradient_w0_%.3f_radial_%.3f.dat" % (w0*1e6,r*1e6),"w+")
filegradtest.write(f"# radius {radius:.4e} w0 {w0:.2e}\n\n")
for i in range(numz):
    I,grad = IntensityGradient([r,0,zstart+i*dz], I0, z0, w0, onedim=False)
    filegradtest.write("%e %e %e %e %e\n" % (zstart+i*dz, I, grad[0],grad[1],grad[2]))
filegradtest.close() """

#Check of annular power distribution
""" z = 20e-6

divangle = wavelength / (math.pi * w0)
if (math.fabs(z) / radius <= 1.0):
    phimax = min(4*divangle,math.pi/2)
else:
    phimax = min(4*divangle, math.asin(radius / math.fabs(z)))
philist = []
for i in range(0,nphi):
    philist.append((i+0.5) * phimax/nphi)
P = 0.0
powerdistrotest = open("Rezultati/Annulus_dP_w0_%.3f_z_%.3f.dat" % (w0*1e6,z*1e6),"w+")
for i in range(nphi):
    u, theta, vartheta, zannulus = AnnulusCoords(z, radius, w0, philist[i])
    dP = AnnulusPower(philist[i] - 0.5 * phimax/nphi , philist[i] + 0.5 * phimax/nphi, I0, z0, w0, zannulus)
    P += dP
    powerdistrotest.write("%.10f %e %e %e %e\n" % (philist[i], dP, P,z / (math.pi/2 - math.atan(philist[i] - 0.5 * phimax/nphi)),z / (math.pi/2 - math.atan(philist[i] + 0.5 * phimax/nphi))))
powerdistrotest.close() """

#Check of annulus coordinate function
""" zstart = -7e-6
zend = 7e-6
numz = 1001
dz = (zend-zstart)/numz
divangle = wavelength / (math.pi * w0)
print(divangle)
fileannulustest = open("Rezultati/raytest_largeparticle.dat","w+")
fileannulustest.write(f"# radius {radius:.4e} w0 {w0:.2e}\n\n")
for i in range(numz):
    u,theta,vartheta,z = AnnulusCoords(zstart+i*dz, radius, w0, divangle)
    fileannulustest.write("%e %e %e %e %e\n" % (zstart+i*dz,u,theta,vartheta,z))
fileannulustest.close() """

#Check of angular force distribution
""" nphi = 100
zoffset = -radius*2.0
divangle = wavelength / (math.pi * w0)
if (math.fabs(zoffset) / radius <= 1.0):
    phimax = 4*divangle
else:
    phimax = min(4*divangle, math.asin(radius / math.fabs(zoffset)))
#write down central angles of each ray annulus (relative to waist)
philist = []
for i in range(nphi):
    philist.append((i+0.5) * phimax/nphi)

fileannulusdisttest = open("Rezultati/annulus_power_distriution_w0_%.3f_radius_%.3f_offset_%.3f_n1_%2f_nphi_%d.dat" % (w0*1e6,radius*1e6,zoffset*1e6,n1,nphi),"w+")
Fopt = 0.0
for t in range(nphi):
    u, theta, vartheta, zannulus = AnnulusCoords(zoffset, radius, w0, philist[t])
    dP = AnnulusPower(philist[t] - 0.5 * phimax/nphi , philist[t] + 0.5 * phimax/nphi, I0, z0, w0, zannulus)
    Fz, Fy = SingleRayForce(theta, n0, n1, dP)
    if (zoffset < 0):
        dFopt = Fz * math.cos(philist[t]) - Fy * math.sin(philist[t])
        Fopt += dFopt
    elif (zoffset >= 0):
        dFopt = Fz * math.cos(philist[t]) + Fy * math.sin(philist[t])
        Fopt += dFopt
    fileannulusdisttest.write("%f %e %e %e %e\n" % (philist[t],Fz,Fy,dFopt,Fopt))
fileannulusdisttest.close() """

#Check of LM coefficients
""" lmax = 100
zoffset = -0.1e-6
fileLMtest = open("Rezultati/LM_terms_distriution_w0_%.3f_radius_%.3f_offset_%.3f_n1_%2f.dat" % (w0*1e6,radius*1e6,zoffset*1e6,n1),"w+")
coeffsum = 0.0
for l in range(1,lmax+1):
    coeff = LorenzMieTerm(l,zoffset,w0, n0, n1, wavelength, radius)
    coeffsum += coeff
    fileLMtest.write("%d %e %e %e\n" % (l,np.real(coeff),np.imag(coeff),np.real(coeffsum)))
fileLMtest.close() """