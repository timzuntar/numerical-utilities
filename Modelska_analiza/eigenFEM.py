#Code for finding oscillation eigenvalues of membrane with finite element method 

import pygmsh
import numpy
import math
import cmath
from scipy.linalg import solve,solve_triangular,cholesky,eig
from scipy.optimize import curve_fit

import FEMshared

def find_neighbors(POI,cells):
    #finds all next-neighboring vertices to the current one (POI) and cells that contain the vertex
    neighbor_list = []
    cell_list = []
    for cellindex,cell in enumerate(cells):
        if POI in cell:
            cell_list.append(cell)
            for vertex in cell:
                if (vertex != POI and vertex not in neighbor_list):
                    neighbor_list.append(vertex)
    return neighbor_list,cell_list

def find_connections(POI1,POI2,cells):
    #the only options are 0,1 and 2 connections, since a line obviously cannot be shared by more than 2 triangles
    cell_list = []
    for cell in cells:
        if (POI1 in cell and POI2 in cell):
            cell_list.append(cell)
    return cell_list

def distance_2D(POI1,POI2,points):
    return math.sqrt((points[POI1,0]-points[POI2,0])**2 + (points[POI1,1]-points[POI2,1])**2)

def AB_diag(A,B,POI,points,cells):
    #
    neighbors,neighbor_cells = find_neighbors(POI,cells)
    sumA = 0.0
    sumB = 0.0
    for cell in neighbor_cells:
        vertices = []
        area = FEMshared.triangle_area(cell,points)
        for vertex in cell:
            if (vertex != POI):
                vertices.append(vertex)
        sumA += (distance_2D(vertices[0],vertices[1],points))**2 /(4.0*area)
        sumB += area/6.0    #possible FIXME, is this always true or just when the vertex connects to 6 cells?
    A[POI][POI] = sumA
    B[POI][POI] = sumB
    return sumA,sumB

def AB_offdiag(A,B,POI1,POI2,points,cells):
    #terribly messy
    connections = find_connections(POI1,POI2,cells)
    sumA = 0.0
    sumB = 0.0

    if (len(connections) > 0):
        for connection in connections:
            for vertex in connection:
                if (vertex != POI1 and vertex != POI2):
                    POI3 = vertex
            area = FEMshared.triangle_area(connection,points)
            sumA += ((points[POI2][0]-points[POI3][0])*(points[POI3][0]-points[POI1][0]) + (points[POI2][1]-points[POI3][1])*(points[POI3][1]-points[POI1][1]))/(4.0*area)
            sumB += area/12.0   #possible FIXME, is this always true or just when the vertex connects to 6 cells?
    A[POI1][POI2] = sumA
    B[POI1][POI2] = sumB
    return sumA,sumB

def enforce_dirichlet_matrix(cells,points,lines,num_vertices,A,B):
    #forces Dirichlet boundary condition (velocity is 0 at all edges).
    #We could fix the diagonal element to a finite value in either A or B, but doing it in B preserves its positive-definiteness
    for i in range(0,num_vertices):
        if (any(i in sublist for sublist in lines) == True):
            for j in range(0,num_vertices):
                A[i][j] = 0.0
                A[j][i] = 0.0
                B[i][j] = 0.0
                B[j][i] = 0.0
            B[i][i] = 1.0
    return A,B

def calculate_eigenvalues_eigenfunctions(meshsize,WriteVTK=False,):
    #mesh = FEMshared.semicircular_mesh(meshsize,1.0)
    mesh = FEMshared.Q_mesh(meshsize)
    if (WriteVTK == True):
        mesh.write("rezultati/Q.vtk")    #optionally export entire mesh
    vertex_file_name = "semicircle_pts_" + "%.3f" % meshsize + ".dat"
    FEMshared.vertices_export(vertex_file_name,mesh)

    lines = numpy.vstack(numpy.array([cells.data for cells in mesh.cells if cells.type == "line"])) #so we'll know which points are at the edges 
    cells = numpy.vstack(numpy.array([cells.data for cells in mesh.cells if cells.type == "triangle"])) #writes triangle definitions to separate array
    num_triangles = numpy.size(cells,0)
    num_vertices = numpy.size(mesh.points,0)
    print("Mesh contains %d vertices, %d triangular cells and a border of length %d." % (num_vertices,num_triangles,len(lines)))
    A = numpy.zeros((num_vertices,num_vertices))
    B = numpy.zeros((num_vertices,num_vertices))
    print("Arrays initialized...")
    for i in range(0,num_vertices):
        for j in range(0,num_vertices):
            if (i == j):
                AB_diag(A,B,i,mesh.points,cells)
            else:
                AB_offdiag(A,B,i,j,mesh.points,cells)
    A,B = enforce_dirichlet_matrix(cells,mesh.points,lines,num_vertices,A,B)
    print("Arrays filled...")
    L = cholesky(B,lower=True)
    print("Cholesky decomposition finished...")
    YT = solve_triangular(L, A.transpose(), lower=True, unit_diagonal=False, overwrite_b=False)
    Y = YT.transpose()
    C = solve_triangular(L, Y, lower=True, unit_diagonal=False, overwrite_b=False)
    lambdas,vl = eig(C, b=None, left=False, right=True)
    a_solution = solve_triangular(L.transpose(), vl, lower=False, unit_diagonal=False, overwrite_b=False)
    print("Eigensystem solved...")
    idx = lambdas.argsort()[::1]   
    lambdas = lambdas[idx]
    a_solution = a_solution[:,idx]
    lambdas_filename = "rezultati/Q_" + "%.3f" % meshsize + "lambda.dat"
    lambdas_out = open(lambdas_filename,"w+")
    for l in range(len(lambdas)-len(lines)):
        lambdas_out.write("%d %.10f %.10f\n" % (l,lambdas[l+len(lines)].real,math.sqrt(abs(lambdas[l+len(lines)].real))))

    #also write down first 15 eigenfunctions
    for i in range(15):
        filename = "rezultati/Q_N_%d_T_%d_eigen_%d.dat" % (num_vertices,num_triangles,i)
        FEMshared.print_to_file(filename,mesh.points,a_solution[:,i+len(lines)],num_vertices)

    return num_vertices,num_triangles,lambdas[len(lines):]

def extrapolation(x,Cinf,alpha,constant):
    #extrapolates eigenvalues (x) for theoretical infinitely dense mesh
    xout = x
    for val in range(len(x)):
        xout[val] = Cinf*(x[val]**(-alpha)) + constant
    return xout

def galerkin_phi_mn(m,n,r,phi):
    return (1.0-r)*(r**(m+n))*math.sin(m*phi)

def galerkin_inner(m,k,n,l):
    if (m == n):
        pt1 = 1.0/(2.0*m+k+l+2.0)
        pt2 = 2.0/(2.0*m+k+l+3.0)
        pt3 = 1.0/(2.0*m+k+l+4.0)
        return 0.5*math.pi*(pt1-pt2+pt3)
    else:
        return 0

def galerkin_nabla_inner(m,k,n,l):
    if (m == n):
        pt1 = k*l/(2.0*m+k+l)
        pt2 = (2.0*k*l+k+l)/(2.0*m+k+l+1.0)
        pt3 = (k+1.0)*(l+1.0)/(2.0*m+k+l+2.0)
        return 0.5*math.pi*(pt1-pt2+pt3)
    else:
        return 0

def galerkin_system(Nr,Nphi):
    A = numpy.zeros((Nr*Nphi,Nr*Nphi))
    B = numpy.zeros((Nr*Nphi,Nr*Nphi))
    for i in range(Nr*Nphi):
        for j in range(Nr*Nphi):
            A[i][j] = galerkin_nabla_inner((i%Nphi)+1,i//Nphi,(j%Nphi)+1,j//Nphi)
            B[i][j] = galerkin_inner((i%Nphi)+1,i//Nphi,(j%Nphi)+1,j//Nphi)

    lambdas,vr = eig(A,B,right=True)
    return lambdas,vr

def galerkin_pointeval_semicircle(Nr,Nphi,eigenvector,radial,angular):
    pointlist = numpy.empty((radial*angular,3))
    for i in range(radial):
        for j in range(angular):
            u = 0.0
            for r in range(Nr):
                for phi in range(Nphi):
                    u += eigenvector[r*Nphi+phi]*galerkin_phi_mn(phi+1,r,i/(radial-1.0),math.pi*j/(angular-1.0))
            pointlist[i*angular+j][0] = i/(radial-1.0)
            pointlist[i*angular+j][1] = math.pi*j/(angular-1.0)
            pointlist[i*angular+j][2] = u
    return pointlist
    
def points_to_file(filename,pointlist):
    for row in range(pointlist.shape[0]):
        filename.write("%f %f %.10f\n" % (pointlist[row][0],pointlist[row][1],pointlist[row][2]))
    return

####################
####MAIN CODE#######
####################
"""
#computation of displacement eigenvalues and -vectors on a (thin) membrane for several mesh densities
#meshsize = 0.3
size_list = [0.04]  #define the local mesh spacings for which eigenvalues are to be calculated
comp_out = open("rezultati/Q_lambdaNdep.dat","a+")
for meshsize in size_list:
    N,T,lambdas = calculate_eigenvalues_eigenfunctions(meshsize)
    comp_out.write("%d %d" % (N,T))
    for i in range(0,15):
        comp_out.write(" %.9f" % (lambdas[i]))
    comp_out.write("\n")
comp_out.close()
"""
"""
data = numpy.loadtxt("rezultati/semicircle_lambdaNdep.dat")
fits_out = open("rezultati/semicircle_extrapolated.dat","w+")
xTdata = data[:,1]
lambdadata = data[:,2:]
parguess = [68.0,1.05,14.7]
for column in range(numpy.size(lambdadata,1)):
    ydata = lambdadata[:,column]
    #print(xTdata,ydata)
    popt,pcov = curve_fit(extrapolation,xTdata,ydata,p0=parguess,method='lm')
    print(popt)
    fits_out.write("%d %.10f %.10f %.10f %.10f %.10f %.10f\n" % (column,popt[0],popt[1],popt[2],pcov[0][0],pcov[1][1],pcov[2][2]))

fits_out.close()
"""

#we now solve the same problem with Galerkin-like test functions
Nr = 3
Nphi = 3

lambdas,vr = galerkin_system(Nr,Nphi)
idx = lambdas.argsort()[::1]   
lambdas = lambdas[idx]
vr = vr[:,idx]
lambdas_filename = "rezultati/galerkin_" + "Nr_%d_Nphi_%d_" % (Nr,Nphi) + "lambda.dat"
lambdas_out = open(lambdas_filename,"w+")
for l in range(len(lambdas)):
    lambdas_out.write("%d %.10f %.10f\n" % (l,lambdas[l].real,math.sqrt(abs(lambdas[l].real))))
for vectorindex in range(6):
    pointlist = galerkin_pointeval_semicircle(Nr,Nphi,vr[:,vectorindex],40,40)
    filename = "rezultati/galerkin_Nr_%d_Nphi_%d_eigen_%d.dat" % (Nr,Nphi,vectorindex)
    vector_out = open(filename,"w+")
    points_to_file(vector_out,pointlist)
    vector_out.close()
