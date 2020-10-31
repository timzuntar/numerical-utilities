import pygmsh
import meshio
import numpy
import math
from scipy.linalg import solve

def triangle_area(cell,points):
    #calculates surface area of chosen triangle (a single cell)
    span = (points[cell[1]][0]-points[cell[0]][0])*(points[cell[2]][1]-points[cell[0]][1])-(points[cell[1]][1]-points[cell[0]][1])*(points[cell[2]][0]-points[cell[0]][0])
    return 0.5*abs(span)

def wf_inner_product(cell,points,m):
    #inner product of basis and weight functions (local stress component)
    vertices = [points[cell[0]],points[cell[1]],points[cell[2]]]    #3 vertices
    determinant = (vertices[(m+1)%3][0]-vertices[m][0])*(vertices[(m+2)%3][1]-vertices[m][1])-(vertices[(m+2)%3][0]-vertices[m][0])*(vertices[(m+1)%3][1]-vertices[m][1])
    return determinant/6.0

def A_mn(cell,points,m,n):
    #local contributions of the 3 vertices to stiffness matrix
    area = triangle_area(cell,points)
    vertices = [points[cell[0]],points[cell[1]],points[cell[2]]]
    product = (vertices[(m+1)%3][1]-vertices[(m+2)%3][1])*(vertices[(n+1)%3][1]-vertices[(n+2)%3][1])+(vertices[(m+2)%3][0]-vertices[(m+1)%3][0])*(vertices[(n+2)%3][0]-vertices[(n+1)%3][0])
    return product/(4.0*area)

def enforce_dirichlet(cells,points,lines,num_vertices,S,g):
    #forces Dirichlet boundary condition (velocity is 0 at all edges)
    for i in range(0,num_vertices):
        if (any(i in sublist for sublist in lines) == True):
            for j in range(0,num_vertices):
                S[i][j] = 0.0
                S[j][i] = 0.0
            S[i][i] = 1.0
            g[i] = 0.0
    return S,g
    
def fill_and_calculate(cells,points,num_triangles,num_vertices,S,g):

    for triangle in range(0,num_triangles):
        for m in range(0,3):
            g[cells[triangle][m]] += wf_inner_product(cells[triangle],points,m)
            for n in range(0,3):
                S[cells[triangle][m]][cells[triangle][n]] += A_mn(cells[triangle],points,m,n)

    Sout,gout = enforce_dirichlet(cells,points,lines,num_vertices,S,g)
    return Sout,gout

def calculate_poiseuille_coeff(cells,points,num_triangles,c):
    #returns the Poiseuille coefficient (from Hagen-Poiseuille law) and total flux
    phi = 0.0
    totalarea = 0.0
    for triangle in range(0,num_triangles):
        area = triangle_area(cells[triangle],points)
        totalarea += area
        average = 0.0
        for m in range(0,3):
            average += c[cells[triangle][m]]
        phi += area*average/3.0
    return 8.0*math.pi*phi/(totalarea**2), phi

def print_to_file(filename,points,c,num_vertices):
    #prints list of 2D point coordinates and corresponding solutions to a file
    out = open(filename,"wt")
    for index in range(0,num_vertices):
        out.write("%f %f %.10f\n" % (points[index][0],points[index][1],c[index]))

    out.close()

####################
####MAIN CODE#######
####################
meshsize = 0.8  #local distance between vertices (I kept the same value throughout the mesh)
"""
#generates a semicircular mesh
with pygmsh.geo.Geometry() as geom:
    p1 = geom.add_point([-1.0, 0.0], meshsize)
    p2 = geom.add_point([1.0, 0.0], meshsize)
    p3 = geom.add_point([0.0, 0.0], meshsize)
    arc = geom.add_circle_arc(p2,p3,p1)
    line1 = geom.add_line(p1,p3)
    line2 = geom.add_line(p3,p2)

    loop = geom.add_curve_loop([arc, line1,line2])
    interior = geom.add_plane_surface(loop)

    mesh = geom.generate_mesh()
"""

#generates a mesh in the shape of a crescent
with pygmsh.geo.Geometry() as geom:
    p1 = geom.add_point([0.0, 4.0], meshsize)
    p2 = geom.add_point([0.0, 1.0], meshsize)
    p3 = geom.add_point([1.0, 0.0], meshsize)
    p4 = geom.add_point([4.0, 0.0], meshsize)
    p5 = geom.add_point([4.0, 2.0], meshsize)
    p6 = geom.add_point([3.0, 1.0], meshsize)
    p7 = geom.add_point([2.0, 1.0], meshsize)
    p8 = geom.add_point([1.0, 2.0], meshsize)
    p9 = geom.add_point([1.0, 3.0], meshsize)
    p10 = geom.add_point([2.0, 4.0], meshsize)
    line1 = geom.add_line(p1,p2)
    line2 = geom.add_line(p2,p3)
    line3 = geom.add_line(p3,p4)
    line4 = geom.add_line(p4,p5)
    line5 = geom.add_line(p5,p6)
    line6 = geom.add_line(p6,p7)
    line7 = geom.add_line(p7,p8)
    line8 = geom.add_line(p8,p9)
    line9 = geom.add_line(p9,p10)
    line10 = geom.add_line(p10,p1)
    loop = geom.add_curve_loop([line1,line2,line3,line4,line5,line6,line7,line8,line9,line10])
    interior = geom.add_plane_surface(loop)

    mesh = geom.generate_mesh()

"""
#writes mesh to format suitable for external viewers (e.g. ParaView)
with open("crescent_pts.dat", "w+") as f:
    for item in mesh.points:
        f.write("%f %f %f\n" % (item[0],item[1],item[2]))
mesh.write("rezultati/crescent.vtk")
"""


lines = numpy.vstack(numpy.array([cells.data for cells in mesh.cells if cells.type == "line"])) #so we'll know which points are at the edges 
cells = numpy.vstack(numpy.array([cells.data for cells in mesh.cells if cells.type == "triangle"])) #writes triangle definitions to separate array
#mesh.points, cells and lines now contain all information on mesh
num_triangles = numpy.size(cells,0)
num_vertices = numpy.size(mesh.points,0)
print("Mesh contains %d vertices and %d triangular cells." % (num_vertices,num_triangles))
bandwidth = 0
for cell in cells:
    min = numpy.min(cell)
    max = numpy.max(cell)
    if (max-min >= bandwidth):
        bandwidth = max-min
print("Matrix bandwidth = %d\n" % (bandwidth*2-1))

#initialise arrays
g = numpy.zeros((num_vertices))
S = numpy.zeros((num_vertices,num_vertices))

#now we calculate everything
Sout,gout = fill_and_calculate(cells,mesh.points,num_triangles,num_vertices,S,g)
solution = solve(Sout,gout)
C,flux = calculate_poiseuille_coeff(cells,mesh.points,num_triangles,solution)

#saving the data
filename = "rezultati/crescent_08distance_profile.dat"
file_poiseuille = open("rezultati/crescent_coefficients.dat","a")
file_poiseuille.write("%.3f %d %d %.10f %.10f %d\n" % (meshsize,num_vertices,num_triangles,C,flux,bandwidth*2-1))
print_to_file(filename,mesh.points,solution,num_vertices)
file_poiseuille.close()