#Two relatively simple examples of the boundary element method - one computes the electrostatic field around a charged plate,
#the other the velocity field around several types of airfoils 

import math
import cmath
import pygmsh
import FEMshared
import numpy as np
from scipy import linalg

def segment_length(point1, point2):
    return math.sqrt((point2[0]-point1[0])**2+(point2[1]-point1[1])**2)

def perp_unit_vector(point1,point2,length):
    nx = (point2[1]-point1[1])/length
    ny = (point1[0]-point2[0])/length
    return [nx,ny]

def check_F_sign(point1,point2,length,xieta,eps):
    nvector = perp_unit_vector(point1,point2,length)
    A = length**2
    B = 2.0*length*(-nvector[1]*(point1[0]-xieta[0]) + nvector[0]*(point1[1]-xieta[1]))
    C = (point1[0]-xieta[0])**2 + (point1[1]-xieta[1])**2
    F = 4*A*C-B**2
    if (F > eps):
        return True,A,B,C,F
    elif (F>0 and F < eps):
        return False,A,B,C,F
    elif (F<0):
        raise ValueError("Calculated negative value of F")

def calculate_integrals(point1,point2,xieta):
    eps = 10e-9
    length = segment_length(point1,point2)
    n = perp_unit_vector(point1,point2,length)
    positive,A,B,C,F = check_F_sign(point1,point2,length,xieta,eps)
    if (positive == False):
        Vn = (0.5*length/math.pi)*(math.log(length)+(1.0+0.5*B/A)*math.log(math.abs(1.0+0.5*B/A))-(0.5*B/A)*math.log(0.5*B/A)-1.0)
        Dn = 0.0
    if (positive == True):
        temp = math.atan2((2*A+B),math.sqrt(F))-math.atan2(B,math.sqrt(F))
        Vn = (0.25*length/math.pi)*(2.0*(math.log(length)-1.0)+(1.0+0.5*B/A)*math.log(math.abs(1.0+B/A+C/A))-(0.5*B/A)*math.log(C/A)+(math.sqrt(F)/A)*temp)
        Dn = length*temp*(n[0]*(point1[0]-xieta[0])+n[1]*(point1[1]-xieta[1]))/(math.pi*math.sqrt(F))
    return Vn,Dn

def twodim_tape(meshsize_middle,meshsize_ends):
    #all these cases could be handled by the same function but I'm lazy.
    with pygmsh.geo.Geometry() as geom:
        end1 = geom.add_point([-0.5, 0.0], meshsize_ends)
        middle = geom.add_point([0.0, 0.0], meshsize_middle)
        end2 = geom.add_point([0.5, 0.0], meshsize_ends)
        line1 = geom.add_line(end1,middle)
        line2 = geom.add_line(middle,end2)
        mesh = geom.generate_mesh()
    return mesh

def twodim_corner(meshsize_middle,meshsize_ends):
    with pygmsh.geo.Geometry() as geom:
        end1 = geom.add_point([-0.5, 0.0], meshsize_ends)
        middle1 = geom.add_point([-0.25, 0.0], meshsize_middle)
        middle = geom.add_point([0.0, 0.0], meshsize_ends)
        middle2 = geom.add_point([0.0, -0.25], meshsize_middle)
        end2 = geom.add_point([0.0, -0.5], meshsize_ends)
        line1 = geom.add_line(end1,middle1)
        line2 = geom.add_line(middle1,middle)
        line3 = geom.add_line(middle,middle2)
        line4 = geom.add_line(middle2,end2)
        mesh = geom.generate_mesh()
    return mesh

def twodim_chevron(meshsize_middle,meshsize_ends,angle):

    with pygmsh.geo.Geometry() as geom:
        end1 = geom.add_point([-0.5*math.sin(angle*math.pi/180), -0.5*math.cos(angle*math.pi/180)], meshsize_ends)
        middle1 = geom.add_point([-0.25*math.sin(angle*math.pi/180), -0.25*math.cos(angle*math.pi/180)], meshsize_middle)
        middle = geom.add_point([0.0, 0.0], meshsize_ends)
        middle2 = geom.add_point([0.25*math.sin(angle*math.pi/180), -0.25*math.cos(angle*math.pi/180)], meshsize_middle)
        end2 = geom.add_point([0.5*math.sin(angle*math.pi/180), -0.5*math.cos(angle*math.pi/180)], meshsize_ends)
        line1 = geom.add_line(end1,middle1)
        line2 = geom.add_line(middle1,middle)
        line3 = geom.add_line(middle,middle2)
        line4 = geom.add_line(middle2,end2)
        mesh = geom.generate_mesh()
    return mesh

def analytic_panel_solution(point_relative,panel_length):
    xplus = point_relative[0]-panel_length/2.0
    xminus = point_relative[0]+panel_length/2.0

    angle1 = math.atan2(point_relative[1],xminus)
    angle2 = math.atan2(point_relative[1],xplus)
    #print(angle1,angle2)

    sum1 =  point_relative[1]*(angle2-angle1)
    sum2 = 0.5*xminus*math.log(xminus**2+(point_relative[1])**2)
    sum3 = 0.5*xplus*math.log(xplus**2+(point_relative[1])**2)
    #print(sum1,sum2,sum3)
    return (0.5/math.pi)*(sum1+sum2-sum3-panel_length)

def shift_and_rotate(point1,point2,xieta,evaluationpoint):
    #returns the position of evaluated point in the local coordinate system of segment spanning between points 1 and 2
    #inclusion of "xieta" isn't technically needed, but it helps cut down on the amount of duplicate calculations
    delta_x = point2[0] - point1[0]
    delta_y = point2[1] - point1[1]
    phi = math.atan2(delta_y, delta_x)
    x_rel = math.cos(phi)*(evaluationpoint[0]-xieta[0]) - math.sin(phi)*(evaluationpoint[1]-xieta[1])
    y_rel = math.sin(phi)*(evaluationpoint[0]-xieta[0]) - math.cos(phi)*(evaluationpoint[1]-xieta[1])
    return [x_rel,y_rel]

def potential_contributions(points,lines,centerpoints):
    N = np.shape(lines)[0]
    G = np.zeros((N,N))
    for j in range(N):
        for i in range(N):
            point1 = [points[lines[i][0]][0],points[lines[i][0]][1]]    #endpoints of the i-th panel
            point2 = [points[lines[i][1]][0],points[lines[i][1]][1]]
            evaluationpoint = centerpoints[j,:]  #centerpoint of j-th panel
            relative = shift_and_rotate(point1,point2,evaluationpoint,centerpoints[i,:])   #calculate coordinates of j-th panel in system of i-th panel
            G[j][i] = analytic_panel_solution(relative,segment_length(point1,point2))
    return G

def evaluate_field(xmin,xmax,ymin,ymax,xpts,ypts,points,lines,centerpoints,sigmas):
    dx = (xmax-xmin)/float(xpts)
    dy = (ymax-ymin)/float(ypts)
    u = np.zeros((xpts,ypts))
    for xpt in range(xpts):
        for ypt in range(ypts):
            evaluationpoint = [xmin+xpt*dx,ymin+ypt*dy]
            for i in range(len(sigmas)):
                point1 = [points[lines[i][0]][0],points[lines[i][0]][1]]
                point2 = [points[lines[i][1]][0],points[lines[i][1]][1]]
                relative = shift_and_rotate(point1,point2,centerpoints[i,:],evaluationpoint)
                G = analytic_panel_solution(relative,segment_length(point1,point2))
                u[xpt][ypt] += sigmas[i]*G
    return u

def evaluate_capacitance(points,lines,sigmas):
    C = 0.0
    for i in range(len(sigmas)):
        point1 = [points[lines[i][0]][0],points[lines[i][0]][1]]    #endpoints of the i-th panel
        point2 = [points[lines[i][1]][0],points[lines[i][1]][1]]        

        l = segment_length(point1,point2)
        C += l*sigmas[i]

    return 0.5*C

def elliptical_profile(numpanels,b,phi):
    points_init = []
    points = np.empty((numpanels,2))
    lines = []
    theta = 0.0
    dtheta = 2*math.pi/numpanels
    for i in range(numpanels):
        points_init.append([math.cos(theta),b*math.sin(theta)])
        theta += dtheta
    for i,point in enumerate(points_init):
        points[i][0] = math.cos(phi)*point[0]-math.sin(phi)*point[1]
        points[i][1] = math.sin(phi)*point[0]+math.cos(phi)*point[1]

    for i in range(np.shape(points)[0]-1):
        lines.append([i,i+1])
    lines.append([np.shape(points)[0]-1,0])

    centerpoints = []
    for line in lines:
        centerpoints.append([0.5*(points[line[0]][0]+points[line[1]][0]),0.5*(points[line[0]][1]+points[line[1]][1])])
    return points,lines,centerpoints

def elliptical_analytic_sol(b,point,v_inf):
    #returns analytic solution for tangential velocity
    #only valid for a non-rotated ellipse with axis scaled with b parallel to y!
    return v_inf*(1+b)*point[1]/math.sqrt(point[1]**2+(b**4)*(point[0]**2))

def NACA_evaluate(x,t):
    main = 1.457122*math.sqrt(x)-0.624424*x-1.727016*(x**2)+1.384087*(x**3)-0.489769*(x**4)
    return t*main/50.0

def Zhukhovsky_evaluate(A,B,C,R,phi):
    temp = (C**2)/((A+R*math.cos(phi))**2+(B+R*math.sin(phi))**2)
    xi_x = 0.5*(A+R*math.cos(phi))*(1+temp)
    xi_y = 0.5*(A+R*math.sin(phi))*(1-temp)
    return [xi_x,xi_y]

def NACA_profile(numpanels_single_side,t):
    points = []
    lines = []
    x = 0.0
    dx = 1.0/numpanels_single_side
    for i in range(numpanels_single_side):
        points.append([x,NACA_evaluate(x,t)])
        x+=dx
    x = 1.0
    for i in range(numpanels_single_side):
        points.append([x,-NACA_evaluate(x,t)])
        x-=dx

    for i in range(np.shape(points)[0]-1):
        lines.append([i,i+1])
    lines.append([np.shape(points)[0]-1,0])

    centerpoints = []
    for line in lines:
        centerpoints.append([0.5*(points[line[0]][0]+points[line[1]][0]),0.5*(points[line[0]][1]+points[line[1]][1])])

    #print(centerpoints)
    return points,lines,centerpoints

def Zhukhovsky_profile(numpanels,A,B):
    R = 0.5
    C = 0.55
    points = []
    lines = []
    phi = 0.0
    dphi = 2.0*math.pi/numpanels
    for i in range(numpanels):
        points.append(Zhukhovsky_evaluate(A,B,C,R,phi))
        phi+=dphi
    for i in range(np.shape(points)[0]-1):
        lines.append([i,i+1])
    lines.append([np.shape(points)[0]-1,0])
    centerpoints = []
    for line in lines:
        centerpoints.append([0.5*(points[line[0]][0]+points[line[1]][0]),0.5*(points[line[0]][1]+points[line[1]][1])])
    return points,lines,centerpoints

def velocity_components(point_relative,panel_length):
    xplus = point_relative[0]-panel_length/2.0
    xminus = point_relative[0]+panel_length/2.0
    vel_parallel = (0.25/math.pi)*math.log((xplus**2+point_relative[1]**2)/(xminus**2+point_relative[1]**2))
    vel_perpendicular = (0.5/math.pi)*(math.atan2(point_relative[1],xplus)-math.atan2(point_relative[1],xminus))
    return [vel_parallel,vel_perpendicular]

def relative_position(point1,point2,centerpoint1,centerpoint2):
    #centerpoint1 is the center of panel spanning from point1 to point2,
    #centerpoint2 are the coordinates for which we wish to compute the panel's contribution. 
    delta_x = point2[0] - point1[0]
    delta_y = point2[1] - point1[1]
    phi = math.atan2(delta_y, delta_x)
    v_parallel = math.cos(phi)*(centerpoint2[0]-centerpoint1[0]) + math.sin(phi)*(centerpoint2[1]-centerpoint1[1])
    v_perp = -math.sin(phi)*(centerpoint2[0]-centerpoint1[0]) + math.cos(phi)*(centerpoint2[1]-centerpoint1[1])
    return [v_parallel,v_perp]

def vel_lab(point1,point2,vel_local):
    delta_x = point2[0] - point1[0]
    delta_y = point2[1] - point1[1]
    phi = math.atan2(delta_y, delta_x)
    vx = math.cos(phi)*vel_local[0]-math.sin(phi)*vel_local[1]
    vy = math.sin(phi)*vel_local[0]+math.cos(phi)*vel_local[1]
    return [vx,vy]

def vel_inverse(point1,point2,vel_lab):
    delta_x = point2[0] - point1[0]
    delta_y = point2[1] - point1[1]
    phi = math.atan2(delta_y, delta_x)
    v_parallel = math.cos(phi)*vel_lab[0]+math.sin(phi)*vel_lab[1]
    v_perp = -math.sin(phi)*vel_lab[0]+math.cos(phi)*vel_lab[1]
    return [v_parallel,v_perp]

def solve_flux(points,lines,centerpoints,v_inf):
    N = np.shape(lines)[0]
    V = np.zeros((N,N))
    perp_components = np.zeros((N))
    for i,line in enumerate(lines):
        point1 = [points[line[0]][0],points[line[0]][1]]
        point2 = [points[line[1]][0],points[line[1]][1]]
        delta_x = point2[0] - point1[0]
        delta_y = point2[1] - point1[1]
        phi = math.atan2(delta_y, delta_x)
        perp_components[i] = -math.sin(phi)*v_inf   #assumes flow at infinity to be parallel to x-axis
    for j in range(0,N):
        pointj1 = [points[lines[j][0]][0],points[lines[j][0]][1]]
        pointj2 = [points[lines[j][1]][0],points[lines[j][1]][1]]
        for i in range(0,N):
            pointi1 = [points[lines[i][0]][0],points[lines[i][0]][1]]
            pointi2 = [points[lines[i][1]][0],points[lines[i][1]][1]]
            if (j == i):
                V[j][i] = -0.5
            else:
                point_relative = relative_position(pointi1,pointi2,centerpoints[i],centerpoints[j])
                local_vel = velocity_components(point_relative,segment_length(pointi1,pointi2))
                v_lab = vel_lab(pointi1,pointi2,local_vel)
                V[j][i] = vel_inverse(pointj1,pointj2,v_lab)[1]
    sigmas = linalg.solve(V,perp_components)
    return sigmas

def evaluate_flux_field(xmin,xmax,ymin,ymax,xpts,ypts,points,lines,centerpoints,sigmas,v_inf):
    dx = (xmax-xmin)/float(xpts)
    dy = (ymax-ymin)/float(ypts)
    u = np.zeros((xpts,ypts,2))
    for xpt in range(xpts):
        for ypt in range(ypts):
            evaluationpoint = [xmin+xpt*dx,ymin+ypt*dy]
            u[xpt][ypt] = [v_inf,0.0]
            for i in range(len(sigmas)):
                point1 = [points[lines[i][0]][0],points[lines[i][0]][1]]
                point2 = [points[lines[i][1]][0],points[lines[i][1]][1]]

                point_relative = relative_position(point1,point2,centerpoints[i],evaluationpoint)
                local_vel = velocity_components(point_relative,segment_length(point1,point2))
                v_lab = vel_lab(point1,point2,local_vel)

                u[xpt][ypt][0] += sigmas[i]*v_lab[0]
                u[xpt][ypt][1] += sigmas[i]*v_lab[1]
    return u

def evaluate_flux_field_border(points,lines,centerpoints,sigmas,v_inf):
    u = np.zeros((np.shape(centerpoints)[0],2))
    for k,evaluationpoint in enumerate(centerpoints):

        u[k] = [v_inf,0.0]
        for i in range(len(sigmas)):
            
            if (i == k):
                point1 = [points[lines[i][0]][0],points[lines[i][0]][1]]
                point2 = [points[lines[i][1]][0],points[lines[i][1]][1]]
                v_lab = vel_lab(point1,point2,[0.0,-0.5])
                u[k][0] += sigmas[i]*v_lab[0]
                u[k][1] += sigmas[i]*v_lab[1]
            
            else:
                point1 = [points[lines[i][0]][0],points[lines[i][0]][1]]
                point2 = [points[lines[i][1]][0],points[lines[i][1]][1]]

                point_relative = relative_position(point1,point2,centerpoints[i],evaluationpoint)
                local_vel = velocity_components(point_relative,segment_length(point1,point2))
                v_lab = vel_lab(point1,point2,local_vel)

                u[k][0] += sigmas[i]*v_lab[0]
                u[k][1] += sigmas[i]*v_lab[1]
            
    return u

def evaluate_sigmas_border(points,lines,centerpoints,sigmas,v_inf):
    u = np.zeros((np.shape(centerpoints)[0],2))
    for k,evaluationpoint in enumerate(centerpoints):
        point1 = [points[lines[k][0]][0],points[lines[k][0]][1]]
        point2 = [points[lines[k][1]][0],points[lines[k][1]][1]]
        v_lab = vel_lab(point1,point2,[0.0,sigmas[k]])

        u[k][0] = v_lab[0]
        u[k][1] = v_lab[1]
    return u

"""
#testing method on a single panel
lines = [[0,1]]
points = [[-0.5,0.0],[0.5,0.0]]
centerpoints = [[0.0,0.0]]
centerpoints = np.array(centerpoints)
"""


#full mesh
meshsize_middle = 0.01
meshsize_ends = 0.001
angle = 45.0
tape = twodim_chevron(meshsize_middle,meshsize_ends,angle)
#tape = twodim_corner(meshsize_middle,meshsize_ends)
#tape = twodim_tape(meshsize_middle,meshsize_ends)
tape.write("rezultati/testmesh.vtk")

points = tape.points
lines = np.vstack(np.array([cells.data for cells in tape.cells if cells.type == "line"]))
centerpoints = np.empty((np.shape(lines)[0],2)) #xi and eta values

for i,line in enumerate(lines):
    centerpoints[i][0] = 0.5*(points[line[0]][0]+points[line[1]][0])
    centerpoints[i][1] = 0.5*(points[line[0]][1]+points[line[1]][1])

G = potential_contributions(points,lines,centerpoints)
bvec = np.ones((np.shape(lines)[0]))
sigmas = linalg.solve(G,bvec)

xmin = -2.0
xmax = 2.0
ymin = -2.0
ymax = 2.0
xpts = 113
ypts = 113
"""
u = evaluate_field(xmin,xmax,ymin,ymax,xpts,ypts,points,lines,centerpoints,sigmas)

sigma_file_name = "rezultati/chargedensities_corner_" + "%d" % np.shape(lines)[0] + "_%.0f_deg" % (angle) + ".dat"
sigmas_out = open(sigma_file_name,"w+")
for n in range(len(sigmas)):
    sigmas_out.write("%f %f %f\n" % (centerpoints[n][0],centerpoints[n][1],sigmas[n]))
sigmas_out.close()

u_file_name = "rezultati/spatialprofile_corner_" + "%d" % np.shape(lines)[0] + "_%.0f_deg" % (angle) + ".dat"
u_out = open(u_file_name,"w+")
for j in range(ypts):
    for i in range(xpts):
        u_out.write("%f %f %.12f\n" % (xmin+i*((xmax-xmin)/float(xpts)),ymin+j*((xmax-xmin)/float(xpts)),u[i][j]))
    u_out.write("\n")
u_out.close()
"""
print("Capacitance of electrode for %d panels: %f" % (np.shape(lines)[0],evaluate_capacitance(points,lines,sigmas)))


"""
#calculation of capacitance as a function of "mesh" fineness
cap_file_name = "rezultati/capacitance_chevron_01-add.dat"
cap_out = open(cap_file_name,"w+")

#meshsize_middle = [0.4,0.33,0.2,0.15,0.1,0.06,0.03,0.01,0.003]
#meshsize_ends = [0.04,0.033,0.02,0.015,0.01,0.006,0.003,0.001,0.0003]
meshsize_middle = 0.01
meshsize_ends = 0.001
angles = [55.0,54.0,53.0,52.0,51.0,50.0,49.0,48.0,47.0,46.0,45.0,44.0,43.0,42.0,41.0,40.0,39.0,38.0,37.0,36.0,35.0]

for p in range(len(angles)):
    #tape = twodim_tape(meshsize_middle[p],meshsize_ends[p])
    tape = twodim_chevron(meshsize_middle,meshsize_ends,angles[p])
    points = tape.points
    lines = np.vstack(np.array([cells.data for cells in tape.cells if cells.type == "line"]))
    centerpoints = np.empty((np.shape(lines)[0],2)) #xi and eta values

    for i,line in enumerate(lines):
        centerpoints[i][0] = 0.5*(points[line[0]][0]+points[line[1]][0])
        centerpoints[i][1] = 0.5*(points[line[0]][1]+points[line[1]][1])

    G = potential_contributions(points,lines,centerpoints)
    bvec = np.ones((np.shape(lines)[0]))
    sigmas = linalg.solve(G,bvec)
    cap = evaluate_capacitance(points,lines,sigmas)
    cap_out.write("%f %.10f\n" % (angles[p],cap))
cap_out.close()
"""

#hydrodynamic part of assignment
numpanels = 100
"""
t = 10
points,lines,centerpoints = NACA_profile(numpanels,t)
airfoil_file_name = "rezultati/NACA_" + "N_%d_" % numpanels + "t_%d" % t + ".dat"
airfoil_profile_out = open(airfoil_file_name,"w+")
for point in points:
    airfoil_profile_out.write("%f %f\n" % (point[0],point[1]))
airfoil_profile_out.close()
"""
"""
A = 0.04
B = 0.08
points,lines,centerpoints = Zhukhovsky_profile(numpanels,A,B)
airfoil_file_name = "rezultati/Zhukhovsky_" + "N_%d_" % numpanels + "A_%.2f_" % A + "B_%.2f" % B + ".dat"
airfoil_profile_out = open(airfoil_file_name,"w+")
for point in points:
    airfoil_profile_out.write("%f %f\n" % (point[0],point[1]))
airfoil_profile_out.close()
"""
"""
b = 0.5
#phi = 0.0
phi = math.pi/4.0
points,lines,centerpoints = elliptical_profile(numpanels,b,phi)
airfoil_file_name = "rezultati/ellipse_" + "N_%d_" % numpanels + "b_%1.2f_" % b + "phi_%.2f" % phi + ".dat"
airfoil_profile_out = open(airfoil_file_name,"w+")
for point in points:
    airfoil_profile_out.write("%f %f\n" % (point[0],point[1]))
airfoil_profile_out.close()

xmin = -2.0
xmax = 3.0
ymin = -2.0
ymax = 2.0
xpts = 250
ypts = 175
v_inf = 1.0 #flow velocity at infinity. We assume the flow to be along x-axis for now.

sigmas = solve_flux(points,lines,centerpoints,v_inf)

u = evaluate_flux_field(xmin,xmax,ymin,ymax,xpts,ypts,points,lines,centerpoints,sigmas,v_inf)
#u_file_name = "rezultati/Zhukhovsky_spatialprofile" + "N_%d_" % numpanels + "A_%.2f_" % A + "B_%.2f" % B + ".dat"
#u_file_name = "rezultati/NACA_spatialprofile_" + "%d" % np.shape(lines)[0] + "_t_%d" % t + ".dat"
u_file_name = "rezultati/ellipse_spatialprofile_" + "%d" % np.shape(lines)[0] + "_b_%1.2f" % b + "_phi_%.2f" % phi + ".dat"
u_out = open(u_file_name,"w+")
for j in range(ypts):
    for i in range(xpts):
        u_out.write("%f %f %.12f %.12f\n" % (xmin+i*((xmax-xmin)/float(xpts)),ymin+j*((ymax-ymin)/float(ypts)),u[i][j][0],u[i][j][1]))
    u_out.write("\n")
u_out.close()
"""
"""
u_prime = evaluate_sigmas_border(points,lines,centerpoints,sigmas,v_inf)
#u_prime_name = "rezultati/Zhukhovsky_panelprofile" + "N_%d_" % numpanels + "A_%.2f_" % A + "B_%.2f" % B + ".dat"
#u_prime_name = "rezultati/NACA_panelprofile_" + "%d" % np.shape(lines)[0] + "_t_%d" % t + ".dat"
u_prime_name = "rezultati/ellipse_panelprofile_" + "%d" % np.shape(lines)[0] + "_b_%1.2f" % b + "_phi_%.2f" % phi + ".dat"
u_out = open(u_prime_name,"w+")
for i in range(np.shape(u_prime)[0]):
    u_out.write("%f %f %.12f %.12f\n" % (centerpoints[i][0],centerpoints[i][1],u_prime[i][0],u_prime[i][1]))
u_out.close()
"""
"""
u_prime = evaluate_flux_field_border(points,lines,centerpoints,sigmas,v_inf)
#u_prime_name = "rezultati/Zhukhovsky_panelprofile" + "N_%d_" % numpanels + "A_%.2f_" % A + "B_%.2f" % B + ".dat"
#u_prime_name = "rezultati/NACA_panelprofile_" + "%d" % np.shape(lines)[0] + "_t_%d" % t + ".dat"
u_prime_name = "rezultati/ellipse_borderprofile_" + "%d" % np.shape(lines)[0] + "_b_%1.2f" % b + "_phi_%.2f" % phi + ".dat"
u_out = open(u_prime_name,"w+")
for i in range(np.shape(u_prime)[0]):
    point1 = [points[lines[i][0]][0],points[lines[i][0]][1]]
    point2 = [points[lines[i][1]][0],points[lines[i][1]][1]]
    delta_x = point2[0] - point1[0]
    delta_y = point2[1] - point1[1]
    phi = math.atan2(delta_y, delta_x)
    u_out.write("%f %f %.12f %.12f\n" % (centerpoints[i][0],centerpoints[i][1],u_prime[i][1]*math.cos(phi) - u_prime[i][0]*math.sin(phi),elliptical_analytic_sol(b,centerpoints[i],1.0)))
u_out.close()
"""