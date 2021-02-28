#Implementation of 2D diffusion limited aggregation process using single-particle random walks
#and irreversible adsorption on contact with existing network
#Written by Timotej Žuntar for Soft Matter Physics course in January 2020.

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import os
import random
import math
import pandas as pd

movelist = ["up", "down", "left", "right"]  #no diagonal movements for now

def init():
    choice = random.choice(movelist)

    if (choice == "up"):
        pos = [random.randint(1,399), 399]
    if (choice == "down"):
        pos = [random.randint(1,399), 1]
    if (choice == "left"):
        pos = [1, random.randint(1,399)]
    if (choice == "right"):
        pos = [399, random.randint(1,399)]

    return pos

def move(movelist, pos):

    choice = random.choice(movelist)
    #print(choice)

    if (choice == "up"):
        pos[1] += 1
    if (choice == "down"):
        pos[1] -= 1
    if (choice == "left"):
        pos[0] -= 1
    if (choice == "right"):
        pos[0] += 1

    return pos

def nextto(grid,pos):     
    if (grid[pos[0]-1,pos[1]] == 1 or grid[pos[0]+1,pos[1]] == 1 or grid[pos[0],pos[1]-1] == 1 or grid[pos[0],pos[1]+1] == 1):
        return True
    else:
        return False

def rad_gyr(grid, size, movecount, num, file):
    COM = [0.,0.]
    N = 0
    Rg = 0.

    for i in range(0, size-1):
        for j in range(0, size-1):
            if (grid[i,j] == 1):
                COM[0] += i
                COM[1] += j
                N += 1

    COM[0] /= N
    COM[1] /= N
    
    for i in range(0, size-1):
        for j in range(0, size-1):
            if (grid[i][j] == 1):
                Rg += (i-COM[0])**2 + (j-COM[1])**2
    
    Rg /= N
    Rg = math.sqrt(Rg)
    file.write("\n%f %d %d %.8f %f %f" % (movecount, num, N, Rg, COM[0], COM[1]))

def corrs(grid, size, file):
    N = 0
    positions = []
    #distances = np.array([[0.,0], [1.,0], [2.,0]])
    

    for i in range(0, size-1):
        for j in range(0, size-1):
            if (grid[i,j] == 1):
                N += 1
                positions.append([i,j])

    distances = np.array([[0.,2*N]])

    for numindex, pos in enumerate(positions):
        
        if (numindex % 1000 == 0):
            print("Calculating for particle %d/%d\n" % (numindex + 1, N))

        for pos2 in positions:

            if (pos[0] == pos2[0]):
                current = math.sqrt((pos[1]-pos2[1])**2)
                newline = np.array([[round(current,6),1]])
                distances = np.concatenate((distances, newline))

            if (pos[1] == pos2[1]):
                current = math.sqrt((pos[0]-pos2[0])**2)
                newline = np.array([[round(current,6),1]])
                distances = np.concatenate((distances, newline))                

            #included = False
            #current = math.sqrt((pos[0]-pos2[0])**2 + (pos[1]-pos2[1])**2)
            """
            for index, distance in enumerate(distances[:,0]):
                if (np.isclose(distance, current, 1E-8, 0.00001) == True):
                    included = True
                    distances[index,1] += 1
                    break
            if (included == False):
                newline = np.array([[current,1]])
                distances = np.concatenate((distances, newline))
            """
            #newline = np.array([[round(current,6),1]])
            #distances = np.concatenate((distances, newline))
    
    #distances.sort(key=lambda x: x[0])
    df = pd.DataFrame(distances, columns = ["distance", "occurences"])
    df_sum = df.groupby("distance",as_index=False).sum()

    df_sum = df_sum.sort_values(by=["distance"])
    df_sum["occurences"] = df_sum["occurences"].apply(lambda x: x/(4*N))
    #distances[distances[:,0].argsort()]
    #distances[:,1] = np.divide(distances[:,1],N)
    """
    for element in distances:
        file.write("%f %f\n" % (element[0],element[1]))
    """
    print(df_sum.shape)
    for number in range(df_sum.shape[0]):
        file.write("%f %f\n" % (df_sum.iloc[number,0],df_sum.iloc[number,1]))
            

grid = np.zeros((401,401),dtype=np.int)
grid[200,200] = 1 #set seed

numparticles = 1
moves = 0.

os.mkdir("cluster")
gyration = open("cluster_radgyr.dat", "w+")

for cycle in range(10000000):
    pos0 = init()
    flag = nextto(grid,pos0)

    while (flag == False):
        pos0 = move(movelist,pos0)
        moves += 1
        if (pos0[0] < 1 or pos0[0] > 399 or pos0[1] < 1 or pos0[1] > 399):
            break
        else:
            flag = nextto(grid,pos0)
        
    if (flag == True):
        grid[pos0[0],pos0[1]] = 1
        numparticles += 1

        rad_gyr(grid, 401, moves, cycle, gyration)

        if ((numparticles % 10) == 0):
            #plt.imsave("%d.png" % numparticles, np.array(grid), cmap=cm.gray)
            plt.imsave("cluster/" + str(numparticles).zfill(5) + ".png", np.array(grid), cmap=cm.gray)
        print(cycle)

    if (numparticles >= 6000):
        break
    if ((all(i < 1 for i in grid[:,0]) == False) or (all(i < 1 for i in grid[:,400]) == False) or (all(i < 1 for i in grid[0,:]) == False) or (all(i < 1 for i in grid[400,:]) == False)):
        break

gyration.close()

correl = open("cluster_corr.dat","w+")
corrs(grid, 401, correl)
correl.close()

