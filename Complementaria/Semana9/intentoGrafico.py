import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

def getRandomList():
    r = np.random.rand()
    theta = np.random.rand()*360
    
    x = r*np.cos(theta)
    y = r*np.sin(theta)
    z = np.random.rand()*3
    return [x,y,z]

def main():
    x = np.array([0,0,3]) #the tip of the cone
    dir = np.array([0,0,-1]) # the normalized axis vector, pointing from the tip to the base
    h = 3
    r = 1

    p = np.array(getRandomList()) # point to test
    cone_dist =  np.dot(p - x, dir)
    cone_radius = (cone_dist / h) * r

    orth_distance = np.linalg.norm((p - x) - cone_dist * dir)
    is_point_inside_cone = (orth_distance < cone_radius)

    return p, is_point_inside_cone

def main1():
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    #ax.scatter3D(x, y, z)

    # cono
    ax.scatter3D(0,0,3)
    angulos = np.linspace(0, 360, 80)

    for theta in angulos:
        
        r = 1
        x =  r*np.cos(theta)
        y =  r*np.sin(theta)
        z = 0
        ax.scatter3D(x, y, z, c="b")
        #ax.quiver(1, 2, 3, 4, 5, 6)
        ax.plot([x, 0], [y, 0],zs=[z, 3], c="b", linewidth=5, markersize=20)

    #plt.show()

    for i in range(4):
        pass
        #ax.plot([VecStart_x[i], VecEnd_x[i]], 
         #   [VecStart_y[i],VecEnd_y[i]],
          #  zs=[VecStart_z[i],VecEnd_z[i]])

    return ax

if __name__ == "__main__":
    #print(main())
    #help(time)
    ax = main1()

    p, cond = main()
    print(cond)
    ax.scatter3D(p[0], p[1], p[2], c="r")
    plt.show()
