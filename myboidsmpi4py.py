# -*- coding: utf-8 -*-

'''
@author: Imad
'''

from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.image as mpimg
import time



my_rank = MPI.COMM_WORLD.Get_rank()
p = MPI.COMM_WORLD.Get_size()
comm = MPI.COMM_WORLD
boids = 100 # enter number of boids you want to generate
start = time.time()
print('start', start)


frames = 300 # enter number of seconds, or frames if animating
boid_num =boids//p
figure = plt.figure()
axes = figure.add_subplot(111, projection='3d')
axes.set_xlim3d([0, 2000])
axes.set_ylim3d([0, 2000])
axes.set_zlim3d([0, 2000])
axes.set_xlabel('X axis')
axes.set_ylabel('Y axis')
axes.set_zlabel('Z axis')
scatter = axes.scatter([], [], [], color='black')



positions = np.empty([3, boid_num])
velocities = np.empty([3, boid_num])
def update_boids(positions, velocities):
    
        limits = np.array([2000, 2000, 2000])

        # generates initial positions
        positions = np.random.rand(3, boid_num) * limits[:, np.newaxis]
        print('positions')
        print(positions)
        print(positions.shape)
        # generates initial velocities
        velocities = np.random.randint(1,10,size=(3,boid_num))*np.random.choice([-1,1],(3,boid_num))
        velocities = np.array(velocities)   
        print('velocities')
        print(velocities,velocities.shape)
        # initialises x,y, and z axis lists containing all positions of boids vs time
        xlist = []
        xlist.append(positions[0].tolist())
        ylist = []
        ylist.append(positions[1].tolist())
        zlist = []
        zlist.append(positions[2].tolist())
        # loops over number of seconds
        for i in range(0,frames):
        
            # Moving to centre of flock
            
            move_to_middle_strength = 0.005
            middle = np.mean(positions, 1)
            direction_to_middle = positions - middle[:, np.newaxis] 
            '''for i, values in np.ndenumerate(direction_to_middle):
                if direction_to_middle[i] > 1000:
                    velocities -= 5'''
            velocities = velocities.astype(np.float) 
            velocities -= direction_to_middle * move_to_middle_strength
            
            # Avoiding collisions
            velocities = velocities.astype(np.float) 
            awareness = 20
            avoidspeed = 2
            xvals = positions[0]
            yvals = positions[1]
            zvals = positions[2]
            diffx = np.diff(xvals)
            diffx = np.array(diffx)
            diffy = np.diff(yvals)
            diffy = np.array(diffy)
            diffz = np.diff(zvals)
            diffz = np.array(diffz)
            for index, values in np.ndenumerate(diffx):
                if np.abs(diffx[index]) < awareness:
                    velocities[0][index] -= avoidspeed*np.sign(diffx[index])
        
            for index, values in np.ndenumerate(diffy):
                if np.abs(diffy[index]) < awareness:
                    velocities[1][index] -= avoidspeed*np.sign(diffy[index])
        
            for index, values in np.ndenumerate(diffz):
                if np.abs(diffz[index]) < awareness:
                    velocities[2][index] -= avoidspeed*np.sign(diffz[index])
                    
            # velocities matching
            velocities = velocities.astype(np.float) 
            awarenessv = 200
            coherence = 0.01
            xvals = positions[0]
            yvals = positions[1]
            zvals = positions[2]
            xvels = velocities[0]
            yvels = velocities[1]
            zvels = velocities[2]
            diffx = np.diff(xvals)
            diffx = np.array(diffx)
            diffy = np.diff(yvals)
            diffy = np.array(diffy)
            diffz = np.diff(zvals)
            diffz = np.array(diffz)
            xvelavg = np.average(xvels)
            yvelavg = np.average(yvels) 
            zvelavg = np.average(zvels)
            for index, values in np.ndenumerate(diffx):
                for index, values in np.ndenumerate(xvelavg):
                    if np.abs(diffx[0][index]) < awarenessv:
                        velocities[0][index] += xvelavg[index] * coherence    
            for index, values in np.ndenumerate(diffy):
                for index, values in np.ndenumerate(yvelavg):
                    if np.abs(diffy[0][index]) < awarenessv:
                        velocities[1][index] += yvelavg[index] * coherence  
            for index, values in np.ndenumerate(diffz):
                for index, values in np.ndenumerate(zvelavg):
                    if np.abs(diffz[0][index]) < awarenessv:
                        velocities[2][index] += zvelavg[index] * coherence      
            # added randomness factor to velocity
            #velocities += np.random.randint(-10,10,size=(3,boid_num))
            # avoid edges
            '''for index, values in np.ndenumerate(positions[0]):
                if 1800 < positions[0][index] < 200:
                    velocities[0][index] -= 2*np.sign(positions[0][index])
            for index, values in np.ndenumerate(positions[1]):
                if 1800 < positions[1][index] < 200:
                    velocities[1][index] -= 2*np.sign(positions[1][index])
            for index, values in np.ndenumerate(positions[2]):
                if 1800 < positions[2][index] < 200:
                    velocities[2][index] -= 2*np.sign(positions[2][index])'''
            '''for i in velocities():
                if velocities[0][i] < 2:
                    velocities[0][i] +=  0.2
                if velocities[1][i] < 2:
                    velocities[0][i] +=  0.2
                if velocities[2][i] < 2:
                    velocities[0][i] +=  0.2'''
            positions += velocities # updating positions
            # compiling positions vs time for plotting later
            xlist.append(positions[0].flatten().tolist())
            ylist.append(positions[1].flatten().tolist())
            zlist.append(positions[2].flatten().tolist())
            i += 1
        return (xlist,ylist,zlist)


start = MPI.Wtime() # start of mpi4py timer
if my_rank == 0:
    # master core collects data from other cores by receiving update_boids function resulats a,b,c
    totalx = []
    totaly = []
    totalz = []
    for k in range(1,p):
        update_boids = comm.recv(source=k)
        a, b, c = update_boids
        totalx.append(a)
        totaly.append(b)
        totalz.append(c)
        
        print("PE", my_rank, "<-",",")
        
else :
    print("PE", my_rank, "->",",",)
    # all the wroker cores receive the function and execute it, to send answer to master
    comm.send(update_boids(positions,velocities), dest=0)

# when all is done, the lists are turned into arrays and plotted to form a video animation 
if (my_rank ==0):
    totalx = np.asarray(totalx)
    print('totalxstart',totalx.shape)
    totalx = np.swapaxes(totalx, 1, 0)

    totaly = np.asarray(totaly)
    totaly = np.swapaxes(totaly, 1, 0)
    totalz = np.asarray(totalz)
    totalz = np.swapaxes(totalz, 1, 0)
    print('finalx',totalx,totalx.shape)
    print('totalxframe1',totalx[0].flatten(),totalx[0].flatten().shape)
# you can remove the animation code below if you like to find the time only of executing mpi4py
    def animate(frame):
        totalx[frame]
        totaly[frame]
        totalz[frame]
        scatter._offsets3d = (totalx[frame].flatten(),totaly[frame].flatten(),totalz[frame].flatten())
    
    anim = animation.FuncAnimation(figure, animate,
                                   frames=frames, interval=50)


    anim.save('myboids_pmpi4py.mp4')
    

end = MPI.Wtime() # end of mpi4py timer
print('end', end)
print('time taken:', (end - start),'seconds') # total time taken is calculated and printed 
MPI.Finalize # MPI is told to safely finish off its programme run
