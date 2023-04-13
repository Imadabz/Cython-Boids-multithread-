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
import random
import os
import pygame
import time
import time

start = time.time()
print('start', start)


# setting limits of boid spawn volume 
limits = np.array([2000, 2000, 2000])
boid_num =100 # setting number of boids to generate
# setting initial boids positions
positions = np.random.rand(3, boid_num) * limits[:, np.newaxis]
print('positions')
print(positions)
print(positions.shape)

# setting initial boids velocities
velocities = np.random.randint(1,10,size=(3,boid_num))*np.random.choice([-1,1],(3,boid_num))
velocities = np.array(velocities)   
print('velocities')
print(velocities,velocities.shape)
# initialising graph for animation
figure = plt.figure()
axes = figure.add_subplot(111, projection='3d')
axes.set_xlim3d([0, 2000])
axes.set_ylim3d([0, 2000])
axes.set_zlim3d([0, 2000])
axes.set_xlabel('X axis')
axes.set_ylabel('Y axis')
axes.set_zlabel('Z axis')
scatter = axes.scatter([], [], [],s=1, color='black')


# boids simulation function that updates velocities and positions
def update_boids(positions, velocities):
    
    # Moving to centre of flock
    
    move_to_middle_strength = 0.01
    middle = np.mean(positions, 1)
    direction_to_middle = positions - middle[:, np.newaxis] 
    '''for i, values in np.ndenumerate(direction_to_middle):
        if direction_to_middle[i] > 1000:
            velocities -= 5'''
    velocities = velocities.astype(np.float) 
    velocities -= direction_to_middle * move_to_middle_strength
    
    # Avoiding collisions
    
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
    awarenessv = 200
    coherence = 0.2
    xvels = velocities[0]
    yvels = velocities[1]
    zvels = velocities[2]
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
    # added randomness factor to velocity, not used
    #velocities += np.random.randint(-10,10,size=(3,boid_num))
    # avoid edges, curently not used
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
    positions += velocities # updates the positions array witht the new altered velocities
    
# defining and executing an animation function to produce a video, can be removed to find just function time
def animate(frame):
    update_boids(positions, velocities)
    scatter._offsets3d = (positions)


anim = animation.FuncAnimation(figure, animate,
                               frames=200, interval=50)
# saving the animation
anim.save('myboids_serial.mp4')
end = time.time()
print('end', end)
print('time taken:', (end - start),'seconds') # printing the time taken to execute programme and save animation

