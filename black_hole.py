import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm
from tqdm import tqdm

omegas = np.random.randint(1, 6, size=4) #random disk parameters
phis = np.random.randint(1, 6, size=2) #random disk parameters

# intensity of the disk
# generate a spiral in the xy plane with exponential decay in z and r
def disc_color(x, y, z):
    r = np.sqrt(x**2 + y**2)
    t = r*2 + np.arctan2(y, x)
    c = np.sin(t*omegas[0])+2*np.sin(t*omegas[1])+3*np.sin(t*omegas[2]+phis[0])+0.5*np.sin(t*omegas[3]+phis[1]) #spirale
    color = (c**2 + 5)*np.exp(-r/4)*np.exp(-np.abs(z*30)) #exponential disk
    color[np.abs(z)>0.2] = 0
    color[np.abs(r)<1.01] = 0
    return color

# time derivative to use with rk4
def frk4(q,h):
    r = -3/2*h**2 * norm(q[:3],axis=0)**4
    velocity = np.array(q[3:])
    acceleration = q[:3]/r[None,:,:]
    return np.concatenate((velocity, acceleration), axis=0)

# step of Runge Kutta
def rk4(q, dt, h):
    k1 = frk4(q,h)
    k2 = frk4(q+k1*dt/2,h)
    k3 = frk4(q+k2*dt/2,h)
    k4 = frk4(q+k3*dt,h)
    return q+(k1+2*k2+2*k3+k4)/6*dt

# initial positions and velocities
# q is an array of coordinates [x,y,z,vx,vy,vz]
# where x[i] is the x position of pixel i on the camera
# and vx[i] is the x component of the vector pointing from the focal point of the camera to pixel i
# h is the angluar momentum
def initq(l, x0, z0, theta):
    x0 = np.array([-x0, 0, z0])
    depth = 0.5
    width = 0.5
    u = -depth*x0/norm(x0)
    v = vect_rot(np.array([0, 1, 0]), x0/norm(x0), theta)*width
    w = np.cross(u/norm(u), v)
    A = x0+u-w-v
    i = np.arange(l)
    position = A[:, None, None]+(v[:, None, None]*i[None,:,None]+w[:, None, None]*i[None,None,:])/l*2
    velocity = position - x0[:,None,None]
    velocity /= norm(velocity, axis=0)
    q = np.zeros((6,l,l))
    q[:3] = position+velocity*np.random.random(size=(l,l))*0.01
    q[3:] = velocity
    h = np.linalg.norm(np.cross(position, velocity, axis=0), axis=0)
    return q,h

def ray_tracing(l, x0, z0, theta):
    q, h = initq(l, x0, z0, theta) # set initial positions and velocity from the camera position
    color = np.zeros((l,l))
    opacity = np.ones((l,l))
    tmax = norm([x0,z0])*2
    t = 0
    dt = 0.02
    pbar = tqdm(total=tmax)
    while t<tmax: # Runge Kutta integration of the light trajectory
        pbar.update(dt)
        color, opacity = updatecolor(q, color, opacity)
        q = rk4(q, dt, h)
        t += dt
    return np.flip(color.T)

# update each pixel color based on the disk intensity at the current position
def updatecolor(q, color, opacity):
    c = disc_color(*q[:3])
    color += c*opacity
    opacity *= 1-c/40
    r = norm(q[:3], axis=0)
    opacity[r<0.4] = 0
    return color, opacity

def vect_rot(v, k, theta):
    A = v * np.cos(theta)
    B = np.cross(k, v) * np.sin(theta)
    C = k * (1 - np.cos(theta)) * np.dot(k, v)
    return A + B + C

theta = 0.0 # view angle. theta=0 is in the plane of the disk
x0 = 8 # x position of the camera
z0 = 1 # z position of the camera
l = 2**10 # image size
M = ray_tracing(l, x0, z0, theta)
plt.imsave('bh.png', np.nan_to_num(M), cmap='inferno')
plt.matshow(M, cmap='inferno')
plt.axis("off")
plt.gca().set_aspect('equal')
plt.show()
