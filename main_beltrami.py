# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 07:11:09 2024


"""
import jax
import jax.numpy as jnp
from jax import random, grad, jit, jacfwd, vmap
import optax
import os
import h5py
import jax
import jax.numpy as jnp
import equinox as eqx
import equinox.nn as nn
from typing import Any, Callable, List, Tuple, Optional




import argparse
import jax
from Reimann_nets import *
# Parse seed argument
parser = argparse.ArgumentParser(description="Set random seed for reproducibility.")
parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility (default: 42)')
args = parser.parse_args()

# Use the seed
seed = args.seed
print(f"Using random seed: {seed}")


from utils import *
from models import * 

import jax
import jax.numpy as jnp
from jax import random

jax.config.update("jax_enable_x64", True)

# Initialize random layer parameters
def random_layer_params(m, n, key, scale=1e-1):
    w_key, b_key = random.split(key)
    return scale * random.normal(w_key, (n, m)), scale * random.normal(b_key, (n,))

# Initialize network parameters
def init_params(sizes, key):
    keys = random.split(key, len(sizes))
    return [random_layer_params(m, n, k) for m, n, k in zip(sizes[:-1], sizes[1:], keys)]



# Define the MLP model
def mlp(activation):
    def model(params, inpt):
        hidden = inpt
        for w, b in params[:-1]:
            outputs = jnp.dot(w, hidden) + b
            hidden = activation(outputs)
        final_w, final_b = params[-1]
        return jnp.dot(final_w, hidden) + final_b
    return model

# Activation function and its derivatives
def tanh(x):
    return jnp.tanh(x)

def tanh_prime(x):
    return 1.0 - jnp.tanh(x)**2

def tanh_double_prime(x):
    return -2.0 * jnp.tanh(x) * (1.0 - jnp.tanh(x)**2)

def tanh_triple_prime(x):
    tanh_x = jnp.tanh(x)
    return -2 * (1 - tanh_x**2) * (1 - 3 * tanh_x**2)

# Initialize model parameters
key = random.PRNGKey(seed)





selectors=build_basis(n=4)

key = random.PRNGKey(seed)
sizes = [4, 50,50, 50,50, 21]

activation = tanh
activation_prime = tanh_prime
activation_double_prime = tanh_double_prime

def derivative_propagation(params, x):
    W, b = zip(*params)
    z = x
    dz_dx = jnp.eye(len(x))
    d2z_dxx = jnp.zeros((len(x), len(x), len(x)))

    for w, b in params[:-1]:
        z = jnp.dot(w, z) + b
        dz_dx = jnp.dot(w, dz_dx)
        d2z_dxx = jnp.einsum('ij,jkl->ikl', w, d2z_dxx)

        sigma_prime = activation_prime(z)
        sigma_double_prime = activation_double_prime(z)


        dz_dx_old = dz_dx.copy()
        d2z_dxx_old = d2z_dxx.copy()


        dz_dx = sigma_prime[:, None] * dz_dx
        d2z_dxx = (sigma_double_prime[:, None, None] * 
                   jnp.einsum('ni,nj->nij', dz_dx_old, dz_dx_old) +
                   sigma_prime[:, None, None] * d2z_dxx)
      
        z = activation(z)

    final_w, final_b = params[-1]
    z = jnp.dot(final_w, z) + final_b
    dz_dx = jnp.dot(final_w, dz_dx)
    d2z_dxx = jnp.einsum('ij,jkl->ikl', final_w, d2z_dxx)


    return z, dz_dx, d2z_dxx
# Example input

model = jax.jit(mlp(activation))
params = init_params(sizes, key)



w=1
def compute_H_ab(x, params):
    # Compute second derivatives (Hessians) of v_i(x)
    _, _, d2z_dxx = derivative_propagation(params, x)  # Shape: (7, 3, 3)
    
    # Extract Hessians for v_i
    H_v = [jnp.squeeze(matrix, axis=0) for matrix in jnp.split(d2z_dxx, d2z_dxx.shape[0], axis=0)]
  

    # Compute H_ab using the selectors
    H_ab = jnp.zeros((4, 4))
    for i in range(21):
       if i not in [0,1,2,6,7,11]: 
        
        
        
        term=w*jnp.einsum('acbd,cd->ab', selectors[i], H_v[i])

        H_ab +=term 

    return H_ab





def model_uvwp(params,x):
    
    
    T=compute_H_ab(x, params)

    T_t=T[1:,1:]
    u = T[0, 1] 
    v = T[0, 2]  
    w=  T[0, 3] 
    m=u**2+v**2+w**2
    p=(1/3)*(T_t.trace()-m)
    return jnp.array([u,v,w,p])





def model_u(params,x):
    
    
    T=compute_H_ab(x, params)

    T_t=T[1:,1:]
    u = T[0, 1] 
    v = T[0, 2]  
    w=  T[0, 3] 
    m=u**2+v**2+w**2
    p=(1/3)*(T_t.trace()-m)
    return jnp.array([u,v,w])



activation_triple_prime = tanh_triple_prime

def derivative_propagation3(params, x):
       W, b = zip(*params)
       z = x
       dz_dx = jnp.eye(len(x))
       d2z_dxx = jnp.zeros((len(x), len(x), len(x)))
       d3z_dxxx = jnp.zeros((len(x), len(x), len(x), len(x)))
     
    
       for w, b in params[:-1]:
           # Linear layer: z = Wx + b
           z = jnp.dot(w, z) + b
           dz_dx = jnp.dot(w, dz_dx)
           d2z_dxx = jnp.einsum('ij,jkl->ikl', w, d2z_dxx)
           d3z_dxxx = jnp.einsum('ij,jklm->iklm', w, d3z_dxxx)
          
           # Activation layer
           sigma_prime = activation_prime(z)
           sigma_double_prime = activation_double_prime(z)
           sigma_triple_prime = activation_triple_prime(z)
         
           # Save the old derivatives for use in the higher-order derivatives
           dz_dx_old = dz_dx.copy()
           d2z_dxx_old = d2z_dxx.copy()
           d3z_dxxx_old = d3z_dxxx.copy()
    
           # Update first-order derivatives
           dz_dx = sigma_prime[:, None] * dz_dx
           
           # Update second-order derivatives
           term1 = sigma_double_prime[:, None, None] * jnp.einsum('ni,nj->nij', dz_dx_old, dz_dx_old)
           term2 = sigma_prime[:, None, None] * d2z_dxx
           d2z_dxx = term1 + term2
           
           # Update third-order derivatives
# term1: sigma_triple_prime * dz_dx_old_i * dz_dx_old_j * dz_dx_old_k
           term1 = sigma_triple_prime[:, None, None, None] * jnp.einsum('ni,nj,nk->nijk', dz_dx_old, dz_dx_old, dz_dx_old)
         
         # term2: sigma_double_prime * (dz_dx_old_i * d2z_dxx_old_jk + dz_dx_old_j * d2z_dxx_old_ik + dz_dx_old_k * d2z_dxx_old_ij)
           term2_a = jnp.einsum('ni,njk->nijk', dz_dx_old, d2z_dxx_old)
           term2_b = jnp.einsum('nj,nik->nijk', dz_dx_old, d2z_dxx_old)
           term2_c = jnp.einsum('nk,nij->nijk', dz_dx_old, d2z_dxx_old)
           term2 = sigma_double_prime[:, None, None, None] * (term2_a + term2_b + term2_c)
         
         # term3: sigma_prime * d3z_dxxx_old
           term3 = sigma_prime[:, None, None, None] * d3z_dxxx_old
         
         # Update d3z_dxxx
           d3z_dxxx = term1 + term2 + term3

        # Update fourth-order derivatives
       
        # Update z after activation
           z = activation(z)
    
       # Final linear layer without activation
       final_w, final_b = params[-1]
       z = jnp.dot(final_w, z) + final_b
       dz_dx = jnp.dot(final_w, dz_dx)
       d2z_dxx = jnp.einsum('ij,jkl->ikl', final_w, d2z_dxx)
       d3z_dxxx = jnp.einsum('ij,jklm->iklm', final_w, d3z_dxxx)
      
    
       return z, dz_dx, d2z_dxx, d3z_dxxx


from models import *

def strain_rate(params, x):
    z, dz_dx, d2z_dxx, d3z_dxxx = derivative_propagation3(params, x)
    
    # Extract third derivatives for each scalar function v_i(x)
    H_v3 = [jnp.squeeze(matrix, axis=0) for matrix in jnp.split(d3z_dxxx, d3z_dxxx.shape[0], axis=0)]
    
    # Compute ∂H_ab/∂x^e
    H_ab_e = jnp.zeros((4, 4, 4))  # Shape: (a, b, e)
    
    for i in range(21):
     if i not in [0,1,2,6,7,11]: 
        selector = selectors[i]  # Shape: (3, 3, 3, 3)

        H_ab_e += w*jnp.einsum('acbd,cde->abe', selector, H_v3[i])
    
        
    D=H_ab_e[1:,0,1:]
    
    
    
    H_v = [jnp.squeeze(matrix, axis=0) for matrix in jnp.split(d2z_dxx, d2z_dxx.shape[0], axis=0)]
  
    
    # Compute H_ab using the selectors
    H_ab = jnp.zeros((4, 4))
    for i in range(21):
       if i not in [0,1,2,6,7,11]: 
        
        
        
        term=w*jnp.einsum('acbd,cd->ab', selectors[i], H_v[i])
        H_ab +=term 
 
    
    
    return D+D.T ,H_ab # Shape: (3, 3, 3)


D2=strain_rate(params, x)[0]


Re=1
nu=1/Re

def stress_loss(params, x):
    

    
    D,T=strain_rate(params, x)
    
    T_t=T[1:,1:]
    u = T[0, 1] 
    v = T[0, 2]  
    w=  T[0, 3] 
    m=u**2+v**2+w**2
    p=(1/3)*(T_t.trace()-m)
    
    
    vel=jnp.array([u,v,w])
    Sigma=T_t-jnp.outer(vel,vel)
    Sigma_t=p*jnp.eye(3)-nu*D
    
   
    
  
    T_t=T[1:,1:]
 


    res=Sigma-Sigma_t
    return res




from domains import *





box = [(0, 1.),(-1, 1.), (-1, 1.), (-1, 1.)]
space = [(-1, 1.), (-1, 1.), (-1, 1.)]
interior = Hyperrectangle(box)
rectangle_boundary = HyperrectangleParabolicBoundary(intervals=box)
space=Hyperrectangle(space)
rectangle_i = HyperrectangleInitial(intervals=box)
# collocation points
x_Omega = interior.random_integration_points(random.PRNGKey(0), N=2601)
x_eval0 = space.random_integration_points(random.PRNGKey(0), N=31*31*10)
time_steps = jnp.arange(0, 1, 0.25)  # [0, 0.25, 0.5, 0.75, 1.0]




# Function to combine time as the first dimension with spatial points
def append_time_to_points(spatial_points, time):
    time_array = jnp.full((spatial_points.shape[0], 1), time)
    return jnp.hstack((time_array, spatial_points))

# Create a list of arrays, each corresponding to a different time step
x_evalt = [append_time_to_points(x_eval0, t) for t in time_steps]


key = random.PRNGKey(seed)
random_time = random.uniform(key, (31*31*6, 1))


x_Gamma = jnp.hstack((random_time, space.random_integration_points(random.PRNGKey(0), N=31*31*6)))


x_Gammai = rectangle_i.random_integration_points(random.PRNGKey(0), N=31*31*10)
x_Gamma=jnp.append(x_Gamma,x_Gammai,axis=0)




a = 1
d = 1
Re = 1

@jit
def u_star(xyz):
    x = xyz[1]
    y = xyz[2]
    z = xyz[3]
    t = xyz[0]

    u = (-a
         * (jnp.exp(a * x) * jnp.sin(a * y + d * z)
            + jnp.exp(a * z) * jnp.cos(a * x + d * y))
         * jnp.exp(-(d ** 2) * t))
    
    v = (-a
         * (jnp.exp(a * y) * jnp.sin(a * z + d * x)
            + jnp.exp(a * x) * jnp.cos(a * y + d * z))
         * jnp.exp(-(d ** 2) * t))
    
    w = (-a
         * (jnp.exp(a * z) * jnp.sin(a * x + d * y)
            + jnp.exp(a * y) * jnp.cos(a * z + d * x))
         * jnp.exp(-(d ** 2) * t))

    return jnp.array([u, v, w])

@jit
def p_star(xyz):
    x = xyz[1]
    y = xyz[2]
    z = xyz[3]
    t = xyz[0]


    p = (-0.5
         * a ** 2
         * (jnp.exp(2 * a * x) + jnp.exp(2 * a * y) + jnp.exp(2 * a * z)
            + 2 * jnp.sin(a * x + d * y) * jnp.cos(a * z + d * x) * jnp.exp(a * (y + z))
            + 2 * jnp.sin(a * y + d * z) * jnp.cos(a * x + d * y) * jnp.exp(a * (z + x))
            + 2 * jnp.sin(a * z + d * x) * jnp.cos(a * y + d * z) * jnp.exp(a * (x + y)))
         * jnp.exp(-2 * d ** 2 * t))

    return p
@jit
def boundary(xyz):
    x = xyz[1]
    y = xyz[2]
    z = xyz[3]
    t = xyz[0]
    
    u,v,w=u_star(xyz)
    p=p_star(xyz)
    return jnp.array([u, v,w,p])

@jit
def boundary_res(params, x):
    return model_uvwp(params, x) - boundary(x)


@jax.jit
def compute_tensor_T(xyz):
    
    x = xyz[1]
    y = xyz[2]
    z = xyz[3]
    t = xyz[0]
    
    
    u = u_star(xyz)  # Velocity vector
    p = p_star(xyz)  # Pressure scalar

    # Identity tensor
    I = jnp.eye(3)

    # Compute gradient of u with respect to spatial variables (x, y, z)
    grad_u = jacrev(u_star, argnums=0)(xyz)[:,1:]  # Exclude time derivative

    # Symmetric part of the velocity gradient tensor
    D = grad_u + grad_u.T

    # Dyadic product of u
    u_dyadic_u = jnp.outer(u, u)

    # Final tensor T
    T = u_dyadic_u + p * I - nu * D

    return T





v_boundary_res_u = vmap(boundary_res, (None, 0))


vmap(model_uvwp, (None, 0))(params, x_Gamma)




@jit
def loss(params):
    
    
    boundary=5. * 0.5 * jnp.mean(v_boundary_res_u(params, x_Gamma)**2)
    sl=0.5 *jnp.mean((vmap(lambda x: stress_loss(params, x))(x_Omega))**2)

    
    
    return boundary+sl



#%%
@jit
def evaluation(params):
    
    
    
        loss_value = loss(params)
        
        # Velocity error computation
        u_pred = vmap(model_uvwp, (None, 0))(params, x_evalt[-1])
        u_exact = vmap(boundary)(x_evalt[-1])
        u_error = jnp.sqrt(jnp.sum((u_pred - u_exact) ** 2) / jnp.sum(u_exact ** 2))

  


        stress_loss = jnp.mean((vmap(lambda x: stress_loss(params, x))(x_evalt[-1])) ** 2)
        
        return loss_value,u_error,stress_loss
#%%

from jax import jit
from jaxopt import LBFGS

# Define the loss function and all the model functions as you have previously.

# JIT-compiled LBFGS update function
@jit
def lbfgs_step(params, state):
    return lbfgs_solver.update(params, state)

# Initialize L-BFGS solver
lbfgs_solver = LBFGS(fun=loss, maxiter=500, tol=1e-5)
state = lbfgs_solver.init_state(params)
#%%


import os
import pickle
from datetime import datetime

# Ensure the "runs" directory exists
os.makedirs("runs", exist_ok=True)

# Initialize arrays to store logs
iterations = []
relative_l2_errors = []
losses = []
simulation_times = []

# Example seed for file naming
seed = args.seed

os.makedirs("runs/RTNN", exist_ok=True)
file_name = f"runs/RTNN/RTNN_seed_{seed}.pkl"

# Training loop
n_epochs=100000
for epoch in range(n_epochs):
    # Perform a single optimization step
    params, state = lbfgs_step(params, state)

    if epoch % 500 == 0:
        loss_value = state.value
        final_loss, u_error, sl = evaluation(params)

        # Log data
        iterations.append(epoch)
        relative_l2_errors.append(float(u_error))
        losses.append(float(loss_value))
        simulation_times.append(float(epoch * 1e-3)) 

        # Optionally print progress
        print(f"Epoch {epoch}, Loss: {loss_value}, Error: {u_error}")

# Prepare a dictionary to save all data
results = {
    "seed": seed,
    "iterations": iterations,
    "relative_l2_errors": relative_l2_errors,
    "losses": losses,
    "simulation_times": simulation_times,
    "params": params
}

# Save everything to a single pickle file
with open(file_name, 'wb') as f:
    pickle.dump(results, f)

print(f"All results saved to {file_name}")
