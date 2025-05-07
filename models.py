# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 07:23:38 2024


"""



import jax
import jax.numpy as jnp
from jax import random
jax.config.update("jax_enable_x64", True)

# Set the random seed for JAX

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



activation = tanh
activation_prime = tanh_prime
activation_double_prime = tanh_double_prime





def random_layer_params(m, n, key, scale=1e-1):
    w_key, b_key = random.split(key)
    return scale * random.normal(w_key, (n, m)), scale * random.normal(b_key, (n,))

# Initialize network parameters
def init_params(sizes, key):
    keys = random.split(key, len(sizes))
    return [random_layer_params(m, n, k) for m, n, k in zip(sizes[:-1], sizes[1:], keys)]
# Initialize model parameters

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

def tanh_triple_prime(x):
    tanh_x = jnp.tanh(x)
    return -2 * (1 - tanh_x**2) * (1 - 3 * tanh_x**2)



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

















#%%
def apply_periodic_embedding(x, periodic_dims, periods):
    """
    Apply sine and cosine periodic embeddings to specific dimensions of the input.
    
    Args:
    x: Input data, expected shape (batch_size, features).
    periodic_dims: List of dimension indices to which to apply the embedding.
    periods: List of periods corresponding to the periodic dimensions.
    
    Returns:
    Augmented feature matrix.
    """

    if len(periodic_dims) != len(periods):
        raise ValueError("Each periodic dimension must have a corresponding period value.")

    original_features = [x[i] for i in range(x.shape[0]) if i not in periodic_dims]
    periodic_features = [x[i] for i in periodic_dims]

    sin_cos_features = []
    for idx, feature in enumerate(periodic_features):
        frequency = 2 * jnp.pi / periods[idx]
        sin_feature = jnp.sin(feature * frequency)
        cos_feature = jnp.cos(feature * frequency)
        sin_cos_features.extend([sin_feature, cos_feature])

    augmented_x = jnp.reshape(jnp.array([original_features+sin_cos_features]),(-1,))
    
    return augmented_x

def derivative_propagation3p(params, x, periodic_dims, periods):
    augmented_input = apply_periodic_embedding(x, periodic_dims, periods)
    num_augmented_features = augmented_input.shape[0]

    # Initialize derivatives
    dz_dx = jnp.zeros((num_augmented_features, x.shape[0]))
    d2z_dxx = jnp.zeros((num_augmented_features, x.shape[0], x.shape[0]))
    d3z_dxxx = jnp.zeros((num_augmented_features, x.shape[0], x.shape[0], x.shape[0]))


    # Fill derivatives for original features
    orig_indices = [i for i in range(x.shape[0]) if i not in periodic_dims]
    for i, orig_idx in enumerate(orig_indices):
        dz_dx = dz_dx.at[i, orig_idx].set(1)

    # Derivatives for sin and cos features
    for i, dim in enumerate(periodic_dims):
        period = periods[i]
        frequency = 2 * jnp.pi / period
        sin_idx = len(orig_indices) + 2 * i
        cos_idx = sin_idx + 1
        sin_val = jnp.sin(x[dim] * frequency)
        cos_val = jnp.cos(x[dim] * frequency)
        dz_dx = dz_dx.at[sin_idx, dim].set(frequency * cos_val)
        dz_dx = dz_dx.at[cos_idx, dim].set(-frequency * sin_val)

        # Second-order derivatives
        d2z_dxx = d2z_dxx.at[sin_idx, dim, dim].set(-frequency**2 * sin_val)
        d2z_dxx = d2z_dxx.at[cos_idx, dim, dim].set(-frequency**2 * cos_val)

        # Third-order derivatives
        d3z_dxxx = d3z_dxxx.at[sin_idx, dim, dim, dim].set(-frequency**3 * cos_val)
        d3z_dxxx = d3z_dxxx.at[cos_idx, dim, dim, dim].set(-frequency**3 * sin_val)


    # Propagate through the network
    z = augmented_input
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
