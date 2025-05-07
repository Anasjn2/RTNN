# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 07:11:26 2024
"""



import warnings
import jax
import jax.numpy as jnp
from jax import jacfwd, jacrev

jax.config.update("jax_enable_x64", True)

warnings.filterwarnings("ignore")
import jax
import jax.numpy as jnp
from itertools import combinations_with_replacement

# Define the dimension
n = 4

# Initialize the basis 2-forms
def create_basis_2forms():
    """
    Create the six basis 2-forms in four dimensions.
    Each 2-form is represented as a (4,4) antisymmetric matrix.
    """
    basis_2forms = []
    for i, j in combinations_with_replacement(range(n), 2):
        if i < j:
            omega = jnp.zeros((n, n))
            omega = omega.at[i, j].set(1)
            omega = omega.at[j, i].set(-1)
            basis_2forms.append(omega)
    return basis_2forms

# Generate the six basis 2-forms
omega = create_basis_2forms()
# omega[0] = e1 ∧ e2, omega[1] = e1 ∧ e3, ..., omega[5] = e3 ∧ e4

# Function to compute the tensor product of two 2-forms
def tensor_product(omega_i, omega_j):
    """
    Compute the tensor product of two 2-forms omega_i and omega_j.
    Returns a (4,4,4,4) tensor.
    """
    return jnp.einsum('ab,cd->abcd', omega_i, omega_j)

# Generate all symmetric tensor products (21 in total)
symmetric_products = []
for i in range(len(omega)):
    for j in range(i, len(omega)):
        sym_prod = tensor_product(omega[i], omega[j]) + tensor_product(omega[j], omega[i])
        symmetric_products.append(sym_prod)

# Now, impose the First Bianchi Identity to reduce from 21 to 20 tensors
# For simplicity, we'll exclude the last tensor
basis_tensors = symmetric_products#[]  # Now we have 20 basis tensors



def build_basis(n=3):
    symmetric_products = []
    
    def create_basis_2forms():
        """
        Create the six basis 2-forms in four dimensions.
        Each 2-form is represented as a (4,4) antisymmetric matrix.
        """
        basis_2forms = []
        for i, j in combinations_with_replacement(range(n), 2):
            if i < j:
                omega = jnp.zeros((n, n))
                omega = omega.at[i, j].set(1)
                omega = omega.at[j, i].set(-1)
                basis_2forms.append(omega)
        return basis_2forms

    # Generate the six basis 2-forms
    omega = create_basis_2forms()
    
    for i in range(len(omega)):
        for j in range(i, len(omega)):
            if i < j:
                # Symmetric product: (omega_i ⊗ omega_j + omega_j ⊗ omega_i) / 2
                sym_prod = (tensor_product(omega[i], omega[j]) + tensor_product(omega[j], omega[i])) / 2
            else:
                # i == j: (omega_i ⊗ omega_i) / 2
                sym_prod = tensor_product(omega[i], omega[j]) 
            symmetric_products.append(sym_prod)
    
                
    return symmetric_products

    

