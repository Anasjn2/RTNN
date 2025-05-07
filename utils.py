# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 07:11:26 2024


"""
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 06:53:09 2024


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

    
    
# Alternatively, if you want to ensure linear independence, you can perform
# a more sophisticated selection or use linear algebra techniques.

# Assign random scalars to each basis tensor (if needed)
# For demonstration, we'll create a random linear combination
# key = jax.random.PRNGKey(0)
# random_scalars = jax.random.normal(key, shape=(21,))

# # Construct the general Riemann-like tensor as a linear combination of the basis tensors
# def construct_riemann_like_tensor(random_scalars, basis_tensors):
#     """
#     Construct a Riemann-like (0,4)-tensor as a linear combination of basis tensors.
    
#     Parameters:
#     - random_scalars: A 1D array of 20 random scalars.
#     - basis_tensors: A list of 20 (4,4,4,4) basis tensors.
    
#     Returns:
#     - A (4,4,4,4) Riemann-like tensor.
#     """
#     tensor = jnp.zeros((n, n, n, n))
#     for scalar, basis in zip(random_scalars, basis_tensors):
#         tensor += scalar * basis
#     return tensor

# # Example usage: Construct a Riemann-like tensor
# riemann_like_tensor = construct_riemann_like_tensor(random_scalars, basis_tensors)

# # Function to verify the symmetries of the Riemann-like tensor
# def verify_symmetries(T):
#     """
#     Verify the symmetries of a Riemann-like (0,4)-tensor.
    
#     Parameters:
#     - T: A (4,4,4,4) tensor.
    
#     Returns:
#     - A dictionary indicating whether each symmetry is satisfied.
#     """
#     symmetries = {}
#     # Antisymmetry in the first two indices
#     symmetries['antisym_first_two'] = jnp.allclose(T, -T.transpose(1,0,2,3))
    
#     # Antisymmetry in the last two indices
#     symmetries['antisym_last_two'] = jnp.allclose(T, -T.transpose(0,1,3,2))
    
#     # Symmetry under exchange of index pairs
#     symmetries['sym_exchange_pairs'] = jnp.allclose(T, T.transpose(2,3,0,1))
    

    
#     return symmetries

# # Verify the symmetries of the constructed tensor
# symmetry_checks = verify_symmetries(riemann_like_tensor)
# print("Symmetry Verification:")
# for symmetry, is_satisfied in symmetry_checks.items():
#     print(f"{symmetry}: {'Satisfied' if is_satisfied else 'Not Satisfied'}")

# # If you want to inspect the basis tensors, you can access them via the `basis_tensors` list
# # For example, to print the first basis tensor:
# # print(basis_tensors[0])

