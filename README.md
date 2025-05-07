License: CC-BY-4.0
#  Riemann Tensor Neural Networks: Learning Conservative Systems with Physics-Constrained Networks

## Abstract

Divergence-free symmetric tensors (DFSTs) are fundamental in continuum mechanics, encoding conservation laws such as mass and momentum conservation. We introduce Riemann Tensor Neural Networks (RTNNs), a novel neural architecture that inherently satisfies the DFST condition to machine precision, providing a strong inductive bias for enforcing these conservation laws. We prove that RTNNs can approximate any sufficiently smooth DFST with arbitrary precision and demonstrate their effectiveness as surrogates for conservative PDEs, achieving improved accuracy across benchmarks. This work is the first to use DFSTs as an inductive bias in neural PDE surrogates and to explicitly enforce the conservation of both mass and momentum within a physics-constrained neural architecture.



We provide a demo for the Beltrami flow, in the file main_beltrami.py , more example will be provided with the Camera-ready version of the paper




## Requirements

- Python 3.10.10 or later
- JAX 0.4.8 or later
- JAXopt 0.6 or later
- Optax 0.1.4 or later
- jaxtyping 0.2.25 or later
- Lineax
- jmp
- equinox


