# QUANTUMCHEMCOMPUTE

# VQE Molecular Simulation

This repository contains a Python script that uses IBM's **Qiskit** framework to simulate the potential energy curves of simple molecules ($H_2$ and $LiH$) using the **Variational Quantum Eigensolver (VQE)** algorithm. 

The script compares three different methods of computing the total molecular energy across varying interatomic distances:
1. **Exact Classical Diagonalization**: Calculated using `NumPyMinimumEigensolver`.
2. **Ideal VQE**: Statevector simulation using the `COBYLA` optimizer.
3. **Noisy VQE**: Simulation using the `SPSA` optimizer with an added artificial noise profile (systematic bias and Gaussian fluctuations) to emulate the behavior of Noisy Intermediate-Scale Quantum (NISQ) hardware.

---

## Features

- **Automated Hamiltonian Mapping**: Builds molecular Hamiltonians via `PySCFDriver` using the `sto-3g` basis set and maps fermionic operators to qubits using the **Jordan-Wigner Transformation**.
- **Active Space Reduction**: Dynamically reduces the active orbital space for Lithium Hydride ($LiH$) from 12 qubits to 6 qubits using the `ActiveSpaceTransformer` to keep simulation times efficient.
- **Data Visualization**: Generates a side-by-side comparative plot of the potential energy curves for both molecules and saves it as a high-resolution PNG image.

---

## Requirements & Installation

Make sure you have Python 3.8+ installed. You will need to install Qiskit, Qiskit Nature, Qiskit Algorithms, PySCF, and standard data science libraries.

You can install all dependencies via `pip`:

```bash
pip install qiskit qiskit-algorithms qiskit-nature pyscf numpy matplotlib
