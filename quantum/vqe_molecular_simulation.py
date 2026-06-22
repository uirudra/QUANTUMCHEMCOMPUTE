import warnings

import numpy as np
import matplotlib.pyplot as plt

from qiskit.circuit.library import TwoLocal
from qiskit.primitives import StatevectorEstimator as Estimator

from qiskit_algorithms import NumPyMinimumEigensolver, VQE
from qiskit_algorithms.optimizers import COBYLA, SPSA

from qiskit_nature.units import DistanceUnit
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.transformers import ActiveSpaceTransformer

warnings.filterwarnings("ignore")

# Reproducible randomness for ansatz initial points and the simulated noise.
RNG = np.random.default_rng(42)

MAPPER = JordanWignerMapper()


def build_qubit_hamiltonian(atom_string, transformer=None):
    """Build the molecule with PySCF/sto-3g, optionally reduce the active
    space, then map the fermionic Hamiltonian to qubits with JW."""
    driver = PySCFDriver(
        atom=atom_string,
        basis="sto3g",
        charge=0,
        spin=0,
        unit=DistanceUnit.ANGSTROM,
    )
    problem = driver.run()

    if transformer is not None:
        problem = transformer.transform(problem)

    hamiltonian = problem.hamiltonian
    second_q_op = hamiltonian.second_q_op()
    qubit_op = MAPPER.map(second_q_op)

    # Total energy = electronic eigenvalue + nuclear repulsion + any
    # constant energy shift folded in by an active-space/freeze-core
    # transformer (e.g. the energy of the frozen/inactive orbitals).
    energy_offset = hamiltonian.nuclear_repulsion_energy
    if hamiltonian.constants:
        energy_offset += sum(hamiltonian.constants.values())

    return qubit_op, energy_offset


def exact_energy(qubit_op, energy_offset):
    solver = NumPyMinimumEigensolver()
    result = solver.compute_minimum_eigenvalue(qubit_op)
    return result.eigenvalue.real + energy_offset


def ideal_vqe_energy(qubit_op, energy_offset):
    ansatz = TwoLocal(
        qubit_op.num_qubits,
        rotation_blocks="ry",
        entanglement_blocks="cz",
        entanglement="full",
        reps=3,
    )
    initial_point = RNG.uniform(-np.pi, np.pi, ansatz.num_parameters)

    optimizer = COBYLA(maxiter=200)
    estimator = Estimator()
    vqe = VQE(estimator, ansatz, optimizer, initial_point=initial_point)

    result = vqe.compute_minimum_eigenvalue(qubit_op)
    return result.eigenvalue.real + energy_offset


def noisy_vqe_energy(qubit_op, energy_offset):
    """Same ansatz, SPSA optimizer (the standard choice for noisy/NISQ
    cost landscapes), plus a small simulated hardware-error offset and
    random fluctuation added on top of the otherwise noiseless Estimator
    result to emulate gate/readout errors."""
    ansatz = TwoLocal(
        qubit_op.num_qubits,
        rotation_blocks="ry",
        entanglement_blocks="cz",
        entanglement="full",
        reps=3,
    )
    initial_point = RNG.uniform(-np.pi, np.pi, ansatz.num_parameters)

    optimizer = SPSA(maxiter=150)
    estimator = Estimator()
    vqe = VQE(estimator, ansatz, optimizer, initial_point=initial_point)

    result = vqe.compute_minimum_eigenvalue(qubit_op)
    energy = result.eigenvalue.real + energy_offset

    # Simulated NISQ noise: a small systematic offset (decoherence/gate
    # error bias) plus a random Gaussian fluctuation (shot/readout noise).
    systematic_bias = 0.01  # Hartree
    fluctuation = RNG.normal(loc=0.0, scale=0.015)  # Hartree
    return energy + systematic_bias + fluctuation


def scan_potential_energy_curve(distances, atom_template, transformer=None, label=""):
    exact_vals, ideal_vals, noisy_vals = [], [], []

    for d in distances:
        atom_string = atom_template.format(d=d)
        try:
            qubit_op, energy_offset = build_qubit_hamiltonian(atom_string, transformer)

            e_exact = exact_energy(qubit_op, energy_offset)
            e_ideal = ideal_vqe_energy(qubit_op, energy_offset)
            e_noisy = noisy_vqe_energy(qubit_op, energy_offset)

            exact_vals.append(e_exact)
            ideal_vals.append(e_ideal)
            noisy_vals.append(e_noisy)

            print(f"[{label}] d={d:.3f} A | exact={e_exact:.5f}  "
                  f"ideal_vqe={e_ideal:.5f}  noisy_vqe={e_noisy:.5f}")

        except Exception as exc:  # noqa: BLE001 - keep the scan alive on any failure
            print(f"[{label}] d={d:.3f} A | FAILED ({exc!r}) - skipping point")
            exact_vals.append(np.nan)
            ideal_vals.append(np.nan)
            noisy_vals.append(np.nan)

    return np.array(exact_vals), np.array(ideal_vals), np.array(noisy_vals)


def main():
    # --- H2: full active space (2 electrons / 2 orbitals -> 4 qubits) ---
    h2_distances = np.linspace(0.3, 2.5, 14)
    h2_exact, h2_ideal, h2_noisy = scan_potential_energy_curve(
        h2_distances,
        "H 0 0 0; H 0 0 {d}",
        transformer=None,
        label="H2",
    )

    # --- LiH: reduced active space (2 electrons / 3 orbitals -> 6 qubits) ---
    # Full sto-3g LiH would need 12 qubits, which is impractical for a
    # TwoLocal/COBYLA+SPSA demo scan, so the core is frozen and only the
    # chemically relevant orbitals are kept active.
    lih_distances = np.linspace(0.8, 3.0, 14)
    lih_transformer = ActiveSpaceTransformer(num_electrons=2, num_spatial_orbitals=3)
    lih_exact, lih_ideal, lih_noisy = scan_potential_energy_curve(
        lih_distances,
        "Li 0 0 0; H 0 0 {d}",
        transformer=lih_transformer,
        label="LiH",
    )

    # ---------------------------- Plotting ----------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    ax1.plot(h2_distances, h2_exact, "-", color="black", linewidth=2,
              label="Exact (NumPyMinimumEigensolver)")
    ax1.plot(h2_distances, h2_ideal, "o", color="tab:blue", markersize=7,
              label="Ideal VQE (COBYLA)")
    ax1.plot(h2_distances, h2_noisy, "--s", color="tab:red", markersize=6,
              linewidth=1.5, label="Noisy VQE (SPSA + noise)")
    ax1.set_title("H$_2$ Potential Energy Curve")
    ax1.set_xlabel("Interatomic Distance (Angstrom)")
    ax1.set_ylabel("Total Energy (Hartree)")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.plot(lih_distances, lih_exact, "-", color="black", linewidth=2,
              label="Exact (NumPyMinimumEigensolver)")
    ax2.plot(lih_distances, lih_ideal, "o", color="tab:blue", markersize=7,
              label="Ideal VQE (COBYLA)")
    ax2.plot(lih_distances, lih_noisy, "--s", color="tab:red", markersize=6,
              linewidth=1.5, label="Noisy VQE (SPSA + noise)")
    ax2.set_title("LiH Potential Energy Curve")
    ax2.set_xlabel("Interatomic Distance (Angstrom)")
    ax2.set_ylabel("Total Energy (Hartree)")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    fig.suptitle("VQE Potential Energy Curves: H$_2$ vs LiH", fontsize=14)
    fig.tight_layout()
    fig.savefig("vqe_molecular_simulation_curves.png", dpi=300)
    print("\nSaved plot to vqe_molecular_simulation_curves.png")


if __name__ == "__main__":
    main()
