
import numpy as np

from qiskit.circuit import ParameterVector, QuantumCircuit
from qiskit.opflow import NaturalGradient, Gradient, QFI, Hessian, CircuitStateFn, StateFn, X, Z, I
 
# The basic procedure of VQE is, given a (short) quantum circuit U(θ) with 
# parameters θ = (θ1, ..., θm), to repeatedly update θ so that the 
# mean energy f(θ) = ⟨φ(θ)|H|φ(θ)⟩ decreases toward its minimum for the ansatz 
# state |φ(θ)⟩ = U(θ)|0⟩ with |0⟩ the initial state

# hamiltonian of H2 molecule can be reduced and modeled using two qubits
h2_hamiltonian = 0.4*((Z ^ I) + (I ^ Z)) + 0.2*(X ^ X)
theta0  = np.array([-0.2, -0.2, 0, 0])

# ansatz circuit U(θ)
params = ParameterVector('theta', length=4)
U = QuantumCircuit(2)
U.ry(2*params[0],0)
U.ry(2*params[1],1)
U.cx(0,1)
U.ry(2*params[2],0)
U.ry(2*params[3],1)

# Fubini-Study metric
F = QFI(qfi_method='overlap_block_diag').convert(
      operator=CircuitStateFn(primitive=U), params=params)
# print(F.to_matrix())

# expectation value corresponding to the energy f(θ)
f = ~StateFn(h2_hamiltonian) @ StateFn(U)

steps = 100
theta = theta0
lrn_rate = 0.05
grad = NaturalGradient(grad_method='param_shift',
                        qfi_method='lin_comb_full',
                        regularization=None).convert(operator=f, params=params)

for _ in range(steps):
    # qfi = F.assign_parameters({params: theta}).eval()
    nat_grad = grad.assign_parameters({params: theta}).eval()
    theta = theta - lrn_rate*nat_grad

    energy = f.assign_parameters({params: theta}).eval()
    print(energy)


    