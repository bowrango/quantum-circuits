import numpy as np

from qiskit.circuit import ParameterVector, QuantumCircuit
from qiskit.opflow import NaturalGradient, Gradient, QFI, Hessian, CircuitStateFn, StateFn, X, Z

# https://qiskit.org/documentation/stable/0.24/tutorials/operators/02_gradients_framework.html

# Instantiate the quantum state
theta = ParameterVector('theta', length=2)
qc = QuantumCircuit(1)
qc.h(0)
qc.rz(theta[0], 0)
qc.rx(theta[1], 0)

value_dict = {theta: [np.pi, np.pi]}

# Instantiate the Hamiltonian observable
# coeffs = ParameterVector('coeffs', length=2)
# H = (coeffs[0]*coeffs[0]*2)*X + coeffs[1] * Z      # parameterized
H = 0.5 * X - 1 * Z                                # fixed

# Combine the Hamiltonian observable and the state
op = ~StateFn(H) @ CircuitStateFn(primitive=qc, coeff=1.)


# The Quantum Fisher Information is a metric tensor to represent
# the capacity of a parameterized quantum state
qfi = QFI(qfi_method='lin_comb_full').convert(
      operator=CircuitStateFn(primitive=qc), params=theta)
qfi.assign_parameters(value_dict).eval()

# === 1st-Order Gradients w.r.t. State Parameters ===

param_grad = Gradient(grad_method='param_shift').convert(
   operator=op, params=theta)
param_grad.assign_parameters(value_dict).eval()


fin_grad = Gradient(grad_method='fin_diff').convert(
   operator=CircuitStateFn(primitive=qc), params=theta)
fin_grad.assign_parameters(value_dict).eval()


nat_grad = NaturalGradient(grad_method='param_shift',
                           qfi_method='overlap_diag',
                           regularization=None).convert(operator=op, params=theta)
nat_grad.assign_parameters(value_dict).eval() 

# === 2nd-Order Gradients w.r.t. State Parameters ===

hess_grad = Hessian(hess_method='param_shift').convert(
      operator=op, params=theta)
hess_grad.assign_parameters(value_dict).eval()