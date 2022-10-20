import numpy as np
from scipy.stats import unitary_group

from qiskit.circuit import ParameterVector, Parameter
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.quantum_info import Operator

from qiskit.transpiler.passes import Optimize1qGates
from qiskit.transpiler import PassManager
from qiskit.visualization import dag_drawer

from preprint import QSD, CART, SPIN, SEQU

n = int(3)                            # number of qubits 
V = np.identity(2**n, dtype=complex)  # target unitary 

circuit, params = SEQU(n, L=int(12), topology="line")
angles = np.random.uniform(low=0, high=2*np.pi, size=len(params))
circuit.assign_parameters({params: angles}, inplace=True)

W = Operator(circuit).data
# print(W)
# print(np.isclose(W, V))