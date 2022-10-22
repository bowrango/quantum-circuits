import numpy as np
from scipy.stats import unitary_group

from qiskit.converters import circuit_to_dag, dag_to_circuit

from qiskit.quantum_info import Operator

from qiskit.transpiler import PassManager
from qiskit.visualization import dag_drawer

from templates import AQCP

n = int(3)                            # number of qubits 
V = np.identity(2**n, dtype=complex)  # target unitary 

# circuit, params = CART(n)
# angles = np.random.uniform(0, 2*np.pi, size=len(params))
# circuit.assign_parameters({params: angles}, inplace=True)
# W = Operator(circuit).data

aqcp = AQCP(U=V, template="cart", depth=int(12), connectity="star")
