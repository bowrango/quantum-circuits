import numpy as np
from scipy.stats import unitary_group

from qiskit.converters import circuit_to_dag, dag_to_circuit

from qiskit.transpiler import PassManager
from qiskit.visualization import dag_drawer

n = int(3)                            # number of qubits 
V = np.identity(2**n, dtype=complex)  # target unitary 

from aqcp import AQCP

aqc = AQCP(target=V, template="cart", depth=15, connectity="full")
aqc.compile()


