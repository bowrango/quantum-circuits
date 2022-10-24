import numpy as np
from scipy.stats import unitary_group

from qiskit.converters import circuit_to_dag, dag_to_circuit

from qiskit.transpiler import PassManager
from qiskit.visualization import dag_drawer

n = int(3)                            # number of qubits 
V = np.identity(2**n, dtype=complex)  # target unitary 
# V = unitary_group.rvs(2**n)

from aqc import AQC

aqc = AQC(target=V, template="sequ", depth=12, connectity="line")
aqc.compile()

# dag_drawer(circuit_to_dag(aqc.ansatz))


