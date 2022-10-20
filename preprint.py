import numpy as np
from scipy.linalg import cossin, eig

from qiskit.circuit import ParameterVector
from qiskit import QuantumCircuit


# Cartan template
def CART(n: int) -> tuple:
    qc = QuantumCircuit(n)
    ct = _cartan_network(n)
    L = ct.shape[1]
    params = ParameterVector('theta', length=4*L)
    for l,(c,t) in enumerate(zip(ct[0], ct[1])):
        p = list(range(4*(l+1)))[-4:]
        qc.cx(c,t)
        qc.ry(params[p[0]], t)
        qc.rz(params[p[1]], t)
        qc.ry(params[p[2]], c)
        qc.rx(params[p[3]], c)
    return (qc, params)


# Spin template
def SPIN(n: int, L: int) -> tuple:
    qc = QuantumCircuit(n)
    ct = _spin_network(n, L)
    params = ParameterVector('theta', length=4*L)
    for l,(c,t) in enumerate(zip(ct[0], ct[1])):
        p = list(range(4*(l+1)))[-4:]
        qc.cx(c,t)
        qc.ry(params[p[0]], t)
        qc.rz(params[p[1]], t)
        qc.ry(params[p[2]], c)
        qc.rx(params[p[3]], c)
    return (qc, params)


# Sequential template
def SEQU(n: int, L: int, topology: str) -> tuple:
    qc = QuantumCircuit(n)
    links = _get_connectivity(n, connectivity=topology)
    ct = _sequential_network(n, links, L)
    params = ParameterVector('theta', length=4*L)
    for l,(c,t) in enumerate(zip(ct[0], ct[1])):
        p = list(range(4*(l+1)))[-4:]
        qc.cx(c,t)
        qc.ry(params[p[0]], t)
        qc.rz(params[p[1]], t)
        qc.ry(params[p[2]], c)
        qc.rx(params[p[3]], c)
    return (qc, params)


# Cartian structure with angles initialized from QSD
from qiskit.circuit.library.standard_gates import CRYGate, CRZGate
from qiskit.quantum_info import OneQubitEulerDecomposer
from qiskit.quantum_info.synthesis.two_qubit_decompose import TwoQubitWeylDecomposition
from qiskit.compiler import transpile
def QSD(U, do_transpile=False) -> QuantumCircuit:
 
    #FIXME: angles 
    n = int(np.log2(len(U)));
    d = len(U)/2

    if n==1:
        qc = QuantumCircuit(n)
        theta, phi, lam = OneQubitEulerDecomposer(basis='ZYZ').angles(unitary=U)
        qc.rz(phi,range(n))
        qc.ry(theta,range(n))
        qc.rz(lam,range(n))
        return qc
    if n==2:
        qc = QuantumCircuit(n)
        qc.append(TwoQubitWeylDecomposition(U).circuit(euler_basis='ZYZ', simplify=False), range(n))
        return qc

    # Fig. 5  https://www.lri.fr/~baboulin/CPC_Householder.pdf
    (a1,a2),ry_angles,(b1h,b2h) = cossin(U, p=len(U)/2, q=d,  separate=True)
    U2, U1, rz1_angles  = demultiplex(b1h.conjugate().transpose(), b2h.conjugate().transpose())
    U4, U3, rz2_angles = demultiplex(a1, a2)
    
    qc = QuantumCircuit(n)
    unitaries = list(range(n))[-2:]
    multiplexrs = list(reversed(range(n)))
   
    qc.append(QSD(U1), qargs=unitaries)
    qc.append(CRZGate(rz1_angles[1]).control(n-2), qargs=multiplexrs)
    qc.append(QSD(U2), qargs=unitaries)

    qc.append(CRYGate(ry_angles[1]).control(n-2), qargs=multiplexrs)

    qc.append(QSD(U3), qargs=unitaries)
    qc.append(CRZGate(rz2_angles[1]).control(n-2), qargs=multiplexrs)
    qc.append(QSD(U4), qargs=unitaries)

    if do_transpile:
        qc = transpile(qc, 
                    basis_gates=['cx','rx','ry','u1'],
                    optimization_level=3, 
                    seed_transpiler=1
                    )
    return qc

def demultiplex(u1, u2):
    # NOTE: right and left eigenvectors correspond to first and second gates 
    D,V,W = eig(np.matmul(u1, u2.conjugate().transpose()), 
            left=True,
            right=True)
    angles = -2*1j*np.log(D) # entires in D have form exp(j*theta/2)?
    # print(angles)
    return V, W, angles.real 


# primitives adapted from:
# https://qiskit.org/documentation/_modules/qiskit/transpiler/synthesis/aqc/cnot_structures.html#make_cnot_network
def _get_connectivity(num_qubits: int, connectivity: str) -> dict:
    """
        connectivity: type of connectivity structure, ``{"full", "line", "star"}``.
    """
    if num_qubits == 1:
        links = {0: [0]}
    elif connectivity == "full":
        # Full connectivity between qubits.
        links = {i: list(range(num_qubits)) for i in range(num_qubits)}
    elif connectivity == "line":
        # Every qubit is connected to its immediate neighbours only.
        links = {i: [i - 1, i, i + 1] for i in range(1, num_qubits - 1)}
        links[0] = [0, 1]
        links[num_qubits - 1] = [num_qubits - 2, num_qubits - 1]
    elif connectivity == "star":
        # Every qubit is connected to the first one only.
        links = {i: [0, i] for i in range(1, num_qubits)}
        links[0] = list(range(num_qubits))
    return links


def _sequential_network(num_qubits: int, links: dict, depth: int) -> np.ndarray:
    layer = 0
    cnots = np.zeros((2, depth), dtype=int)
    while True:
        for i in range(0, num_qubits - 1):
            for j in range(i + 1, num_qubits):
                if j in links[i]:
                    cnots[0, layer] = i
                    cnots[1, layer] = j
                    layer += 1
                    if layer >= depth:
                        return cnots


def _spin_network(num_qubits: int, depth: int) -> np.ndarray:
    layer = 0
    cnots = np.zeros((2, depth), dtype=int)
    while True:
        for i in range(0, num_qubits - 1, 2):
            cnots[0, layer] = i
            cnots[1, layer] = i + 1
            layer += 1
            if layer >= depth:
                return cnots

        for i in range(1, num_qubits - 1, 2):
            cnots[0, layer] = i
            cnots[1, layer] = i + 1
            layer += 1
            if layer >= depth:
                return cnots


def _cartan_network(num_qubits: int) -> np.ndarray:
    n = num_qubits
    if n > 3:
        cnots = np.array([[0, 0, 0], [1, 1, 1]])
        mult = np.array([[n - 2, n - 3, n - 2, n - 3], [n - 1, n - 1, n - 1, n - 1]])
        for _ in range(n - 2):
            cnots = np.hstack((np.tile(np.hstack((cnots, mult)), 3), cnots))
            mult[0, -1] -= 1
            mult = np.tile(mult, 2)
    elif n == 3:
        cnots = np.array(
            [
                [0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0],
                [1, 1, 1, 2, 2, 2, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 2, 2, 2, 1, 1, 1],
            ]
        )
    return cnots


def _cyclic_spin_network(num_qubits: int, depth: int) -> np.ndarray:
    cnots = np.zeros((2, depth), dtype=int)
    z = 0
    while True:
        for i in range(0, num_qubits, 2):
            if i + 1 <= num_qubits - 1:
                cnots[0, z] = i
                cnots[1, z] = i + 1
                z += 1
            if z >= depth:
                return cnots

        for i in range(1, num_qubits, 2):
            if i + 1 <= num_qubits - 1:
                cnots[0, z] = i
                cnots[1, z] = i + 1
                z += 1
            elif i == num_qubits - 1:
                cnots[0, z] = i
                cnots[1, z] = 0
                z += 1
            if z >= depth:
                return cnots


def _cyclic_line_network(num_qubits: int, depth: int) -> np.ndarray:
    cnots = np.zeros((2, depth), dtype=int)
    for i in range(depth):
        cnots[0, i] = (i + 0) % num_qubits
        cnots[1, i] = (i + 1) % num_qubits
    return cnots
