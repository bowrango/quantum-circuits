import numpy as np
from scipy.linalg import cossin, eig
from qiskit import QuantumCircuit

#TODO: make this into a class which AQCP can inherit from?

# Cartian structure with angles initialized from QSD
from qiskit.circuit.library.standard_gates import CRYGate, CRZGate
from qiskit.quantum_info import OneQubitEulerDecomposer
from qiskit.quantum_info.synthesis.two_qubit_decompose import TwoQubitWeylDecomposition
from qiskit.compiler import transpile
def QSD(U, cleanup=False) -> QuantumCircuit:

    #FIXME: angles; also I think the paper used uniformlly-controlled gates unlike Qiskit  

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

    if cleanup:
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
    return V, W, angles.real 