
# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import numpy as np

#TODO: chimera?
def get_connectivity(num_qubits: int, connectivity: str) -> dict:
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


def sequential_network(num_qubits: int, links: dict, depth: int) -> np.ndarray:
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


def spin_network(num_qubits: int, depth: int) -> np.ndarray:
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


def cartan_network(num_qubits: int) -> np.ndarray:
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


def cyclic_spin_network(num_qubits: int, depth: int) -> np.ndarray:
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


def cyclic_line_network(num_qubits: int, depth: int) -> np.ndarray:
    cnots = np.zeros((2, depth), dtype=int)
    for i in range(depth):
        cnots[0, i] = (i + 0) % num_qubits
        cnots[1, i] = (i + 1) % num_qubits
    return cnots
