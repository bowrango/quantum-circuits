**Abstract**
This is a technical note on the task of compiling unitary matrices for quantum circuit synthesis. I mainly explore the work proposed in [3] which was implemented in Qiskit [5]. What better way to learn something than teach it?

The Approximate Quantum Compiler seeks to best approximate an input unitary matrix as an ordered sequence of gates. As a mathematical framework, gates represent unitary matrices that through successive Kronecker tensor products form the matrix of the circuit. Comment on physical meaning? 

This matrix descibes the transformation of a quantum state as a result of the circuit execution

In this work I introduce my own version of Qiskit's Approximate Quantum Compiler 

```
aqcp = AQC(U=V, template="cart", depth=int(12), connectity="star")
```


**Preface**
For a realistic quantum computation we can not know the unitary matrix that we need to compile, if we did, there would no need for the quantum computer in the first place. However, compiling smaller unitaries into an efficient sequence of gates that obey the hardware connectivity is a important task.

For this task [3] introduces several templates that form the circuit structure ansatz of "CX units", and treat 1q rotation angles as learnable parameters. The compilation routine starts by enforcing sparsity and eliminates zero rotation gates, then compresses the structure via compaction rules, and re-optimizes with the regular gradient descent (not LASSO) on the compacted structure. 

They use it to compress the Quantum Shannon Decomposition, which already produces an exact compilation using twice the number of CX gates as the theoretical lower bound. 

It's really interesting to see a general procedure that considers hardware connectivity and leverages machine learning techniques, instead of layers of heuristic methods.


**References**

[3](https://arxiv.org/abs/2106.05649)

[5](https://qiskit.org/documentation/apidoc/synthesis_aqc.html) 
