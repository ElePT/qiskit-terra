
from qiskit._accelerate.reproduce_elenas_issues import *
from qiskit.converters import dag_to_circuit, circuit_to_dag
from qiskit.visualization import dag_drawer

# print("\nONLY 1 GATE")
# dag = create_dag_from_scratch_1_op()
# # print("\nDAG: ", dag)

# # print("dag.qubits out: ", dag.qubits)
# # print("list(dag.topological_op_nodes()) out: ", list(dag.topological_op_nodes()))

# # circuit = dag_to_circuit(dag)
# # print("Circuit: ", circuit)
# circuit = dag_to_circuit(dag)
# print("Circuit data: ", circuit.data, "\n")
# print("Circuit draw: ", circuit)

# print("--------")
# 
# for qubit in dag.qubits:
#     print("QUBIT", qubit)
# for node in dag.topological_op_nodes():
#     print("NODE QARGS", node.qargs)
#     for qubit in node.qargs:
#         print("NODE QARG QUBIT", qubit)

print("--------")
print("3 GATES")
dag = create_dag_from_scratch_3_ops()

for node in dag.topological_op_nodes():
    print("node qarg: ", node.qargs)
# dag_drawer(dag, filename="out.png")

# new_dag = circuit_to_dag(dag_to_circuit(dag))
# dag_drawer(new_dag, filename="out3.png")
# # print("\nDAG: ", dag)

# print("dag.qubits out: ", dag.qubits)
# print("list(dag.topological_op_nodes()) out: ", list(dag.topological_op_nodes()))

# print("dag.qubits out: ", new_dag.qubits)
# print("list(dag.topological_op_nodes()) out: ", list(new_dag.topological_op_nodes()))

# circuit = dag_to_circuit(new_dag)
# print("Circuit data: ", circuit.data, "\n")
# print("Circuit draw: ", circuit)


# print("--------")

# for qubit in dag.qubits:
#     print("QUBIT", qubit)
# for node in dag.topological_op_nodes():
#     print("NODE QARGS", node.qargs)
#     for qubit in node.qargs:
#         print("NODE QARG QUBIT", qubit)