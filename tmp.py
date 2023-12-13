import numpy as np
from qiskit import QuantumCircuit, Aer, transpile
from qiskit_aer import AerSimulator
from qiskit.quantum_info import SparsePauliOp, Statevector
from sklearn.metrics import mean_absolute_error
from qiskit.providers.fake_provider import FakeAthensV2
from qiskit_aer.noise import NoiseModel

theta = 0.36430834792628597
op = SparsePauliOp.from_list([("IIIIZ", 1)])
qc = QuantumCircuit(5)
for j in range(30):
    for i in range(5):
        qc.rx(theta, i)
    for i in range(4):
        qc.cx(i, i + 1)

exact_expval = Statevector(qc).expectation_value(op).real
print("exact", exact_expval)

# ---------
backend = FakeAthensV2()
for k in backend.target.keys():
    for i in backend.target[k].keys():
        if k != 'reset' and k != 'delay':
            backend.target[k][i].error = 0 # Put the noise equal to 0 for each instruction

sim = AerSimulator.from_backend(backend) # Fake backend with 0 noise simulator
sim.set_options(seed_simulator=0)

shots = 10000 # number of shots per circuit
qc = transpile(qc, sim, optimization_level=0)
qc.measure_all()

result = sim.run(qc, shots=shots).result()
key = [list(i) for i in (list(result.get_counts().keys()))] # output bit strings
key = np.array(key)
val = np.array(list(result.get_counts().values())) # output bit strings probabilities
indices_plus = np.where(key[:, -1] == '0')
indices_minus = np.where(key[:, -1] != '0')
expval_fake = np.sum(val[indices_plus]) - np.sum(val[indices_minus])
print("fake", expval_fake/shots)

ideal_expval = []
ideal_shots_expval = []
fake_backend_expval = []


# for samples in range(10): # I calculate the mean absolute error of the expectation value over 10 circuits
#
#     theta = np.random.uniform(0,1)
#
#     qc = QuantumCircuit(5)
#     for j in range(30):
#         for i in range(5):
#             qc.rx(theta,i)
#         for i in range(4):
#             qc.cx(i,i+1)
#
#     qc = transpile(qc, sim, optimization_level=0)
#     exact_expval = Statevector(qc).expectation_value(op).real
#     ideal_expval.append(exact_expval)
#     qc.measure_all()
#
#     # Fake Backend
#     result = sim.run(qc, shots=shot).result()
#     key = [list(i) for i in (list(result.get_counts().keys()))] # output bit strings
#     key = np.array(key)
#     val = np.array(list(result.get_counts().values())) # output bit strings probabilities
#     indices_plus = np.where(key[:, -1] == '0')
#     indices_minus = np.where(key[:, -1] != '0')
#     expval_fake = np.sum(val[indices_plus]) - np.sum(val[indices_minus])
#     fake_backend_expval.append(expval_fake/shot)
#
#     # Ideal Backend with shots
#     result = sim_ideal.run(qc, shots=shot).result()
#     key = [list(i) for i in (list(result.get_counts().keys()))] # output bit strings
#     key = np.array(key)
#     val = np.array(list(result.get_counts().values())) # output bit strings probabilities
#     indices_plus = np.where(key[:, -1] == '0')
#     indices_minus = np.where(key[:, -1] != '0')
#     expval_ideal_shots = np.sum(val[indices_plus]) - np.sum(val[indices_minus])
#     ideal_shots_expval.append(expval_ideal_shots/shot)
#
# print('MAE between exact expectation values and expectation value calculated with ideal backend with 10000 shots:',mean_absolute_error(ideal_expval, ideal_shots_expval))
# print('MAE between exact expectation values and expectation value calculated with fake backend with 0 noise:',mean_absolute_error(ideal_expval, fake_backend_expval))
