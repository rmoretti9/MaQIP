"""We execute 21 identical qsvms to study site performance"""
import numpy as np
from pathlib import Path
from qiskit_ibm_runtime import Session

from collections import OrderedDict
from typing import Union
from qiskit.circuit import QuantumCircuit
from qiskit.compiler import transpile
from maqip.genetic_brisbanne import to_quantum, get_kernel_entry
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import csv
from qiskit_ibm_provider import IBMProvider

###################################################################
############ LOAD AN ACCOUNT THAT CAN RUN ON BRISBANE #############
###################################################################

IBMProvider.save_account(token='', overwrite=True)
provider = IBMProvider(instance='')
backend = provider.get_backend('ibm_brisbane')
###################################################################
###################################################################
###################################################################

qsvm_connections = [
    [2,1,0,14],
    [4,15,22,21],
    [6,7,8,16],
    [10, 11,12,13],
    [20,33,39,40],
    [17,30,29,28],
    [32,36,51,50],
    [41,53,60,59],
    [43,44,45,46],
    [62,63,64,54],
    [57,58,71,77],
    [85,84,83,82],
    [87,88,89,74],
    [76,75,90,94],
    [100,99,98,97],
    [105,104,103,102],
    [108,107,106,93],
    [114,109,96,95],
    [116,117,118,110],
    [125,124,123,122]
]
gate_dict = OrderedDict(
    [
        ("single_non_parametric", ["Id", "X","SX"]),
        ("single_parametric", ["RZ", "RZ"]),
        ("two_non_parametric", ["ECR"]),
        ("two_parametric", []),
    ]
)

nb_qubits = 4
gates_per_qubits = 8
data_cv, data_labels = np.load("../processed_dataset/data_cv.npy"), np.load("../processed_dataset/data_labels.npy")
useful_ft = [0, 5, 6, 8, 10, 12, 14, 15]
data_cv = data_cv[:, useful_ft]

nb_features = data_cv.shape[1]
nb_cbits = 4*20
cbits = [item for item in range(0, nb_cbits)]
nb_samples = 100
genes = [2,0,1,5,1,2,1,1,2,5,0,1,1,6,3,1,1,0,5,7,3,0,0,3,5,5,2,0,0,5,2,2,0,4,3,5,
         0,1,7,6,3,2,0,1,5,3,2,1,0,5,1,2,0,7,1,3,0,0,4,7,3,2,1,4,7,3,2,1,0,5,3,1,
         0,2,3,5,1,0,1,2,0,0,1,5,6,2,1,1,6,5,4,1,1,1,0,2,0,1,4,0,1,2,1,5,6,5,0,0,
         6,4,2,2,1,7,0,0,0,0,4,6,2,0,1,0,2,2,0,0,0,1,3,1,0,0,1,1,2,1,0,2,3,0,0,5,
         7,0,2,0,5,2,2,0,0,2,1,2,2,0,2,6
        ]
#############################################################
suffix = 'brisbanne_run'
save_path = "../../Output_genetic/" + suffix
Path("../../Output_genetic").mkdir(exist_ok=True)
Path(save_path).mkdir(exist_ok=True)
with open(
    save_path + "/genes" + suffix + ".csv", "a", encoding="UTF-8"
) as file:
    writer = csv.writer(file)
    writer.writerow(genes)

def fitness_function(accuracy: float, density: float, depth: int = None) -> Union[np.float64, list[np.float64]]:
    fitness_score = accuracy + 0.05*density
    return fitness_score

backend_coupling_map = backend.coupling_map
backend_basis_gates = backend.basis_gates
flattened_qsvm_connections = [
    item for sublist in qsvm_connections for item in sublist]
fmap, x_idxs = to_quantum(
    genes, gate_dict, nb_features, gates_per_qubits, nb_qubits, 0, qsvm_connections[0], coupling_map = backend_coupling_map
)

nb_samples = data_cv.shape[0]
generation_kernels = np.zeros((nb_samples, nb_samples, 21))
combined_circuits = []
max_circuit_per_job = 300 # depends on the backend
counter = 0

print("preparing all QSVM circuits (this can take a few minutes and take up to 2 GB RAM)")
for i in range(nb_samples):
    for j in range(i+1, data_cv.shape[0]):
        combined_circuit = QuantumCircuit(127, nb_cbits)
        for k in range(20):
            bound_circuit = fmap.assign_parameters(data_cv[i, x_idxs]).compose(
                fmap.assign_parameters(data_cv[j, x_idxs]).inverse())
            combined_circuit.compose(bound_circuit, qubits=[
                                    qsvm_connections[k][l] for l in range(nb_qubits)], inplace=True)
        combined_circuit.measure(flattened_qsvm_connections, cbits)
        if counter % max_circuit_per_job == 0:
            combined_circuit_batch = []
        combined_circuit_batch.append(
            transpile(combined_circuit, basis_gates=backend_basis_gates, coupling_map=backend_coupling_map, optimization_level=0))

        if counter % max_circuit_per_job == max_circuit_per_job - 1 or counter == (nb_samples**2 - nb_samples)/2 - 1:            
            combined_circuits.append(combined_circuit_batch)
        counter += 1

# Prepare, send, retrieve jobs

job_results = []
print("Running job")
with Session(backend=backend): #, service=service):
    job_executions = [backend.run(combined_circuits[i], shots=8000) for i in range(counter // max_circuit_per_job + 1)]
for i in range(counter // max_circuit_per_job + 1):
    job_executions[i].wait_for_final_state()
    job_results.append(job_executions[i].result())

n_jobs= len(job_results)
big_counts = []
for i in range(n_jobs):
    print("getting counts from job", i)
    big_counts.append(job_results[i].get_counts())
counter = 0

print("Creating kernel matrices from job output. This can take a few minutes")
for i in range(nb_samples):
    for j in range(i+1, nb_samples):
        n_job = counter // max_circuit_per_job
        counts = big_counts[n_job][counter % max_circuit_per_job]
        counter += 1
        for k in range(21):
            generation_kernels[i, j, k] = get_kernel_entry(cbits[nb_qubits*k:nb_qubits*(k+1)], counts, nb_qubits)

# Symmetrizing and adding 1s to kernel diagonals
for k in range(21):
    generation_kernels[:, :, k] += generation_kernels[:,
                                                        :, k].T + np.eye(nb_samples)
    clf = SVC(kernel="precomputed")

    param_grid = {"C": [0.01, 0.1, 1, 10, 100, 1000, 10000]}
    grid_search = GridSearchCV(
        clf, param_grid, cv=5, scoring="accuracy", verbose=0)
    grid_search.fit(generation_kernels[:, :, k], data_labels)
    best_clf = grid_search.best_estimator_
    accuracy_cv_cost = cross_val_score(
        best_clf,
        generation_kernels[:, :, k],
        data_labels,
        cv=5,
        scoring="accuracy",
    ).mean()
    qker_matrix_0 = generation_kernels[:, :, k][data_labels == 0]
    qker_matrix_0 = np.triu(qker_matrix_0[:, data_labels == 0], 1)
    qker_array_0 = qker_matrix_0[np.triu_indices(
        qker_matrix_0.shape[0], 1)]
    qker_matrix_1 = generation_kernels[:, :, k][data_labels == 1]
    qker_matrix_1 = np.triu(qker_matrix_1[:, data_labels == 1], 1)
    qker_array_1 = qker_matrix_1[np.triu_indices(
        qker_matrix_1.shape[0], 1)]

    qker_matrix_01 = generation_kernels[:, :, k][data_labels == 0]
    qker_matrix_01 = qker_matrix_01[:, data_labels == 1]

    sparsity_cost = (np.mean(qker_array_0) + np.mean(qker_array_1)) / 2 - np.mean(
        qker_matrix_01
    )
    offdiagonal_mean = np.mean(np.triu(generation_kernels[:, :, k], 1))
    offdiagonal_std = np.std(np.triu(generation_kernels[:, :, k], 1))
    fitness_value = fitness_function(
        accuracy_cv_cost, offdiagonal_std)
    print("sparsity", sparsity_cost)
    print("accuracy", accuracy_cv_cost)
    print("fitness_value", fitness_value)

    with open(
        save_path + "/kernels_flattened" + suffix + ".csv", "a", encoding="UTF-8"
    ) as file:
        writer = csv.writer(file)
        writer.writerow(generation_kernels[:, :, k].reshape(-1))
    with open(
        save_path + "/sparsity" + suffix + ".txt", "a", encoding="UTF-8"
    ) as file:
        file.write(str(sparsity_cost) + "\n")
    with open(
        save_path + "/accuracy" + suffix + ".txt", "a", encoding="UTF-8"
    ) as file:
        file.write(str(accuracy_cv_cost) + "\n")
    with open(
        save_path + "/fitness_values_iter_" + suffix + ".txt", "a", encoding="UTF-8"
    ) as file:
        file.write(str(fitness_value) + "\n")
    with open(
        save_path + "/offdiagonal_mean_" + suffix + ".txt", "a", encoding="UTF-8"
    ) as file:
        file.write(str(offdiagonal_mean) + "\n")
    with open(
        save_path + "/offdiagonal_std_" + suffix + ".txt", "a", encoding="UTF-8"
    ) as file:
        file.write(str(offdiagonal_std) + "\n")

# Saving last-generation information that will be loaded by the fitness function.
    with open(save_path + "/last_generation_fitness_values_" + suffix + ".csv", "w" if k == 0 else "a", encoding="UTF-8") as file:
        writer = csv.writer(file)
        writer.writerow(fitness_value) if hasattr(
            fitness_value, "len") > 1 else writer.writerow([fitness_value])
    with open(save_path + "/last_generation_genes_" + suffix + ".csv", "w" if k == 0 else "a", encoding="UTF-8") as file:
        writer = csv.writer(file)
        writer.writerow(genes)
