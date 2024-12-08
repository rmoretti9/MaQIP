""" This module executes an automatized quantum featuremap optimization via genetic algorithm using integer genes"""
import numpy as np
from pathlib import Path
from collections import OrderedDict
from maqip import genetic_statevector as genetic
from typing import Union

from qiskit_aer import StatevectorSimulator

import time
import shutil
import os

data_cv, data_labels = np.load("../processed_dataset/data_cv.npy"), np.load("../processed_dataset/data_labels.npy")
nb_features = data_cv.shape[1]

###########################################
############ Genetic settings #############
###########################################

NB_QUBITS = 4
GATES_PER_QUBITS = 8
NB_INIT_INDIVIDUALS = 30
gate_dict = OrderedDict(
    [
        ("single_non_parametric", ["Id", "X","SX"]),
        ("single_parametric", ["RZ", "RZ"]),
        ("two_non_parametric", ["ECR"]),
        ("two_parametric", []),
    ]
)
# One could instead encode gate-block (involving more than one qubit for example) and minimize metrix like depth or optimize particular transpilings.
coupling_map = None # [[0,1], [1,2], [2,3], [3,4], [4,5], [5,6], [6,7]]
basis_gates = None # ['cx', 'id', 'rz', 'sx', 'x'] 

generation_zero = genetic.initial_population(
    NB_INIT_INDIVIDUALS, NB_QUBITS, GATES_PER_QUBITS, gate_dict, nb_features
)

gene_space = genetic.get_gene_space(gate_dict, nb_features, NB_QUBITS, GATES_PER_QUBITS)

def fitness_function(accuracy: float, density: float, depth: int) -> Union[np.float64, list[np.float64]]:
    """Customizable fitness function of some QSVM metrics.

    Parameters
    ----------
    accuracy: float
        5-fold cross validation accuracy.
    density: float
        Function of the off-diagonal kernel elements.
    depth: int
        Transpiled quantum circuit depth.

    Returns
    -------
    fitness_score: Union[np.float64, list[np.float64]]
        Quantum kernel fitness value. If it is a list, the run will be optimized with the NSGA-II algorithm for multi-objective optimization.
    """
    fitness_score = accuracy + 0.025*density
    return fitness_score

# Defining inputs for the genetic instance
options = {
    "num_generations": 100,
    "num_parents_mating": 20,
    "initial_population": generation_zero,
    "parent_selection_type": "sss",
    "mutation_by_replacement": True,
    "stop_criteria": "saturate_250",
    "mutation_type": "random",
    "mutation_percent_genes": 1.5 * 4 / NB_QUBITS * 8 / GATES_PER_QUBITS,
    "crossover_probability": 0.1,
    "crossover_type": "two_points",
    "allow_duplicate_genes": True,
    "keep_elitism": 5,
    "fit_fun": fitness_function,
    "noise_std": 0,
}

# Running the instance and retrieving data
backend = StatevectorSimulator(precision='single')
projected = False
timestr = time.strftime("%Y_%m_%d -%H_%M_%S")

ga_instance = genetic.genetic_instance(
    gene_space,
    data_cv,
    data_labels,
    backend,
    gate_dict,
    nb_features,
    GATES_PER_QUBITS,
    NB_QUBITS,
    projected,
    timestr,
    coupling_map=coupling_map,
    basis_gates=basis_gates,
    **options,
)

ga_instance.run()

solution, solution_fitness, _ = ga_instance.best_solution(
    ga_instance.last_generation_fitness
)
save_path = "../../Output_genetic/" + timestr
Path(save_path).mkdir(exist_ok=True)
with open(save_path + "/best_solution.txt", "w") as genes_file:
    np.savetxt(genes_file, solution)
with open(save_path + "/best_fitness.txt", "w") as file:
    file.write(str(solution_fitness) + "\n")
with open(save_path + "/best_fitness_per_generation.txt", "w") as file:
    file.write(str(ga_instance.best_solutions_fitness) + "\n")
np.save(save_path + "/data_cv", data_cv)
np.save(save_path + "/labels_cv", data_labels)

copied_script_name = time.strftime("%Y-%m-%d_%H%M") + "_" + os.path.basename(__file__)
shutil.copy(__file__, save_path + os.sep + copied_script_name)
