import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from PEMD.analysis import coordination
from statsmodels.tsa.stattools import acovf

def calc_acf(a_values: dict[str, np.ndarray]) -> list[np.ndarray]:
    acfs = []
    for neighbors in a_values.values():  # for _atom_id, neighbors in a_values.items():
        acfs.append(acovf(neighbors, demean=False, adjusted=True, fft=True))
    return acfs


distance_dict = {"polymer": 3.625, "anion": 3.125}

select_dict = {
    "cation": "resname LIP and name Li",
    "anion": "resname NSC and name OBT",
    "polymer": "resname MOL and name O",
}

def calc_neigh_corr(run, distance_dict, select_dict, run_start, run_end, center_atom):
    acf_avg = {}
    center_atoms = run.select_atoms(select_dict[center_atom])
    # center_atoms = run.select_atoms('resname LIP and name Li')
    for kw in distance_dict:
        acf_all = []
        for atom in tqdm(center_atoms[::]):
            distance = distance_dict.get(kw)
            assert distance is not None
            bool_values = {}
            for time_count, _ts in enumerate(run.trajectory[run_start:run_end:]):
                if kw in select_dict:
                    selection = (
                            "("
                            + select_dict[kw]
                            + ") and (around "
                            + str(distance)
                            + " index "
                            + str(atom.id - 1)
                            + ")"
                    )
                    shell = run.select_atoms(selection)
                else:
                    raise ValueError("Invalid species selection")
                for atom in shell.atoms:
                    if str(atom.id) not in bool_values:
                        bool_values[str(atom.id)] = np.zeros(int((run_end - run_start) / 1))
                    bool_values[str(atom.id)][time_count] = 1
            acfs = calc_acf(bool_values)
            acf_all.extend(list(acfs))
        acf_avg[kw] = np.mean(acf_all, axis=0)
    return acf_avg



















