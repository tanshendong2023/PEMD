import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from statsmodels.tsa.stattools import acovf

def times_array(run, run_start, run_end, time_step=5):
    times = []
    for step, _ts in enumerate(run.trajectory[run_start:run_end]):
        times.append(step * time_step)
    return np.array(times)

def calc_acf(a_values: dict[str, np.ndarray]) -> list[np.ndarray]:
    acfs = []
    for neighbors in a_values.values():  # for _atom_id, neighbors in a_values.items():
        acfs.append(acovf(neighbors, demean=False, adjusted=True, fft=True))
    return acfs

def calc_neigh_corr(run, distance_dict, select_dict, run_start, run_end):
    acf_avg = {}
    center_atoms = run.select_atoms('resname LIP and name Li')
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
                # 获取这些原子所属的分子ID
                mols = set(atom.residue for atom in shell)

                for mol in mols:
                    if str(mol.resid) not in bool_values:
                        bool_values[str(mol.resid)] = np.zeros(int((run_end - run_start) / 1))
                    bool_values[str(mol.resid)][time_count] = 1

                # for shell_atom in shell.atoms:
                #     if str(shell_atom.id) not in bool_values:
                #         bool_values[str(shell_atom.id)] = np.zeros(int((run_end - run_start) / 1))
                #     bool_values[str(shell_atom.id)][time_count] = 1

            acfs = calc_acf(bool_values)
            acf_all.extend(list(acfs))
        acf_avg[kw] = np.mean(acf_all, axis=0)
    return acf_avg

def biexponential_func(t, a, tau_res, beta, tau_short):
    return a * np.exp(-(t / tau_res) ** beta) + (1 - a) * np.exp(-t / tau_short)

def fit_residence_time(times, acf_avg_dict, cutoff_time, time_step):
    acf_avg_norm = {}
    popt = {}
    pcov = {}
    tau = {}
    species_list = list(acf_avg_dict.keys())

    # Exponential fit of solvent-Li ACF
    for kw in species_list:
        acf_avg_norm[kw] = acf_avg_dict[kw] / acf_avg_dict[kw][0]

        popt[kw], pcov[kw] = curve_fit(
            biexponential_func,
            times[:cutoff_time],
            acf_avg_norm[kw][:cutoff_time],
            p0=(0.5, 100, 1, 10)
        )
        tau[kw] = popt[kw][1]  # ps

    # Plot ACFs
    colors = ["b", "g", "r", "c", "m", "y"]
    line_styles = ["-", "--", "-.", ":"]
    for i, kw in enumerate(species_list):
        plt.plot(times, acf_avg_norm[kw], label=kw, color=colors[i])
        fitted_x = np.linspace(0, cutoff_time * time_step, cutoff_time)
        fitted_y = biexponential_func(np.linspace(0, cutoff_time * time_step, cutoff_time), *popt[kw])
        save_decay = np.vstack(
            (
                times[:cutoff_time],
                acf_avg_norm[kw][:cutoff_time],
                fitted_x,
                fitted_y,
            )
        )

        plt.plot(
            fitted_x,
            fitted_y,
            line_styles[i],
            color="k",
            label=kw + " Fit",
        )

    plt.xlabel("Time (ps)")
    plt.legend()
    plt.ylabel("Neighbor Auto-correlation Function")
    plt.ylim(0, 1)
    plt.xlim(0, cutoff_time * time_step)
    plt.show()

    return tau

