# Copyright (c) 2024. PEMD developers. All rights reserved.
# Distributed under the terms of the MIT License.

# ******************************************************************************
# Module Docstring
# ******************************************************************************

import os
import numpy as np
from tqdm.auto import tqdm
from PEMD.analysis import coordination
from concurrent.futures import ProcessPoolExecutor, as_completed

def distance(x0, x1, box_length):
    delta = x1 - x0
    delta = np.where(delta > 0.5 * box_length, delta - box_length, delta)
    delta = np.where(delta < -0.5 * box_length, delta + box_length, delta)
    return delta

def process_frame(ts, cations, anions, index):
    all_atoms = cations + anions
    merged_list = list(range(len(all_atoms)))
    box_size = ts.dimensions[0]
    all_clusters = []

    while merged_list:
        this_cluster = [merged_list[0]]
        merged_list.remove(merged_list[0])

        for i in this_cluster:
            for j in merged_list:
                d_vec = distance(all_atoms.positions[i], all_atoms.positions[j], box_size)
                d = np.linalg.norm(d_vec)
                if d <= 3.4:
                    this_cluster.append(j)
                    merged_list.remove(j)

        all_clusters.append(this_cluster)

    type_id = 'Li'
    type_id3 = 'NBT'
    pop_matrix = np.zeros((50, 50, 1))

    for cluster in all_clusters:
        cations_count = 0
        anions_count = 0

        for atom_id in cluster:
            if all_atoms[atom_id].type == type_id:
                cations_count += 1
            if all_atoms[atom_id].type == type_id3:
                anions_count += 1

        if cations_count < 50 and anions_count < 50:
            pop_matrix[cations_count][anions_count] += 1

    return pop_matrix, index

def create_position_arrays(u, anions, cations, times, run_start, dt_collection):
    time = 0
    anion_positions = np.zeros((len(times), len(anions), 3))
    cation_positions = np.zeros((len(times), len(cations), 3))
    for ts in tqdm(u.trajectory[int(run_start/dt_collection):]):
        anion_positions[time, :, :] = anions.positions - u.atoms.center_of_mass(wrap=True)
        cation_positions[time, :, :] = cations.positions - u.atoms.center_of_mass(wrap=True)
        time += 1
    return anion_positions, cation_positions


if __name__ == "__main__":
    work_dir = './'
    tpr_file = 'nvt_prod.tpr'
    xtc_wrap_file = 'nvt_prod.xtc'
    run_start = 0
    run_end = 10001

    run = coordination.load_md_trajectory(work_dir, tpr_file, xtc_wrap_file)

    stacked_population = np.array([])

    with ProcessPoolExecutor() as executor:
        futures = []
        for idx, ts in enumerate(run.trajectory[run_start:run_end]):
            cations = run.select_atoms('name Li')
            anions = run.select_atoms('name NBT or name OBT or name F1')
            futures.append(executor.submit(process_frame, ts, cations, anions, idx))

        results = [future.result() for future in
                   tqdm(as_completed(futures), total=len(futures), desc='Processing trajectory')]

    # Sort results by index and combine
    sorted_results = sorted(results, key=lambda x: x[1])
    for current_population, _ in sorted_results:
        if stacked_population.size == 0:
            stacked_population = current_population
        else:
            stacked_population = np.dstack((stacked_population, current_population))

    avg_population = np.mean(stacked_population, axis=2)

    cations = run.select_atoms('name Li')
    anions = run.select_atoms('name NBT')

    dt = 0.001  # simulation timestep (fs)
    dt_collection = 5e3  # position data is collected every 5e3 steps
    run_start = 0
    run_end = 80001
    t_total = run_end - run_start
    times = np.arange(0, t_total * dt * dt_collection, dt * dt_collection, dtype=int)

    anion_positions, cation_positions = create_position_arrays(run, anions, cations, times, run_start, dt_collection)

    A2cm = 1e-8
    ps2s = 1e-12
    msd_anion = np.mean(np.sum((anion_positions[-1] - anion_positions[0]) ** 2, axis=-1))
    diffusivity_anion = msd_anion / (len(anion_positions) - 1) / 6 / dt / dt_collection  # A^2/s
    diffusivity_anion = diffusivity_anion * (A2cm ** 2) / ps2s  # cm^2/s

    msd_cation = np.mean(np.sum((cation_positions[-1] - cation_positions[0]) ** 2, axis=-1))
    diffusivity_cation = msd_cation / (len(cation_positions) - 1) / 6 / dt / dt_collection  # A^2/s
    diffusivity_cation = diffusivity_cation * (A2cm ** 2) / ps2s  # cm^2/s

    max_cluster = 10
    T = 450
    pop_mat = avg_population
    z_i, z_j = 1, 1
    tfsi_diff = diffusivity_anion
    li_diff = diffusivity_cation
    V = run.coord.volume
    A2cm = 1e-8  # Angstroms to cm
    ps2s = 1e-12  # picoseconds to seconds
    e2c = 1.60217662e-19  # elementary charge to Coulomb
    kb = 1.38064852e-23  # Boltzmann Constant, J/K

    cond = 0.
    total_ion = 0.
    tn_numerator, tn_denominator = 0., 0.

    for i in range(max_cluster):
        for j in range(max_cluster):
            if i > j:
                cond += e2c ** 2 / V / kb / T * (i * z_i - j * z_j) ** 2 * pop_mat[i, j] * li_diff / A2cm ** 3 * 1000
                tn_numerator += i * z_i * (i * z_i - j * z_j) * pop_mat[i, j] * li_diff
                tn_denominator += (i * z_i - j * z_j) ** 2 * pop_mat[i, j] * li_diff
            elif i < j:
                cond += e2c ** 2 / V / kb / T * (i * z_i - j * z_j) ** 2 * pop_mat[i, j] * tfsi_diff / A2cm ** 3 * 1000
                tn_numerator += i * z_i * (i * z_i - j * z_j) * pop_mat[i, j] * tfsi_diff
                tn_denominator += (i * z_i - j * z_j) ** 2 * pop_mat[i, j] * tfsi_diff
            else:
                pass
            total_ion += (i + j) * pop_mat[i, j]
    tn = tn_numerator / tn_denominator

    save_dir = './'
    file_path_1 = os.path.join(save_dir, 'population.txt')
    np.savetxt(file_path_1, avg_population, fmt='%f', delimiter=',')

    # Define the path for the results file
    file_path_2 = os.path.join(save_dir, 'cNE.txt')

    # Write results to a text file
    with open(file_path_2, 'w') as file:
        file.write(f"Calculated conductivity: {cond:.2f} ms/cm\n")
        file.write(f"Calculated D_cations: {diffusivity_cation:.2f} cm2/s\n")
        file.write(f"Calculated D_anions: {diffusivity_anion:.2f} cm2/s\n")
        file.write(f"Calculated transfer number: {tn:.2f} \n")

    print(f"Results saved to {file_path_2}")



