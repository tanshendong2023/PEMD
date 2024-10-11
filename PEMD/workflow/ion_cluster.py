import numpy as np
from tqdm.auto import tqdm
from PEMD.analysis import coordination
from concurrent.futures import ProcessPoolExecutor, as_completed


def distance(x0, x1, box_length):
    delta = x1 - x0
    delta = np.where(delta > 0.5 * box_length, delta - box_length, delta)
    delta = np.where(delta < -0.5 * box_length, delta + box_length, delta)
    return delta

def calc_population_frame(ts, cations, anions, index):
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

def calc_population_parallel(run, run_start, run_end, select_cations, select_anions):
    stacked_population = np.array([])

    with ProcessPoolExecutor() as executor:
        futures = []
        for idx, ts in enumerate(run.trajectory[run_start:run_end]):
            cations = run.select_atoms(select_cations)
            anions = run.select_atoms(select_anions)
            futures.append(executor.submit(calc_population_frame, ts, cations, anions, idx))

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
    return avg_population

if __name__ == '__main__':

    work_dir = '../'
    tpr_file = 'nvt_prod.tpr'
    xtc_wrap_file = 'nvt_prod.xtc'
    select_cations = 'name Li'
    select_anions = 'name NBT or name OBT or name SBT'
    run_start = 0
    run_end = 40001

    run = coordination.load_md_trajectory(work_dir, tpr_file, xtc_wrap_file)
    avg_population = calc_population_parallel(run, run_start, run_end, select_cations, select_anions)

    np.savetxt('avg_population.txt', avg_population, fmt='%.6f')