
# ****************************************************************************** #
#     The module implements functions to calculate the mean square distance      #
# ****************************************************************************** #

import os
import numpy as np
import MDAnalysis as mda
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

def get_position(work_dir, data_tpr_file, dcd_xtc_file, select_cations, select_anions, dt, dt_collection, run_start,
                run_end):

    data_tpr_file_path = os.path.join(work_dir, data_tpr_file)
    dcd_xtc_file_path = os.path.join(work_dir, dcd_xtc_file)

    run = mda.Universe(data_tpr_file_path, dcd_xtc_file_path)

    cations = run.select_atoms(select_cations).residues
    anions = run.select_atoms(select_anions).residues

    cations_list = cations.atoms.split("residue")
    anions_list = anions.atoms.split("residue")

    t_total = run_end - run_start
    times = np.arange(0, t_total * dt * dt_collection, dt * dt_collection, dtype=int)

    return run, cations, cations_list, anions, anions_list, times

def autocorrFFT(x):

    N = len(x)
    F = np.fft.fft(x, n=2 * N)
    PSD = F * F.conjugate()
    res = np.fft.ifft(PSD)
    res = (res[:N]).real
    n = N * np.ones(N) - np.arange(0, N)
    acf = res / n
    return acf

def msd_fft(r):

    N = len(r)
    D = np.square(r).sum(axis=1)
    D = np.append(D, 0)
    S2 = sum([autocorrFFT(r[:, i]) for i in range(r.shape[1])])
    Q = 2 * D.sum()
    S1 = np.zeros(N)
    for m in range(N):
        Q = Q - D[m - 1] - D[N - m]
        S1[m] = Q / (N - m)
    msd = S1 - 2 * S2
    return msd

def cross_corr(x, y):

    N = len(x)
    F1 = np.fft.fft(x, n=2 ** (N * 2 - 1).bit_length())
    F2 = np.fft.fft(y, n=2 ** (N * 2 - 1).bit_length())
    PSD = F1 * F2.conjugate()
    res = np.fft.ifft(PSD)
    res = (res[:N]).real
    n = N * np.ones(N) - np.arange(0, N)
    cf = res / n
    return cf

def msd_fft_cross(r, k):

    N = len(r)
    D = np.multiply(r, k).sum(axis=1)
    D = np.append(D, 0)
    S2 = sum([cross_corr(r[:, i], k[:, i]) for i in range(r.shape[1])])
    S3 = sum([cross_corr(k[:, i], r[:, i]) for i in range(k.shape[1])])
    Q = 2 * D.sum()
    S1 = np.zeros(N)
    for m in range(N):
        Q = Q - D[m - 1] - D[N - m]
        S1[m] = Q / (N - m)
    msd = S1 - S2 - S3
    return msd

def calc_Lii_self(atom_positions, times):

    Lii_self = np.zeros(len(times))
    n_atoms = np.shape(atom_positions)[1]
    for atom_num in tqdm(range(n_atoms), desc="Calculating MSD"):
        r = atom_positions[:,atom_num, :]
        msd_temp = msd_fft(np.array(r))
        Lii_self += msd_temp
    msd = np.array(Lii_self)
    return msd

def calc_Lii(atom_positions,):

    r_sum = np.sum(atom_positions, axis = 1)
    msd = msd_fft(r_sum)
    return np.array(msd)

def calc_Lij(cation_positions, anion_positions,):

    r_cat = np.sum(cation_positions, axis = 1)
    r_an = np.sum(anion_positions, axis = 1)
    msd = msd_fft_cross(np.array(r_cat),np.array(r_an))
    return np.array(msd)

def compute_all_Lij(cation_positions, anion_positions, times):

    msd_self_cation = calc_Lii_self(cation_positions, times)
    msd_self_anion =  calc_Lii_self(anion_positions, times)
    msd_cation = calc_Lii(cation_positions,)
    msd_anion = calc_Lii(anion_positions,)
    msd_distinct_catAn = calc_Lij(cation_positions, anion_positions,)
    msds_all = [msd_cation, msd_self_cation, msd_anion, msd_self_anion, msd_distinct_catAn]
    return msds_all

def create_position_arrays(run, cations_list, anions_list, times, run_start):

    time = 0

    cation_positions = np.zeros((len(times), len(cations_list), 3))
    anion_positions = np.zeros((len(times), len(anions_list), 3))

    for ts in enumerate(tqdm(run.trajectory[int(run_start):])):
        system_com = run.atoms.center_of_mass(wrap=True)
        for index, cation in enumerate(cations_list):
            cation_positions[time, index, :] = cation.center_of_mass() - system_com
        for index, anion in enumerate(anions_list):
            anion_positions[time, index, :] = anion.center_of_mass() - system_com
        time += 1

    return cation_positions, anion_positions

def calc_slope_msd(times_array, msd_array, dt_collection, dt, interval_time=5000, step_size=20):
    # Log transformation
    log_time = np.log(times_array[1:])
    log_msd = np.log(msd_array[1:])

    # calculate the time interval
    dt_ = dt_collection * dt
    interval_msd = int(interval_time / dt_)

    # Initialize a list to store the average slope for each large interval
    average_slopes = []
    closest_slope = float('inf')
    time_range = (None, None)

    # Use a sliding window to calculate the average slope for each large interval
    for i in range(0, len(log_time) - interval_msd, step_size):
        if i + interval_msd > len(log_time):  # Ensure not to go out of bounds
            break
        # 使用 np.gradient 计算斜率
        local_slope = np.gradient(log_msd[i:i + interval_msd], log_time[i:i + interval_msd])
        slope = np.mean(local_slope)
        average_slopes.append(slope)

        # Update the average slope closest to 1 and its range
        if abs(slope - 1) < abs(closest_slope - 1):
            closest_slope = slope
            time_range = (times_array[i], times_array[i + interval_msd])

    # Calculate the final slope
    final_slope = (msd_array[int(time_range[1] / dt_)] - msd_array[int(time_range[0] / dt_)]) / (time_range[1] - time_range[0])

    return final_slope, time_range

def compute_self_diffusion(atom_positions, times, dt_collection, dt, interval_time, step_size):

    n_atoms = np.shape(atom_positions)[1]
    msd = calc_Lii_self(atom_positions, times) / n_atoms  # mean for particle

    # Utilize the common slope calculation function
    slope, time_range = calc_slope_msd(times, msd, dt_collection, dt, interval_time, step_size)

    A2cm = 1e-8  # Angstroms to cm
    ps2s = 1e-12  # picoseconds to seconds
    convert = (A2cm ** 2) / ps2s   # cm^2/s
    D = slope * convert / 6

    return msd, D, time_range

def plot_msd(msd_data, times, time_ranges, dt_collection, dt, labels, save_file):
    font_list = {"title": 20, "label": 18, "legend": 16, "ticket": 18, "data": 14}
    color_list = ["#DF543F", "#2286A9", "#FBBF7C", "#3C3846"]

    dt_ = dt_collection * dt
    fig, ax = plt.subplots()

    # 判断是单个MSD还是双MSD
    if isinstance(msd_data, list):  # 假设msd_data是列表，包含两个MSD数组
        for i, msd_ in enumerate(msd_data):
            mid_time = (time_ranges[i][1] + time_ranges[i][0]) / 2
            start = int(10 ** (np.log10(mid_time) - 0.15) / dt_)
            end = int(10 ** (np.log10(mid_time) + 0.15) / dt_)
            scale = (msd_[int(mid_time / dt_)] + 40) / mid_time

            x_log = times[start:end]
            y_log = x_log * scale

            ax.plot(times[1:], msd_[1:], '-', linewidth=1.5, color=color_list[i], label=labels[i])
            ax.plot(x_log, y_log, '--', linewidth=2, color=color_list[i])
    else:  # 单个MSD
        mid_time = (time_ranges[1] + time_ranges[0]) / 2
        start = int(10 ** (np.log10(mid_time) - 0.15) / dt_)
        end = int(10 ** (np.log10(mid_time) + 0.15) / dt_)
        scale = (msd_data[int(mid_time / dt_)] + 40) / mid_time

        x_log = times[start:end]
        y_log = x_log * scale

        ax.plot(times[1:], msd_data[1:], '-', linewidth=1.5, color=color_list[0], label=labels)
        ax.plot(x_log, y_log, '--', linewidth=2, color="grey")

    ax.legend(fontsize=font_list["legend"], frameon=False)
    ax.set_xlabel(r'$t$ (ps)', fontsize=font_list["label"])
    ax.set_ylabel(r'MSD ($\AA^2$)', fontsize=font_list["label"])
    ax.tick_params(axis='both', which='both', direction='in', labelsize=font_list["ticket"])

    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.set_xlim(1e2,)

    ax.grid(True, linestyle='--')
    fig.set_size_inches(5.5, 4)
    plt.tight_layout()

    # Save the plot
    plt.savefig(f'{save_file}', bbox_inches='tight', dpi=300)
    plt.show()
