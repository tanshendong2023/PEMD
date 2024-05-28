import numpy as np
from tqdm.auto import tqdm
from PEMD.analysis import msd as msd_module


def compute_conductivity(run, run_start, dt_collection, cations_list, anions_list, times, dt, T, interval_time=5000):

    # compute sum over all charges and positions
    qr = []
    for _ts in tqdm(run.trajectory[int(run_start / dt_collection):]):
        qr_temp = np.zeros(3)
        for cation in cations_list:
            qr_temp += cation.center_of_mass() * int(1)
        for anion in anions_list:
            qr_temp += anion.center_of_mass() * int(-1)
        qr.append(qr_temp)
    msd = msd_module.msd_fft(np.array(qr))

    # Utilize the common slope calculation function
    slope, time_range = msd_module.compute_slope_msd(msd, times, dt_collection, dt, interval_time)

    A2cm = 1e-8  # Angstroms to cm
    ps2s = 1e-12  # picoseconds to seconds
    e2c = 1.60217662e-19  # elementary charge to Coulomb
    kb = 1.38064852e-23  # Boltzmann Constant, J/K
    convert = e2c * e2c / ps2s / A2cm * 1000
    v = (run.dimensions[0]) ** 3.0

    cond = slope / 6 / kb / T / v * convert   # "mS/cm"

    return msd, cond, time_range





