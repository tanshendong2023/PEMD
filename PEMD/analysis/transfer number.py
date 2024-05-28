from PEMD.analysis import msd


def compute_transfer_number(run, dt_collection, cation_positions, anion_positions, times, dt, T, interval_time):

    msds_all = msd.compute_all_Lij(cation_positions, anion_positions, times)

    # Utilize the common slope calculation function
    slope_plusplus, time_range_plusplus = msd.compute_slope_msd(msds_all[0], times, dt_collection, dt, interval_time)
    slope_minusminus, time_range_minusminus = msd.compute_slope_msd(msds_all[2], times, dt_collection, dt,
                                                                    interval_time)
    slope_plusminus, time_range_plusminus = msd.compute_slope_msd(msds_all[4], times, dt_collection, dt, interval_time)

    A2cm = 1e-8  # Angstroms to cm
    ps2s = 1e-12  # picoseconds to seconds
    e2c = 1.60217662e-19  # elementary charge to Coulomb
    kb = 1.38064852e-23  # Boltzmann Constant, J/K
    convert = e2c * e2c / ps2s / A2cm * 1000
    v = (run.dimensions[0]) ** 3.0

    t = (slope_plusplus + slope_minusminus - 2 * slope_plusminus) / 6 / kb / T / v * convert   # mS/cm

    return t