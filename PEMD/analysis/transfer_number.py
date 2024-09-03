
# ****************************************************************************** #
#      The module implements functions to calculate the transfer number          #
# ****************************************************************************** #

from PEMD.analysis import msd

def compute_transfer_number(run, dt_collection, cation_positions, anion_positions, times, dt, T, interval_time, cond):

    msds_all = msd.compute_all_Lij(cation_positions, anion_positions, times)

    # Utilize the common slope calculation function
    k_plusplus, time_range_plusplus = msd.calc_slope_msd(times, msds_all[0],  dt_collection, dt, interval_time)
    k_minusminus, time_range_minusminus = msd.calc_slope_msd(times, msds_all[2], dt_collection, dt,
                                                                    interval_time)

    A2cm = 1e-8  # Angstroms to cm
    ps2s = 1e-12  # picoseconds to seconds
    e2c = 1.60217662e-19  # elementary charge to Coulomb
    kb = 1.38064852e-23  # Boltzmann Constant, J/K
    convert = e2c * e2c / ps2s / A2cm * 1000
    v = (run.dimensions[0]) ** 3.0

    k_plusminus = (cond / convert * 6 * kb * T * v - k_plusplus - k_minusminus) / -2

    t = (k_plusplus-k_plusminus)/(k_plusplus+k_minusminus-2*k_plusminus)   # mS/cm

    return t


