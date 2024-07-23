
import os
import numpy as np
import MDAnalysis as mda
from PEMD.analysis.conductivity import calc_cond_msd, calculate_slope_msd, calculate_conductivity


class PEMDAnalysis:

    def __init__(self, run_wrap, run_unwrap, cation_name, anion_name, run_start, run_end, dt, dt_collection,
                 temperature):
        self.run_wrap = run_wrap
        self.run_unwrap = run_unwrap
        self.run_start = run_start
        self.run_end = run_end
        self.dt = dt
        self.dt_collection = dt_collection
        self.temp = temperature
        self.times = self.times_range(run_end)
        self.cation_name = cation_name
        self.anion_name = anion_name
        self.cations_unwrap = run_unwrap.select_atoms(self.cation_name)
        self.anions_unwrap = run_unwrap.select_atoms(self.anion_name)

    @classmethod
    def from_gromacs(cls, work_dir, tpr_file, wrap_xtc_file, unwrap_xtc_file, cation_name, anion_name, run_start,
                     run_end, dt, dt_collection, temperature):

        tpr_path = os.path.join(work_dir, tpr_file)
        wrap_xtc_path = os.path.join(work_dir, wrap_xtc_file)
        unwrap_xtc_path = os.path.join(work_dir, unwrap_xtc_file)

        run_wrap = mda.Universe(tpr_path, wrap_xtc_path)
        run_unwrap = mda.Universe(tpr_path, unwrap_xtc_path)

        return cls(run_wrap, run_unwrap, cation_name, anion_name, run_start, run_end, dt, dt_collection, temperature,)

    def times_range(self, end_time):
        t_total = end_time - self.run_start
        return np.arange(0, t_total * self.dt * self.dt_collection, self.dt * self.dt_collection, dtype=int)

    def get_cond_array(self):

        run = self.run_unwrap
        return calc_cond_msd(
            run,
            self.cations_unwrap,
            self.anions_unwrap,
            self.run_start,
        )

    def get_slope_msd(self, times_array, msd_array, interval_time=5000, step_size=20):

        slope, time_range = calculate_slope_msd(
            times_array,
            msd_array,
            self.dt_collection,
            self.dt,
            interval_time,
            step_size
        )

        return slope, time_range

    def get_conductivity(self,):

        v = (self.run_unwrap.dimensions[0]) ** 3.0
        slope, time_range = self.get_slope_msd(self.times, self.get_cond_array())
        return calculate_conductivity(slope, v, self.temp)
