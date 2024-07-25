
import os
import numpy as np
import MDAnalysis as mda
from typing import Tuple
from functools import lru_cache
from PEMD.analysis.conductivity import calc_cond_msd, calc_conductivity
from PEMD.analysis.transfer_number import calc_transfer_number
from PEMD.analysis.coordination import calc_rdf_coord, obtain_rdf_coord
from PEMD.analysis.msd import (
    calc_slope_msd,
    create_position_arrays,
    calc_Lii_self,
    calc_Lii,
    calc_self_diffusion_coeff,
)


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
        self.times = self.times_range(self.run_end)
        self.cation_name = cation_name
        self.anion_name = anion_name
        self.cations_unwrap = run_unwrap.select_atoms(self.cation_name)
        self.anions_unwrap = run_unwrap.select_atoms(self.anion_name)
        self.volume = self.run_unwrap.coord.volume

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

    def get_slope_msd(self, msd_array, interval_time=5000, step_size=20):

        slope, time_range = calc_slope_msd(
            self.times,
            msd_array,
            self.dt_collection,
            self.dt,
            interval_time,
            step_size
        )

        return slope, time_range

    def get_conductivity(self):

        slope, time_range = self.get_slope_msd(self.get_cond_array())

        return calc_conductivity(slope, self.volume, self.temp)

    @lru_cache(maxsize=128)
    def get_ions_positions_array(self):

        cations_positions, anions_positions = create_position_arrays(
            self.run_unwrap,
            self.cations_unwrap,
            self.anions_unwrap,
            self.times,
            self.run_start,
        )
        return cations_positions, anions_positions

    def get_Lii_self_array(self, atom_positions):
        n_atoms = np.shape(atom_positions)[1]
        return calc_Lii_self(atom_positions, self.times) / n_atoms

    def get_self_diffusion_coefficient(self) -> Tuple[float, float]:

        cations_positions, anions_positions = self.get_ions_positions_array()
        slope_cations, time_range_cations = self.get_slope_msd(self.get_Lii_self_array(cations_positions))
        slope_anions, time_range_anions = self.get_slope_msd(self.get_Lii_self_array(anions_positions))

        D_cations = calc_self_diffusion_coeff(slope_cations)
        D_anions = calc_self_diffusion_coeff(slope_anions)

        return D_cations, D_anions

    def get_transfer_number(self):

        cations_positions, anions_positions = self.get_ions_positions_array()
        slope_plusplus, time_range_plusplus = self.get_slope_msd(calc_Lii(cations_positions))
        slope_minusminus, time_range_minusminus = self.get_slope_msd(calc_Lii(anions_positions))

        return calc_transfer_number(
            slope_plusplus,
            slope_minusminus,
            self.temp,
            self.volume,
            self.get_conductivity()
        )

    def get_rdf_coordination_array(self, group1_name, group2_name):

        group1 = self.run_wrap.select_atoms(group1_name)
        group2 = self.run_wrap.select_atoms(group2_name)
        bins, rdf, coord_number = calc_rdf_coord(
            group1,
            group2,
            self.volume
        )
        return bins, rdf, coord_number

    def get_coordination_number(self, group1_name, group2_name):

        bins, rdf, coord_number = self.get_rdf_coordination_array(group1_name, group2_name)
        y_coord = obtain_rdf_coord(bins, rdf, coord_number)[1]

        return y_coord
