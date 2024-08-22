# Copyright (c) 2024. PEMD developers. All rights reserved.
# Distributed under the terms of the MIT License.

# ******************************************************************************
# Module Docstring
# ******************************************************************************

from PEMD.core.model import PEMDModel
from PEMD.model.build import (
    gen_poly_smiles,
    gen_poly_3D,
)
from PEMD.simulation.qm import (
    conformer_search_xtb,
    conformer_search_gaussian,
    calc_resp_gaussian
)
from PEMD.simulation.md import (
    gen_poly_gmx_oplsaa,
)

class PEMDSimulation:

    def __init__(
            self,
    ):
        pass

    def conformer_search(
            self,
            smiles,
            epsilon,
            core,
            memory,
            function,
            basis_set,
    ):
        structures = conformer_search_xtb(
            smiles,
            epsilon,
            max_conformers=1000,
            top_n_MMFF=100,
            top_n_xtb=10,
        )

        return conformer_search_gaussian(
            structures,
            core,
            memory,
            function,
            basis_set,
            epsilon
        )

    def calc_resp_charge(
            self,
            epsilon,
            core,
            memory,
            function,
            basis_set,
            method, # resp1 or resp2
    ):
        sorted_df = self.conformer_search(
            epsilon,
            core,
            memory,
            function,
            basis_set,
        )

        return calc_resp_gaussian(
            sorted_df,
            epsilon,
            core,
            memory,
            method,
        )

    def build_polymer(self,):

        return  gen_poly_3D(
            self.poly_name,
            self.length,
            self.gen_poly_smiles(),
        )

    def gen_polymer_force_field(self,):

        gen_poly_gmx_oplsaa(
            self.poly_name,
            self.poly_resname,
            self.poly_scale,
            self.poly_charge,
            self.length,
        )








