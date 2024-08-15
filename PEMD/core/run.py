# Copyright (c) 2024. PEMD developers. All rights reserved.
# Distributed under the terms of the MIT License.

# ******************************************************************************
# Module Docstring
# ******************************************************************************


import os
import json
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
            poly_name,
            poly_resname,
            repeating_unit,
            leftcap,
            rightcap,
            length_short,
            length,
            poly_scale,
            poly_charge
    ):
        self.poly_name = poly_name
        self.poly_resname = poly_resname
        self.repeating_unit = repeating_unit
        self.leftcap = leftcap
        self.rightcap = rightcap
        self.length_short = length_short
        self.length = length
        self.poly_scale = poly_scale
        self.poly_charge = poly_charge

    @classmethod
    def from_json(
            cls,
            work_dir,
            json_file
    ):

        json_path = os.path.join(work_dir, json_file)
        with open(json_path, 'r', encoding='utf-8') as file:
            model_info = json.load(file)

        poly_name = model_info['polymer']['compound']
        poly_resname = model_info['polymer']['resname']
        repeating_unit = model_info['polymer']['repeating_unit']
        leftcap = model_info['polymer']['terminal_cap']
        rightcap = model_info['polymer']['terminal_cap']
        length_short = model_info['polymer']['length'][0]
        length = model_info['polymer']['length'][1]
        poly_scale = model_info['polymer']['scale']
        poly_charge = model_info['polymer']['charge']

        return cls(poly_name, poly_resname, repeating_unit, leftcap, rightcap, length, length_short, poly_scale, poly_charge)

    def gen_poly_smiles(
            self,
            short=False
    ):

        if short:
            return gen_poly_smiles(
                self.poly_name,
                self.repeating_unit,
                self.leftcap,
                self.rightcap,
                self.length_short,
            )
        else:
            return gen_poly_smiles(
                self.poly_name,
                self.repeating_unit,
                self.leftcap,
                self.rightcap,
                self.length,
            )

    def conformer_search(
            self,
            epsilon,
            core,
            memory,
            function,
            basis_set,
    ):
        smiles = self.gen_poly_smiles(short=True)
        structures = conformer_search_xtb(
            smiles,
            epsilon,
            core,
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








