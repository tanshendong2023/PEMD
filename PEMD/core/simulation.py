# Copyright (c) 2024. PEMD developers. All rights reserved.
# Distributed under the terms of the MIT License.

# ******************************************************************************
# Module Docstring
# ******************************************************************************


import os
import json
from PEMD.simulation.qm import conformer_search_xtb, conformer_search_gaussian


class PEMDSimulation:
    def __init__(self, poly_name, repeating_unit, leftcap, rightcap, length_short, length, ):
        self.poly_name = poly_name
        self.repeating_unit = repeating_unit
        self.leftcap = leftcap
        self.rightcap = rightcap
        self.length_short = length_short
        self.length = length

    @classmethod
    def from_json(cls, work_dir, json_file):

        json_path = os.path.join(work_dir, json_file)
        with open(json_path, 'r', encoding='utf-8') as file:
            model_info = json.load(file)

        poly_name = model_info['polymer']['compound']
        repeating_unit = model_info['polymer']['repeating_unit']
        leftcap = model_info['polymer']['terminal_cap']
        rightcap = model_info['polymer']['terminal_cap']
        length_short = model_info['polymer']['length'][0]
        length = model_info['polymer']['length'][1]

        return cls(poly_name, repeating_unit, leftcap, rightcap, length, length_short)

    def conformer_search(self, name, smiles, epsilon, core, ):

        structures = conformer_search_xtb(
            name,
            smiles,
            epsilon,
            core,
            max_conformers=1000,
            top_n_MMFF=100,
            top_n_xtb=10,
        )

        sorted_df = conformer_search_gaussian(
            structures,
            name,
            core=32,
            memory='64GB',
            function='B3LYP',
            basis_set='6-311+g(d,p)',
            dispersion_corr='em=GD3BJ',
        )

        return sorted_df

