"""
Polymer model building tools.

Developed by: Tan Shendong
Date: 2024.03.26
"""


import os
from foyer import Forcefield
import parmed as pmd
from PEMD.model import PEMD_lib
import importlib.resources as pkg_resources


def gen_gmx_oplsaa(unit_name, out_dir, length):
    current_path = os.getcwd()
    filepath = current_path + '/' + out_dir

    top_filename = filepath + '/' + f'{unit_name}{length}.top'
    gro_filename = filepath + '/' + f'{unit_name}{length}.gro'

    pdb_filename = None

    for file in os.listdir(filepath):
        if file.endswith(".xyz"):
            file_base = '{}_N{}'.format(unit_name, length)
            xyz_filename = out_dir + '/' + f'{file_base}_gmx.xyz'
            pdb_filename = out_dir + '/' + f'{file_base}_gmx.pdb'

            PEMD_lib.convert_xyz_to_pdb(xyz_filename, pdb_filename, f'{unit_name}', f'{unit_name}')

    untyped_str = pmd.load_file(pdb_filename, structure=True)

    with pkg_resources.path("PEMD.sim", "oplsaa.xml") as oplsaa_path:
        oplsaa = Forcefield(forcefield_files=str(oplsaa_path))
    typed_str = oplsaa.apply(untyped_str)

    # Save to any format supported by ParmEd
    typed_str.save(top_filename)
    typed_str.save(gro_filename)

    nonbonditp_filename = out_dir + '/' + f'{unit_name}_nonbonded.itp'
    bonditp_filename = out_dir + '/' + f'{unit_name}_bonded.itp'

    PEMD_lib.extract_from_top(top_filename, nonbonditp_filename, nonbonded=True, bonded=False)

    PEMD_lib.extract_from_top(top_filename, bonditp_filename, nonbonded=False, bonded=True)

    try:
        os.remove(top_filename)
    except Exception:
        pass  # 忽略任何异常

    try:
        os.remove(gro_filename)
    except Exception:
        pass  # 忽略任何异常

def assign_chg_to_gmx():
    pass


