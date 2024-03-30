"""
Polymer model building tools.

Developed by: Tan Shendong
Date: 2024.03.26
"""


import os
import parmed as pmd
from foyer import Forcefield
from PEMD.model import PEMD_lib
import importlib.resources as pkg_resources


def gen_gmx_oplsaa(unit_name, out_dir, length):

    current_path = os.getcwd()
    relax_polymer_lmp_dir = os.path.join(current_path, out_dir, 'relax_polymer_lmp')

    file_base = f"{unit_name}_N{length}"
    top_filename = os.path.join(relax_polymer_lmp_dir, f"{file_base}.top")
    gro_filename = os.path.join(relax_polymer_lmp_dir, f"{file_base}.gro")

    pdb_filename = None

    for file in os.listdir(relax_polymer_lmp_dir):
        if file.endswith(".xyz"):
            xyz_filename = os.path.join(relax_polymer_lmp_dir, f"{file_base}_gmx.xyz")
            pdb_filename = os.path.join(relax_polymer_lmp_dir, f"{file_base}_gmx.pdb")

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



















