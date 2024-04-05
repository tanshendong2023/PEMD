#!/usr/bin/env python


"""
This script is used to start the molecular dynamics of polymer.
Author: Shendong Tan
Date: 2024-04-01
"""


from PEMD.model import poly
from PEMD.sim import qm, MD
from PEMD.analysis import prop


unit_name = 'PEO'
repeating_unit = '[*]CCO[*]'
leftcap = 'CO[*]'
rightcap = 'C[*]'
length_resp = 10                     # for resp charge fitting via show chain polymer
out_dir_resp = 'PEO_N10'             # for resp charge fitting via show chain polymer
length_MD = 50                       # for MD
out_dir_MD = 'PEO_N50'               # for MD
end_repeating = 2                    # keep the charge of polymer end group
density = 0.8                        # system density
add_length = 25                      # unit: Å
numbers = [20]                       # the number of polymer chain
pdb_files =[]
compositions=['PEO']
top_filename='topol.top'


if __name__ == '__main__':
    # 1. obtain RESP charge fitting result
    # Generate polymer monomer from smiles
    smiles_resp, mol_resp = poly.mol_from_smiles(unit_name, repeating_unit, leftcap, rightcap, length_resp,)

    # Perform first conformation search using xtb for RESP charge fitting
    structures = qm.poly_conformer_search(mol_resp, out_dir_resp, max_conformers=1000, top_n_MMFF=100, top_n_xtb=10,
                                          epsilon=5, )

    # Perform second conformation search using Gaussian for RESP charge fitting
    sorted_df = qm.conformer_search_gaussian(out_dir_resp, structures, unit_name, charge=0, multiplicity=1, core=32,
                                             memory= '64GB', chk=True, opt_method='B3LYP', opt_basis='6-311+g(d,p)',
                                             dispersion_corr='em=GD3BJ',freq='freq',
                                             solv_model='scrf=(pcm,solvent=generic,read)',
                                             custom_solv='eps=5.0 \nepsinf=2.1',)

    # Perform RESP charge fitting
    qm.calc_resp_gaussian(unit_name, length_resp, out_dir_resp, sorted_df, numconf=5, core=16, memory='64GB', eps=5.0,
                          epsinf=2.1, method='resp', )

    # Generate polymer chain from smiles
    smiles_MD, mol_MD = poly.mol_from_smiles(unit_name, repeating_unit, leftcap, rightcap, length_MD)

    # Build polymer chain
    poly.build_polymer(unit_name, smiles_MD, out_dir_MD, length_MD, opls=False, core = 32)

    # Generate the topology and itp files
    PEO_pdb, nonbonditp_filename, bonditp_filename = MD.gen_gmx_oplsaa(unit_name, out_dir_MD, length_MD)

    # Append the pdb file to the pdb_files list
    pdb_files.append(PEO_pdb)

    # Apply RESP charge to the polymer chain
    qm.apply_chg_to_gmx(unit_name, out_dir_MD, length_MD, repeating_unit, end_repeating, method='resp',
                        target_total_charge=0, correction_factor=1.0)

    # start MD simulation
    # Generate the packmol input file
    poly.gen_packmol_input(out_dir_MD, density, numbers, pdb_files, add_length, packinp_name='pack.inp', packout_name='pack_cell.pdb')

    # Run packmol
    poly.run_packmol(out_dir_MD, input_file = 'pack.inp', output_file = 'pack.out', )

    # Run simulation via gromacs
    MD.run_gmx_annealing(out_dir_MD, compositions, numbers, pdb_files, top_filename, density, add_length,
                         packout_name='pack_cell.pdb', core=64, Tg=False, Tinit=1000, Tfinial=100, )

