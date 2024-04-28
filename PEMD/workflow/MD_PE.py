#!/usr/bin/env python


"""
This script is used to start the molecular dynamics of polymer.
Author: Shendong Tan
Date: 2024-04-22
"""


from PEMD.model import poly
from PEMD.sim import qm, MD
from PEMD.analysis import prop


unit_name = 'PEO'
repeating_unit = '[*]CCO[*]'
leftcap = 'C[*]'
rightcap = 'C[*]'
length_resp = 10                     # for resp charge fitting via show chain polymer
out_dir_resp = 'PEO_N10'             # for resp charge fitting via show chain polymer
length_MD = 50                       # for MD
out_dir_MD = 'PEO_N50'               # for MD
end_repeating = 2                    # keep the charge of polymer end group
density = 0.8                        # system density
add_length = 25                      # unit: Å


model_info = {
    'polymer': {
        'compound': 'PEO',
        'resname': 'MOL',
        'numbers': 20,
        'scale': 1.0,
    },
    'Li_cation': {
        'compound': 'Li',
        'resname': 'LIP',
        'numbers': 50,
        'smiles':'[Li+]',
        'scale': 0.75,
    },
    'salt_anion':{
        'compound': 'TFSI',
        'resname': 'NSC',
        'numbers': 50,
        'smiles': 'C(F)(F)(F)(F)S(=O)(=O)[O-]',
        'scale': 0.75,
    },
    'solvent':{
        'compound': 'EC',
        'resname': 'EC',
        'numbers': 50,
        'smiles': 'CCOC(=O)C',
        'scale': 1.0,
    },
    }


if __name__ == '__main__':
    # 1. obtain RESP charge fitting result
    # Generate polymer monomer from smiles
    smiles_resp, mol_resp = poly.mol_from_smiles(unit_name, repeating_unit, leftcap, rightcap, length_resp,)

    # Perform first conformation search using xtb for RESP charge fitting
    structures = qm.poly_conformer_search(mol_resp, out_dir_resp, core=32, max_conformers=1000, top_n_MMFF=100,
                                          top_n_xtb=10, epsilon=5, )

    # Perform second conformation search using Gaussian for RESP charge fitting
    sorted_df = qm.conformer_search_gaussian(out_dir_resp, structures, unit_name, charge=0, multiplicity=1, core=32,
                                             memory= '64GB', chk=True, opt_method='B3LYP', opt_basis='6-311+g(d,p)',
                                             dispersion_corr='em=GD3BJ',freq='freq',
                                             solv_model='scrf=(pcm,solvent=generic,read)',
                                             custom_solv='eps=5.0 \nepsinf=2.1',)

    # Perform RESP charge fitting
    qm.calc_resp_gaussian(unit_name, length_resp, out_dir_resp, sorted_df, numconf=5, core=32, memory='64GB', eps=5.0,
                          epsinf=2.1, method='resp2', )

    # 2. start MD simulation for sigle-chain polymer
    # Generate polymer chain from smiles
    smiles_MD, mol_MD = poly.mol_from_smiles(unit_name, repeating_unit, leftcap, rightcap, length_MD)

    # Build polymer chain
    poly.build_polymer(unit_name, smiles_MD, out_dir_MD, length_MD, opls=False, core = 32)

    # Generate the topology and itp files
    nonbonditp_filename, bonditp_filename = MD.gen_gmx_oplsaa(unit_name, out_dir_MD, length_MD, model_info,)

    # Apply RESP charge to the polymer chain
    qm.apply_chg_to_gmx(unit_name, out_dir_resp, out_dir_MD, length_resp, length_MD, repeating_unit, end_repeating,
                        method='resp2', target_total_charge=0, correction_factor=1.0)

    # 3. production the force filed for the small molecules

    MD.gen_oplsaa_ff_molecule(model_info, out_dir='MD_dir')

    # 4. start MD simulation for amorphous polymer system
    # Generate the packmol input file
    poly.gen_packmol_input(unit_name, length_MD, density, model_info, add_length, out_dir='MD_dir',
                           packinp_name='pack.inp', packout_name='pack_cell.pdb',)

    # Run packmol
    poly.run_packmol(out_dir_MD, input_file='pack.inp', output_file='pack.out', )

    # Pre-run gromacs
    MD.pre_run_gmx(unit_name, length_MD, model_info, density, add_length, out_dir='MD_dir',
                   packout_name='pack_cell.pdb', core=64, T_target=333, top_filename='topol.top',
                   module_soft='GROMACS/2021.7-ompi', output_str='pre_eq')

    # Run gromacs for production simulation, 200 ns
    MD.run_gmx_prod(out_dir='MD_dir', core=64, T_target=333, input_str='pre_eq', top_filename='topol.top',
                    module_soft='GROMACS/2021.7-ompi', nstep_ns=200, output_str='nvt_prod')

    # post-analysis for the production simulation





