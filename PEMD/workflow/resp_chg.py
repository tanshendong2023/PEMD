#!/usr/bin/env python


"""
This script is used to generate the Fluorinated polymer monomer and perform QM calculations,
including the HOMO, LUMO energy and the dipole_moment.
Author: Shendong Tan
Date: 2024-03-30
"""


from PEMD.model import poly
from PEMD.sim import qm
from PEMD.analysis import prop

unit_name = 'PEO'
repeating_unit = '[*]CCO[*]'
leftcap = 'CO[*]'
rightcap = 'C[*]'
length = 5
out_dir = 'PEO_N5'

smiles, mol = poly.mol_from_smiles(unit_name, repeating_unit, leftcap, rightcap, length,)

structures = qm.poly_conformer_search(mol, out_dir, unit_name, length, max_conformers=1000,
                                   top_n_MMFF=100, top_n_xtb=10, epsilon=5, )

sorted_df = qm.conformer_search_gaussian(out_dir, structures, unit_name, charge=0, multiplicity=1, core=32,
                                      memory= '64GB', chk=True, opt_method='B3LYP', opt_basis='6-311+g(d,p)',
                                      dispersion_corr='em=GD3BJ',freq='freq',
                                      solv_model='scrf=(pcm,solvent=generic,read)',
                                      custom_solv='eps=5.0 \nepsinf=2.1',)

qm.calc_resp_gaussian(unit_name, out_dir, sorted_df, core=32, memory='64GB', eps=5.0, epsinf=2.1,)

resp_chg_df = prop.RESP_fit_Multiwfn(unit_name, length, out_dir, method='resp2',)