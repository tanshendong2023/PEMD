#!/usr/bin/env python

from PEMD.model import poly, PEMD_lib
from PEMD.sim import qm
from PEMD.analysis import property

unit_name = 'PEM'
repeating_unit = '[*]CCOC(=O)CC(=O)O[*]'
leftcap = 'C[*]'
rightcap = 'C[*]'
length = 1
out_dir = 'PEM_N1'

smiles, mol = poly.mol_from_smiles(unit_name, repeating_unit, leftcap, rightcap, length)
qm.conformation_search(mol, unit_name, out_dir, length, numconf=10, charge =0, multiplicity=1, memory='64GB', core = '32', 
                       chk = True, opt_method='B3LYP', opt_basis='6-311+g(d,p)', dispersion_corr = 'em=GD3BJ', 
                       freq = 'freq', solv_model = '', custom_solv='')
homo_energy,lumo_energy = property.extract_homo_lumo(unit_name, out_dir,length)





