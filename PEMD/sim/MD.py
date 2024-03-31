"""
Polymer model building tools.

Developed by: Tan Shendong
Date: 2024.03.26
"""


import os
import time
import subprocess
import parmed as pmd
from foyer import Forcefield
from simple_slurm import Slurm
from PEMD.model import poly, PEMD_lib
import importlib.resources as pkg_resources


def gen_gmx_oplsaa(unit_name, out_dir, length):

    current_path = os.getcwd()
    relax_polymer_lmp_dir = os.path.join(current_path, out_dir, 'relax_polymer_lmp')

    pdb_filename = None
    file_base = f"{unit_name}_N{length}"

    for file in os.listdir(relax_polymer_lmp_dir):
        if file.endswith(".xyz"):
            xyz_filename = os.path.join(relax_polymer_lmp_dir, f"{file_base}_gmx.xyz")
            pdb_filename = os.path.join(relax_polymer_lmp_dir, f"{file_base}_gmx.pdb")

            PEMD_lib.convert_xyz_to_pdb(xyz_filename, pdb_filename, f'{unit_name}', f'{unit_name}')

    untyped_str = pmd.load_file(pdb_filename, structure=True)

    with pkg_resources.path("PEMD.sim", "oplsaa.xml") as oplsaa_path:
        oplsaa = Forcefield(forcefield_files=str(oplsaa_path))
    typed_str = oplsaa.apply(untyped_str)

    # build directory
    MD_dir =  os.path.join(out_dir, 'MD_dir')
    PEMD_lib.build_dir(MD_dir)

    top_filename = os.path.join(MD_dir, f"{file_base}.top")
    gro_filename = os.path.join(MD_dir, f"{file_base}.gro")

    # Save to any format supported by ParmEd
    typed_str.save(top_filename)
    typed_str.save(gro_filename)

    nonbonditp_filename = os.path.join(MD_dir, f'{unit_name}_nonbonded.itp')
    bonditp_filename = os.path.join(MD_dir, f'{unit_name}_bonded.itp')

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

    return pdb_filename, nonbonditp_filename, bonditp_filename


def run_gmx_workflow(out_dir, compositions, numbers, pdb_files, top_filename, density, add_length, packout_name, core):

    current_path = os.getcwd()
    MD_dir = os.path.join(current_path, out_dir, 'MD_dir')
    os.chdir(MD_dir)

    # convert pdb to gro
    try:
        box_length = poly.calculate_box_size(numbers, pdb_files, density) + add_length
        run_command(['gmx_mpi', 'editconf', '-f', f'{ packout_name}', '-o', 'conf.gro', '-box', f'{box_length}',
                     f'{box_length}', f'{box_length}'])
        print(f"GROMACS convert pdb to gro executed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error executing GROMACS onvert pdb to gro: {e.stderr}")

    # generate top file
    gen_top_file(compositions, numbers, top_filename)

    # generation minimization mdp file
    gen_min_mdp_file()

    # generation nvt mdp file
    gen_nvt_mdp_file(nsteps=500000, nvt_temperature=298, file_name='nvt.mdp', )

    # generation npt mdp file, 1ns
    gen_npt_mdp_file(nsteps=1000000, npt_temperature=898, file_name='npt.mdp', )

    # generation npt anneal mdp file, anneal rate 0.08K/ps
    gen_npt_anneal_mdp_file(nsteps=7500000, file_name='npt_anneal.mdp', )

    # generation npt production mdp file, 200ns
    gen_npt_mdp_file(nsteps=200000000, npt_temperature=298, file_name='npt_production.mdp', )

    # generation slurm file
    slurm = Slurm(J='gromacs',
                  N=1,
                  n=f'{core}',
                  output=f'{MD_dir}/slurm.{Slurm.JOB_ARRAY_MASTER_ID}.out'
                  )
    slurm.add_cmd('module load GROMACS/2021.7-ompi')
    slurm.add_cmd(f'gmx_mpi grompp -f em.mdp -c conf.gro -p {top_filename} -o em.tpr')
    slurm.add_cmd('gmx_mpi mdrun -v -deffnm em')
    slurm.add_cmd(f'gmx_mpi grompp -f nvt.mdp -c em.gro -p {top_filename} -o nvt.tpr')
    slurm.add_cmd('gmx_mpi mdrun -v -deffnm nvt')
    slurm.add_cmd(f'gmx_mpi grompp -f npt.mdp -c nvt.gro -p {top_filename} -o npt.tpr')
    slurm.add_cmd('gmx_mpi mdrun -v -deffnm npt')
    slurm.add_cmd(f'gmx_mpi grompp -f npt_anneal.mdp -c npt.gro -p {top_filename} -o npt_anneal.tpr')
    slurm.add_cmd('gmx_mpi mdrun -v -deffnm npt_anneal')
    slurm.add_cmd(f'gmx_mpi grompp -f npt_production.mdp -c npt_anneal.gro -p {top_filename} -o npt_production.tpr')
    slurm.add_cmd('gmx_mpi mdrun -v -deffnm npt_production')
    job_id = slurm.sbatch()

    os.chdir(current_path)

    while True:
        status = PEMD_lib.get_slurm_job_status(job_id)
        if status in ['COMPLETED', 'FAILED', 'CANCELLED']:
            print("MD simulation finish, executing the XX task...")
            break
        else:
            print("MD simulation not finish, waiting...")
            time.sleep(30)


# Define a function to execute commands and capture output
def run_command(command, input_text=None, output_file=None):
    # 确保命令是字符串列表
    if isinstance(command, str):
        command = command.split()

    with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.PIPE, text=True) as process:
        stdout, stderr = process.communicate(input=input_text)

    if output_file and stdout:
        with open(output_file, 'w') as file:
            file.write(stdout)

    return stdout, stderr


# generate top file for MD simulation
def gen_top_file(compositions, numbers, top_filename):
    file_contents = "; gromcs generation top file\n"
    file_contents += "; Created by PEMD\n\n"

    file_contents += "[ defaults ]\n"
    file_contents += ";nbfunc  comb-rule   gen-pairs   fudgeLJ  fudgeQQ\n"
    file_contents += "1        3           yes         0.5      0.5\n\n"

    file_contents += ";LOAD atomtypes\n"
    file_contents += "[ atomtypes ]\n"

    for com in compositions:
        file_contents += f'#include "{com}_non_bond.itp"\n'
        file_contents += f'#include "{com}_bond.itp"\n\n'

    file_contents += "[ system ]\n"
    file_contents += ";name "
    for i in compositions:
        file_contents += f"{i}"

    file_contents += "\n\n"

    file_contents += "[ molecules ]\n"
    for com, num in zip(compositions, numbers):
        file_contents += f"{com} {num}\n"

    file_contents += "\n"

    # write to file
    with open(top_filename, 'w') as file:
        file.write(file_contents)
    print(f"Top file generation successful：{top_filename}")


# generation minimization mdp file
def gen_min_mdp_file(file_name = 'em.mdp'):
    file_contents = ";em.mdp - used as input into grompp to generate em.tpr\n"
    file_contents += "; Created by PEMD\n\n"

    file_contents += "integrator      = steep\n"
    file_contents += "nsteps          = 50000\n"
    file_contents += "emtol           = 1000.0\n"
    file_contents += "emstep          = 0.01\n\n"

    file_contents = "; Parameters describing how to find the neighbors of each atom and how to calculate the interactions\n"
    file_contents += "nstlist         = 1\n"
    file_contents += "cutoff-scheme   = Verlet\n"
    file_contents += "ns_type         = grid\n"
    file_contents += "rlist           = 1.0\n"
    file_contents += "coulombtype     = PME\n"
    file_contents += "rcoulomb        = 1.0\n"
    file_contents += "rvdw            = 1.0\n"
    file_contents += "pbc             = xyz\n"
    file_contents += "cutoff-scheme   = Verlet\n\n"

    # write to file
    with open(file_name, 'w') as file:
        file.write(file_contents)
    print(f"Minimization mdp file generation successful：{file_name}")


# generation nvt mdp file
def gen_nvt_mdp_file(nsteps, nvt_temperature, file_name = 'nvt.mdp', ):
    file_contents = ";nvt.mdp - used as input into grompp to generate nvt.tpr\n"
    file_contents += "; Created by PEMD\n\n"

    file_contents += "; RUN CONTROL PARAMETERS\n"
    file_contents += "integrator            = md\n"
    file_contents += "dt                    = 0.001 \n"
    file_contents += f"nsteps                = {nsteps}\n"
    file_contents += "comm-mode             = Linear\n\n"

    file_contents += "; OUTPUT CONTROL OPTIONS\n"
    file_contents += "nstxout               = 10000\n"
    file_contents += "nstvout               = 10000\n"
    file_contents += "nstfout               = 10000\n"
    file_contents += "nstlog                = 10000\n"
    file_contents += "nstenergy             = 10000\n"
    file_contents += "nstxout-compressed    = 10000\n\n"

    file_contents += "; NEIGHBORSEARCHING PARAMETERS\n"
    file_contents += "cutoff-scheme         = verlet\n"
    file_contents += "ns_type               = grid\n"
    file_contents += "nstlist               = 20\n"
    file_contents += "rlist                 = 1.4\n"
    file_contents += "rcoulomb              = 1.4\n"
    file_contents += "rvdw                  = 1.4\n"
    file_contents += "verlet-buffer-tolerance = 0.005\n\n"

    file_contents += "; OPTIONS FOR ELECTROSTATICS AND VDW\n"
    file_contents += "coulombtype           = PME\n"
    file_contents += "vdw_type              = PME\n"
    file_contents += "fourierspacing        = 0.15\n"
    file_contents += "pme_order             = 4\n"
    file_contents += "ewald_rtol            = 1e-05\n\n"

    file_contents += "; OPTIONS FOR WEAK COUPLING ALGORITHMS\n"
    file_contents += "tcoupl                = v-rescale\n"
    file_contents += "tc-grps               = System\n"
    file_contents += "tau_t                 = 1.0\n"
    file_contents += f"ref_t                 = {nvt_temperature}\n"
    file_contents += "Pcoupl                = no\n"
    file_contents += "Pcoupltype            = isotropic\n"
    file_contents += "tau_p                 = 1.0\n"
    file_contents += "compressibility       = 4.5e-5\n"
    file_contents += "ref_p                 = 1.0\n\n"

    file_contents += "; GENERATE VELOCITIES FOR STARTUP RUN\n"
    file_contents += "gen_vel               = no\n\n"

    file_contents += "; OPTIONS FOR BONDS\n"
    file_contents += "constraints           = hbonds\n"
    file_contents += "constraint_algorithm  = lincs\n"
    file_contents += "unconstrained_start   = no\n"
    file_contents += "shake_tol             = 0.00001\n"
    file_contents += "lincs_order           = 4\n"
    file_contents += "lincs_warnangle       = 30\n"
    file_contents += "morse                 = no\n"
    file_contents += "lincs_iter            = 2\n"

    # write to file
    with open(file_name, 'w') as file:
        file.write(file_contents)
    print(f"NVT mdp file generation successful：{file_name}")


# generation npt mdp file
def gen_npt_mdp_file(nsteps, npt_temperature, file_name = 'npt.mdp', ):
    file_contents = ";npt.mdp - used as input into grompp to generate npt.tpr\n"
    file_contents += "; Created by PEMD\n\n"

    file_contents += "; RUN CONTROL PARAMETERS\n"
    file_contents += "integrator            = md\n"
    file_contents += "dt                    = 0.001 \n"
    file_contents += f"nsteps                = {nsteps}\n"
    file_contents += "comm-mode             = Linear\n\n"

    file_contents += "; OUTPUT CONTROL OPTIONS\n"
    file_contents += "nstxout               = 10000\n"
    file_contents += "nstvout               = 10000\n"
    file_contents += "nstfout               = 10000\n"
    file_contents += "nstlog                = 10000\n"
    file_contents += "nstenergy             = 10000\n"
    file_contents += "nstxout-compressed    = 10000\n\n"

    file_contents += "; NEIGHBORSEARCHING PARAMETERS\n"
    file_contents += "cutoff-scheme         = verlet\n"
    file_contents += "ns_type               = grid\n"
    file_contents += "nstlist               = 20\n"
    file_contents += "rlist                 = 1.4\n"
    file_contents += "rcoulomb              = 1.4\n"
    file_contents += "rvdw                  = 1.4\n"
    file_contents += "verlet-buffer-tolerance = 0.005\n\n"

    file_contents += "; OPTIONS FOR ELECTROSTATICS AND VDW\n"
    file_contents += "coulombtype           = PME\n"
    file_contents += "vdw_type              = PME\n"
    file_contents += "fourierspacing        = 0.15\n"
    file_contents += "pme_order             = 4\n"
    file_contents += "ewald_rtol            = 1e-05\n\n"

    file_contents += "; OPTIONS FOR WEAK COUPLING ALGORITHMS\n"
    file_contents += "tcoupl                = v-rescale\n"
    file_contents += "tc-grps               = System\n"
    file_contents += "tau_t                 = 1.0\n"
    file_contents += f"ref_t                 = {npt_temperature}\n"
    file_contents += "Pcoupl                = Berendsen\n"
    file_contents += "Pcoupltype            = isotropic\n"
    file_contents += "tau_p                 = 1.0\n"
    file_contents += "compressibility       = 4.5e-5\n"
    file_contents += "ref_p                 = 1.0\n\n"

    file_contents += "; GENERATE VELOCITIES FOR STARTUP RUN\n"
    file_contents += "gen_vel               = no\n\n"

    file_contents += "; OPTIONS FOR BONDS\n"
    file_contents += "constraints           = hbonds\n"
    file_contents += "constraint_algorithm  = lincs\n"
    file_contents += "unconstrained_start   = no\n"
    file_contents += "shake_tol             = 0.00001\n"
    file_contents += "lincs_order           = 4\n"
    file_contents += "lincs_warnangle       = 30\n"
    file_contents += "morse                 = no\n"
    file_contents += "lincs_iter            = 2\n"

    # write to file
    with open(file_name, 'w') as file:
        file.write(file_contents)
    print(f"NPT mdp file generation successful：{file_name}")


# generation npt anneal mdp file
def gen_npt_anneal_mdp_file(nsteps, file_name = 'npt_anneal.mdp', ):
    file_contents = ";npt_anneal.mdp - used as input into grompp to generate npt_anneal.tpr\n"
    file_contents += "; Created by PEMD\n\n"

    file_contents += "; RUN CONTROL PARAMETERS\n"
    file_contents += "integrator            = md\n"
    file_contents += "dt                    = 0.001 \n"
    file_contents += f"nsteps                = {nsteps}\n"
    file_contents += "comm-mode             = Linear\n\n"

    file_contents += "; OUTPUT CONTROL OPTIONS\n"
    file_contents += "nstxout               = 5000\n"
    file_contents += "nstvout               = 5000\n"
    file_contents += "nstfout               = 5000\n"
    file_contents += "nstlog                = 5000\n"
    file_contents += "nstenergy             = 5000\n"
    file_contents += "nstxout-compressed    = 5000\n\n"

    file_contents += "; NEIGHBORSEARCHING PARAMETERS\n"
    file_contents += "cutoff-scheme         = verlet\n"
    file_contents += "ns_type               = grid\n"
    file_contents += "nstlist               = 20\n"
    file_contents += "rlist                 = 1.4\n"
    file_contents += "rcoulomb              = 1.4\n"
    file_contents += "rvdw                  = 1.4\n"
    file_contents += "verlet-buffer-tolerance = 0.005\n\n"

    file_contents += "; OPTIONS FOR ELECTROSTATICS AND VDW\n"
    file_contents += "coulombtype           = PME\n"
    file_contents += "vdw_type              = PME\n"
    file_contents += "fourierspacing        = 0.15\n"
    file_contents += "pme_order             = 4\n"
    file_contents += "ewald_rtol            = 1e-05\n\n"

    file_contents += "; OPTIONS FOR WEAK COUPLING ALGORITHMS\n"
    file_contents += "tcoupl                = v-rescale\n"
    file_contents += "tc-grps               = System\n"
    file_contents += "tau_t                 = 1.0\n"
    file_contents += "ref_t                 = 298.15\n"
    file_contents += "Pcoupl                = Berendsen\n"
    file_contents += "Pcoupltype            = isotropic\n"
    file_contents += "tau_p                 = 1.0\n"
    file_contents += "compressibility       = 4.5e-5\n"
    file_contents += "ref_p                 = 1.0\n\n"

    file_contents += "; Simulated annealing\n"
    file_contents += "annealing             = single\n"
    file_contents += "annealing-npoints     = 2\n"
    file_contents += "annealing-time        = 0 7500000\n"
    file_contents += "annealing-temp        = 898 298\n\n"

    file_contents += "; GENERATE VELOCITIES FOR STARTUP RUN\n"
    file_contents += "gen_vel               = no\n\n"

    file_contents += "; OPTIONS FOR BONDS\n"
    file_contents += "constraints           = hbonds\n"
    file_contents += "constraint_algorithm  = lincs\n"
    file_contents += "unconstrained_start   = no\n"
    file_contents += "shake_tol             = 0.00001\n"
    file_contents += "lincs_order           = 4\n"
    file_contents += "lincs_warnangle       = 30\n"
    file_contents += "morse                 = no\n"
    file_contents += "lincs_iter            = 2\n"

    # write to file
    with open(file_name, 'w') as file:
        file.write(file_contents)
    print(f"NPT anneal mdp file generation successful：{file_name}")



























