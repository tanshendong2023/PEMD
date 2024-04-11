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


def pre_run_gmx(out_dir, compositions, numbers, pdb_files, top_filename, density, add_length, packout_name, core,
                T_target, module_soft='GROMACS/2021.7-ompi',output_str='pre_eq'):

    current_path = os.getcwd()
    MD_dir = os.path.join(current_path, out_dir, 'MD_dir')
    os.chdir(MD_dir)

    box_length = (poly.calculate_box_size(numbers, pdb_files, density) + add_length)/10  # A to nm

    # generate top file
    gen_top_file(compositions, numbers, top_filename)

    # generation minimization mdp file
    gen_min_mdp_file(file_name='em.mdp')

    # generation nvt mdp file
    gen_nvt_mdp_file(nsteps_nvt=200000,
                     nvt_temperature=f'{T_target}',
                     file_name='nvt.mdp', )

    # generation npt anneal mdp file, anneal rate 0.05K/ps
    T_high = T_target + 500
    gen_npt_anneal_mdp_file(nsteps_annealing=22000000,
                            npt_temperature=f'{T_target}',
                            annealing_npoints=5,
                            annealing_time='0 1000 11000 21000 22000',
                            annealing_temp=f'{T_target} {T_target} {T_high} {T_target} {T_target}',
                            file_name='npt_anneal.mdp', )

    # generation nvt mdp file
    gen_nvt_mdp_file(nsteps_nvt=1000000,
                     nvt_temperature=f'{T_target}',
                     file_name='nvt_eq.mdp', )

    # generation slurm file
    slurm = Slurm(J='gromacs',
                  N=1,
                  n=f'{core}',
                  output=f'{MD_dir}/slurm.{Slurm.JOB_ARRAY_MASTER_ID}.out'
                  )

    slurm.add_cmd(f'module load {module_soft}')
    slurm.add_cmd(f'gmx_mpi editconf -f {packout_name} -o conf.gro -box {box_length} {box_length} {box_length}')
    slurm.add_cmd(f'gmx_mpi grompp -f em.mdp -c conf.gro -p {top_filename} -o em.tpr')
    slurm.add_cmd('gmx_mpi mdrun -v -deffnm em')
    slurm.add_cmd(f'gmx_mpi grompp -f nvt.mdp -c em.gro -p {top_filename} -o nvt.tpr')
    slurm.add_cmd('gmx_mpi mdrun -v -deffnm nvt')
    slurm.add_cmd(f'gmx_mpi grompp -f npt_anneal.mdp -c nvt.gro -p {top_filename} -o npt_anneal.tpr')
    slurm.add_cmd('gmx_mpi mdrun -v -deffnm npt_anneal')
    slurm.add_cmd(f'gmx_mpi grompp -f nvt_eq.mdp -c npt_anneal.gro -p {top_filename} -o {output_str}.tpr')
    slurm.add_cmd(f'gmx_mpi mdrun -v -deffnm {output_str}')

    job_id = slurm.sbatch()

    os.chdir(current_path)

    while True:
        status = PEMD_lib.get_slurm_job_status(job_id)
        if status in ['COMPLETED', 'FAILED', 'CANCELLED']:
            print("MD simulation finish, executing the XX task...")
            break
        else:
            print("MD simulation not finish, waiting...")
            time.sleep(10)


def run_gmx_prod(out_dir, top_filename, core, T_target, input_str, module_soft='GROMACS/2021.7-ompi',
                 nstep_ns=200, output_str='nvt_prod'):

    current_path = os.getcwd()
    MD_dir = os.path.join(current_path, out_dir, 'MD_dir')
    os.chdir(MD_dir)

    # generation nvt production mdp file, 200ns
    nstep=nstep_ns*1000000
    gen_nvt_mdp_file(nsteps_nvt=f'{nstep}',
                     nvt_temperature=f'{T_target}',
                     file_name='nvt_prod.mdp',)

    # generation slurm file
    slurm = Slurm(J='gromacs',
                  N=1,
                  n=f'{core}',
                  output=f'{MD_dir}/slurm.{Slurm.JOB_ARRAY_MASTER_ID}.out'
                  )

    slurm.add_cmd(f'module load {module_soft}')
    slurm.add_cmd(f'gmx_mpi grompp -f nvt_prod.mdp -c {input_str}.gro -p {top_filename} -o {output_str}.tpr')
    slurm.add_cmd(f'gmx_mpi mdrun -v -deffnm {output_str}')

    job_id = slurm.sbatch()

    os.chdir(current_path)

    while True:
        status = PEMD_lib.get_slurm_job_status(job_id)
        if status in ['COMPLETED', 'FAILED', 'CANCELLED']:
            print("MD simulation finish, executing the XX task...")
            break
        else:
            print("MD simulation not finish, waiting...")
            time.sleep(10)


def run_gmx_tg(out_dir, top_filename, input_str, out_str, module_soft='GROMACS/2021.7-ompi', anneal_rate=0.01,
               core=64, Tinit=1000, Tfinal=100,):

    current_path = os.getcwd()
    MD_dir = os.path.join(current_path, out_dir, 'MD_dir')
    os.chdir(MD_dir)

    # heating up for anneal, heating rate 0.05K/ps
    dT = Tinit - Tfinal
    total_time_ps = int(dT/0.05) # unit: ps
    gen_npt_anneal_mdp_file(nsteps_annealing=int(dT/0.05*1000),
                            npt_temperature=f'{Tfinal}',
                            annealing_npoints=2,
                            annealing_time=f'0 {total_time_ps}',
                            annealing_temp=f'{Tfinal} {Tinit}',
                            file_name='npt_heating.mdp', )

    # 每个温度点之间的温度差
    delta_temp = 20  # 每20K进行一次退火

    # 退火所需的时间（ps），每升高1K需要的时间是1/0.01=100ps，20K需要2000ps, unit:k/ps
    time_per_delta_temp = delta_temp / anneal_rate

    # 温度列表和时间点列表
    temperatures = [Tinit]
    time_points = [0]  # 从0开始

    # 当前温度和时间
    current_temp = Tinit
    current_time = 0

    while current_temp > Tfinal:
        # 添加当前温度，进行2ns的恒定温度模拟
        temperatures.append(current_temp)
        # print(temperatures)
        current_time += 2000  # 2ns的模拟
        # print(current_time)
        time_points.append(int(current_time))
        # print(time_points)

        # 退火到下一个温度点
        if current_temp - delta_temp >= Tfinal:  # 确保不会低于最终温度
            current_temp -= delta_temp
            temperatures.append(current_temp)
            current_time += time_per_delta_temp  # 退火过程
            time_points.append(int(current_time))
        else:
            break

    temperatures.append(Tfinal)
    current_time += 2000  # 最终温度再进行2ns的恒定温度模拟
    time_points.append(int(current_time))

    # 生成npt退火mdp文件
    gen_npt_anneal_mdp_file(nsteps_annealing=int(current_time * 1000),  # 总步数
                            npt_temperature=f'{Tinit}',  # 最终温度
                            annealing_npoints=len(temperatures),  # 温度点的数量
                            annealing_time=' '.join(str(t) for t in time_points),  # 每个温度点的开始时间，单位：步
                            annealing_temp=' '.join(str(temp) for temp in temperatures),  # 对应的温度点
                            file_name='npt_anneal_tg.mdp')

    # generation slurm file
    slurm = Slurm(J='gromacs',
                  N=1,
                  n=f'{core}',
                  output=f'{MD_dir}/slurm.{Slurm.JOB_ARRAY_MASTER_ID}.out'
                  )

    slurm.add_cmd(f'module load {module_soft}')
    slurm.add_cmd(f'gmx_mpi grompp -f npt_heating.mdp -c {input_str}.gro -p {top_filename} -o npt_heating.tpr')
    slurm.add_cmd('gmx_mpi mdrun -v -deffnm npt_heating')
    slurm.add_cmd(f'gmx_mpi grompp -f npt_anneal_tg.mdp -c npt_heating.gro -p {top_filename} -o {out_str}.tpr')
    slurm.add_cmd(f'gmx_mpi mdrun -v -deffnm {out_str}')

    job_id = slurm.sbatch()

    os.chdir(current_path)

    while True:
        status = PEMD_lib.get_slurm_job_status(job_id)
        if status in ['COMPLETED', 'FAILED', 'CANCELLED']:
            print("MD simulation finish, executing the XX task...")
            break
        else:
            print("MD simulation not finish, waiting...")
            time.sleep(10)


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
        file_contents += f'#include "{com}_nonbonded.itp"\n'
    file_contents += "\n"
    for com in compositions:
        file_contents += f'#include "{com}_bonded.itp"\n'
    file_contents += "\n"

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
    file_contents = "; em.mdp - used as input into grompp to generate em.tpr\n"
    file_contents += "; Created by PEMD\n\n"

    file_contents += "integrator      = steep\n"
    file_contents += "nsteps          = 50000\n"
    file_contents += "emtol           = 1000.0\n"
    file_contents += "emstep          = 0.01\n\n"

    file_contents += ("; Parameters describing how to find the neighbors of each atom and how to calculate the "
                      "interactions\n")
    file_contents += "nstlist         = 1\n"
    file_contents += "cutoff-scheme   = Verlet\n"
    file_contents += "ns_type         = grid\n"
    file_contents += "rlist           = 1.0\n"
    file_contents += "coulombtype     = PME\n"
    file_contents += "rcoulomb        = 1.0\n"
    file_contents += "rvdw            = 1.0\n"
    file_contents += "pbc             = xyz\n"

    file_contents += "; output control is on\n"
    file_contents += "energygrps      = System\n"

    # write to file
    with open(file_name, 'w') as file:
        file.write(file_contents)
    print(f"Minimization mdp file generation successful：{file_name}")


# generation nvt mdp file
def gen_nvt_mdp_file(nsteps_nvt, nvt_temperature, file_name = 'nvt.mdp', ):
    file_contents = "; nvt.mdp - used as input into grompp to generate nvt.tpr\n"
    file_contents += "; Created by PEMD\n\n"

    file_contents += "; RUN CONTROL PARAMETERS\n"
    file_contents += "integrator            = md\n"
    file_contents += "dt                    = 0.001 \n"
    file_contents += f"nsteps                = {nsteps_nvt}\n"
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
def gen_npt_mdp_file(nsteps_npt, npt_temperature, file_name = 'npt.mdp', ):
    file_contents = "; npt.mdp - used as input into grompp to generate npt.tpr\n"
    file_contents += "; Created by PEMD\n\n"

    file_contents += "; RUN CONTROL PARAMETERS\n"
    file_contents += "integrator            = md\n"
    file_contents += "dt                    = 0.001 \n"
    file_contents += f"nsteps                = {nsteps_npt}\n"
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
def gen_npt_anneal_mdp_file(nsteps_annealing, npt_temperature, annealing_npoints, annealing_time, annealing_temp,
                            file_name):

    file_contents = "; npt_anneal.mdp - used as input into grompp to generate npt_anneal.tpr\n"
    file_contents += "; Created by PEMD\n\n"

    file_contents += "; RUN CONTROL PARAMETERS\n"
    file_contents += "integrator            = md\n"
    file_contents += "dt                    = 0.001 \n"
    file_contents += f"nsteps                = {nsteps_annealing}\n"
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
    file_contents += f"ref_t                 = {npt_temperature}\n"
    file_contents += "Pcoupl                = Berendsen\n"
    file_contents += "Pcoupltype            = isotropic\n"
    file_contents += "tau_p                 = 1.0\n"
    file_contents += "compressibility       = 4.5e-5\n"
    file_contents += "ref_p                 = 1.0\n\n"

    file_contents += "; Simulated annealing\n"
    file_contents += "annealing             = single\n"
    file_contents += f"annealing-npoints     = {annealing_npoints}\n"
    file_contents += f"annealing-time        = {annealing_time}\n"
    file_contents += f"annealing-temp        = {annealing_temp}\n\n"

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



























