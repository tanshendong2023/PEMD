import argparse
import math
def write_mdp_files():
    # 在这里添加用于生成 em.mdp, npt.mdp 等文件的代码
    with open('em.mdp', 'w') as f:
        f.write('; minim.mdp - used as input into grompp to generate em.tpr\n')
        f.write('; Parameters for energy minimization\n\n')
        f.write('title                 = Minimization  ; 标题\n')
        f.write('define                = -DFLEXIBLE    ; 如果需要，定义某些预处理变量\n\n')

        f.write('integrator            = steep         ; 使用最速下降法进行能量极小化\n')
        f.write('emtol                 = 1000.0        ; 停止极小化的能量容差(kJ/mol/nm)\n')
        f.write('emstep                = 0.01          ; 极小化步长\n')
        f.write('nsteps                = 50000         ; 最大步数\n\n')

        f.write('n; 电荷和范德华力参数\n')
        f.write('nstlist               = 1             ; 更新邻居列表的频率\n')
        f.write('cutoff-scheme         = Verlet        ; 截断方案\n')
        f.write('ns_type               = grid          ; 使用格点搜索邻居\n')
        f.write('rlist                 = 1.0           ; 截断半径，单位是纳米\n')
        f.write('coulombtype           = PME           ; 库仑相互作用处理方法\n')
        f.write('rcoulomb              = 1.0           ; 库仑截断半径\n')
        f.write('rvdw                  = 1.0           ; 范德华截断半径\n\n')

        f.write('; 输出控制\n')
        f.write('energygrps            = System        ; 定义能量组\n')

    with open('npt-gas.mdp', 'w') as f:
        f.write('; RUN CONTROL PARAMETERS =\n\n')

        f.write('integrator            = md            ; md intergrator\n')
        f.write('tinit                 = 0             ; starting time for run\n')
        f.write('dt                    = 0.001         ; time step for intergration\n')
        f.write('nsteps                = 10000000      ; max number of steps to intergrate\n')
        f.write('comm-mode             = angular       ; Remove center of mass translation\n\n')

        f.write('; OUTPUT CONTROL OPTIONS =\n')
        f.write('nstxout               = 25000         ; [steps] freq to write coordinates to trajectory\n')
        f.write('nstvout               = 25000         ; [steps] freq to write velocities to trajectory\n')
        f.write('nstfout               = 25000         ; [steps] freq to write forces to trajectory\n')
        f.write('nstlog                = 2000          ; [steps] freq to write energies to log file\n')
        f.write('nstenergy             = 10000         ; group(s) to write to energy file\n\n')

        f.write('; NEIGHBORSEARCHING PARAMETERS =\n')
        f.write('cutoff-scheme         = group         ; Generate a pair list for groups of atoms\n')
        f.write('nstlist               = 0             ; [steps] freq to update neighbor list\n')
        f.write('ns_type               = simple        ; method of updating neighbor list\n')
        f.write('pbc                   = no            ; periodic boundary conditions in all directions\n')
        f.write('rlist                 = 0             ; Cut-off distance for the short-range neighbor list.\n\n')

        f.write('; OPTIONS FOR ELECTROSTATICS AND VDW =\n')
        f.write('coulombtype           = cut-off       ; Particle-Mesh Ewald electrostatics\n')
        f.write('rcoulomb              = 0             ; [nm] distance for Coulomb cut-off\n')
        f.write(
            'vdw_type              = cut-off       ; Twin range cut-offs with neighbor list cut-off rlist and VdW cut-off rvdw, where rvdw = rlist.\n')
        f.write('rvdw                  = 0             ; [nm] distance for LJ cut-off\n')
        f.write('fourierspacing        = 0.15          ; [nm] grid spacing for FFT grid when using PME\n')
        f.write('pme_order             = 4             ; interpolation order for PME, 4 = cubic\n')
        f.write(
            'ewald_rtol            = 1e-05         ; relative strength of Ewald-shifted potential at rcoulomb\n\n')

        f.write('; OPTIONS FOR WEAK COUPLING ALGORITHMS =\n\n')

        f.write('; Temperature coupling\n')
        f.write('tcoupl                = V-rescale\n')
        f.write('tc-grps               = System\n')
        f.write('tau_t                 = 1.0\n')
        f.write('ref_t                 = 298.0\n\n')

        f.write('; Pressure coupling\n')
        f.write(';Pcoupl               = berendsen\n')
        f.write(';Pcoupltype           = isotropic\n')
        f.write(';tau_p                = 1.0\n')
        f.write(';compressibility      = 4.5e-5\n')
        f.write(';ref_p                = 1.01325 ; bar\n\n')

        f.write('; GENERATE VELOCITIES FOR STARTUP RUN =\n')
        f.write('gen_vel               = yes\n')
        f.write('gen_temp              = 298.0\n')
        f.write('gen_seed              = 473529\n\n')

        f.write('; OPTIONS FOR BONDS =\n')
        f.write('constraints           = hbonds\n')
        f.write('constraint_algorithm  = lincs\n')
        f.write('unconstrained_start   = no\n')
        f.write('shake_tol             = 0.00001\n')
        f.write('lincs_order           = 4\n')
        f.write('lincs_warnangle       = 30\n')
        f.write('morse                 = no\n')
        f.write('lincs_iter            = 2\n')

    with open('npt.mdp', 'w') as f:
        f.write('; RUN CONTROL PARAMETERS =\n\n')

        f.write('integrator            = md            ; md integrator\n')
        f.write('tinit                 = 5000          ; [ps] starting time for run\n')
        f.write('dt                    = 0.001         ; [ps] time step for integration\n')
        f.write('nsteps                = 40000000      ; maximum number of steps to integrate\n')
        f.write('comm-mode             = Linear        ; Remove center of mass translation\n\n')

        f.write('; OUTPUT CONTROL OPTIONS =\n')
        f.write('nstxout               = 25000         ; [steps] freq to write coordinates to trajectory\n')
        f.write('nstvout               = 25000         ; [steps] freq to write velocities to trajectory\n')
        f.write('nstfout               = 25000         ; [steps] freq to write forces to trajectory\n')
        f.write('nstlog                = 2000          ; [steps] freq to write energies to log file\n')
        f.write('nstenergy             = 10000         ; group(s) to write to energy file\n')
        f.write('nstxout-compressed    = 10000         ; freq to write coordinates to xtc trajectory\n\n')

        f.write('; NEIGHBORSEARCHING PARAMETERS =\n')
        f.write(
            'cutoff-scheme         = verlet        ; This option has an explicit, exact cut-off at rvdw=rcoulomb.\n')
        f.write('nstlist               = 20            ; [steps] freq to update neighbor list\n')
        f.write('ns_type               = grid          ; method of updating neighbor list\n')
        f.write('pbc                   = xyz           ; periodic boundary conditions in all directions\n')
        f.write('rlist                 = 1.2           ; [nm] cut-off distance for the short-range neighbor list\n')
        f.write(
            'verlet-buffer-tolerance = 0.005       ; sets the maximum allowed error for pair interactions per particle. Indirectly sets rlist\n\n')

        f.write('; OPTIONS FOR ELECTROSTATICS AND VDW =\n')
        f.write('coulombtype           = PME           ; Particle-Mesh Ewald electrostatics\n')
        f.write('rcoulomb              = 1.2           ; [nm] distance for Coulomb cut-off\n')
        f.write('vdw_type              = PME           ; twin-range cut-off with rlist where rvdw >= rlist\n')
        f.write('rvdw                  = 1.2           ; [nm] distance for LJ cut-off\n')
        f.write('fourierspacing        = 0.15          ; [nm] grid spacing for FFT grid when using PME\n')
        f.write('pme_order             = 4             ; interpolation order for PME, 4 = cubic\n')
        f.write(
            'ewald_rtol            = 1e-05         ; relative strength of Ewald-shifted potential at rcoulomb\n\n')

        f.write('; OPTIONS FOR WEAK COUPLING ALGORITHMS =\n')
        f.write('tcoupl                = v-rescale     ; temperature coupling method\n')
        f.write('tc-grps               = System        ; groups to couple seperately to temperature bath\n')
        f.write('tau_t                 = 1.0           ; [ps] time constant for coupling\n')
        f.write('ref_t                 = 298.15        ; reference temperature for coupling\n')
        f.write('Pcoupl                = Berendsen     ; pressure coupling method\n')
        f.write('Pcoupltype            = isotropic     ; pressure coupling in x-y-z directions\n')
        f.write('tau_p                 = 1.0           ; [ps] time constant for coupling\n')
        f.write('compressibility       = 4.5e-5        ; [bar^-1] compressibility\n')
        f.write('ref_p                 = 1.0 ; bar     ; reference pressure for coupling\n\n')

        f.write('; GENERATE VELOCITIES FOR STARTUP RUN =\n')
        f.write('gen_vel               = no      ; velocity generation turned on\n\n')

        f.write('; OPTIONS FOR BONDS =\n')
        f.write('constraints           = hbonds\n')
        f.write('constraint_algorithm  = lincs\n')
        f.write('unconstrained_start   = no\n')
        f.write('shake_tol             = 0.00001\n')
        f.write('lincs_order           = 4\n')
        f.write('lincs_warnangle       = 30\n')
        f.write('morse                 = no\n')
        f.write('lincs_iter            = 2\n\n')

    with open('vis.mdp', 'w') as f:
        f.write('; minim.mdp - used as input into grompp to generate em.tpr\n')
        f.write('; Parameters for energy minimization\n\n')
        f.write('title                 = Minimization  ; 标题\n')
        f.write('define                = -DFLEXIBLE    ; 如果需要，定义某些预处理变量\n\n')

        f.write('; RUN CONTROL PARAMETERS =\n\n')
        f.write('integrator            = md            ; md integrator\n')
        f.write('tinit                 = 5000          ; [ps] starting time for run\n')
        f.write('dt                    = 0.001         ; [ps] time step for integration\n')
        f.write('nsteps                = 10000000      ; maximum number of steps to integrate\n')
        f.write('comm-mode             = Linear        ; Remove center of mass translation\n\n')

        f.write('; OUTPUT CONTROL OPTIONS =\n')
        f.write('nstxout               = 1000\n')
        f.write('nstvout               = 1000\n')
        f.write('nstenergy             = 1000\n')
        f.write('nstlog                = 1000\n')
        f.write('nstcalcenergy         = 1000\n\n')

        f.write('; NEIGHBORSEARCHING PARAMETERS =\n')
        f.write('cutoff-scheme         = verlet        ; explicit cut-off\n')
        f.write('nstlist               = 20            ; freq to update neighbor list\n')
        f.write('ns_type               = grid          ; method of updating neighbor list\n')
        f.write('pbc                   = xyz           ; periodic boundary conditions\n')
        f.write('rlist                 = 1.2           ; cut-off distance for neighbor list\n')
        f.write('verlet-buffer-tolerance = 0.005       ; error for pair interactions\n\n')

        f.write('; OPTIONS FOR ELECTROSTATICS AND VDW =\n')
        f.write('coulombtype           = PME           ; Particle-Mesh Ewald electrostatics\n')
        f.write('rcoulomb              = 1.2           ; distance for Coulomb cut-off\n')
        f.write('vdw_type              = PME           ; twin-range cut-off\n')
        f.write('rvdw                  = 1.2           ; distance for LJ cut-off\n')
        f.write('fourierspacing        = 0.15          ; grid spacing for FFT\n')
        f.write('pme_order             = 4             ; interpolation order for PME\n')
        f.write('ewald_rtol            = 1e-05         ; relative strength of Ewald-shifted potential\n')
        f.write('cos_acceleration      = 0.04          ; amplitude for calculating viscosity\n\n')

        f.write('; OPTIONS FOR WEAK COUPLING ALGORITHMS\n')
        f.write('; Temperature coupling\n')
        f.write('Tcoupl                = nose-hoover\n')
        f.write('nsttcouple            = -1\n')
        f.write('nh-chain-length       = 1\n')
        f.write('; Groups to couple separately\n')
        f.write('tc-grps               = System\n')
        f.write('; Time constant and reference temperature\n')
        f.write('tau_t                 = 0.5\n')
        f.write('ref_t                 = 298.15\n')
        f.write('; Pressure coupling\n')
        f.write('Pcoupl                = no\n')
        f.write('Pcoupltype            = isotropic\n')
        f.write('; Time constant, compressibility and reference P\n')
        f.write('tau_p                 = 0.5\n')
        f.write('compressibility       = 4.5e-5\n')
        f.write('ref_p                 = 1.0\n')
        f.write('refcoord_scaling      = com\n\n')

        f.write('; GENERATE VELOCITIES FOR STARTUP RUN =\n')
        f.write('gen_vel               = no      ; velocity generation turned on\n\n')

        f.write('; OPTIONS FOR BONDS =\n')
        f.write('constraints           = hbonds\n')
        f.write('constraint_algorithm  = lincs\n')
        f.write('unconstrained_start   = no\n')
        f.write('shake_tol             = 0.00001\n')
        f.write('lincs_order           = 4\n')
        f.write('lincs_warnangle       = 30\n')
        f.write('morse                 = no\n')
        f.write('lincs_iter            = 2\n')

    with open('sub_gromacs.sh', 'w') as f:
        f.write("#!/bin/bash\n")
        f.write("#SBATCH -J gromacs\n")
        f.write("#SBATCH -N 1\n")
        f.write("#SBATCH -n 64\n\n")
        f.write("source /home/tanshendong/soft/GROMACS/5.1.4/bin/GMXRC\n")
        f.write("gmx mdrun -v -deffnm npt\n")
def calculate_box_size(density, molar_masses, nmols):
    NA = 6.02214076e23  # Avogadro's number
    num_ions = nmols[0]
    total_mass = num_ions * molar_masses
    volume_per_mole = molar_masses / density  # cm^3/mol
    volume_for_ions = (num_ions / NA) * volume_per_mole  # cm^3
    edge_length_cm = volume_for_ions ** (1 / 3)  # cm
    edge_length_angstrom = edge_length_cm * 1e8  # cm to Angstroms conversion
    return edge_length_angstrom  # Convert to Angstroms

# 新的替换文件第五行的函数
def replace_fifth_line(file_path, edge_length):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    lines[4] = f"CRYST1   {edge_length:.2f} {edge_length:.2f} {edge_length:.2f}  90.00  90.00  90.00 P 1           1\n"

    with open(file_path, 'w') as file:
        file.writelines(lines)

def main():
    parser = argparse.ArgumentParser(description='BoxGo: Generate Packmol input file.')
    parser.add_argument('-b', '--box', default='', help='box length in A ' \
                                                        '(cubic), for non-cubic boxes supply '
                                                        'a,b,c[,alpha,beta,gamma] default box ' \
                                                        'is orthogonal (alpha = beta = gamma = 90.0)')
    parser.add_argument('infiles', nargs='+', help='n1 infile1 [n2 infile2 ...], where n_i are the numbers of molecules defined in infile_i.')
    parser.add_argument('-m', '--molar_masses', type=float, help='Molar mass of each molecule type in g/mol')
    parser.add_argument('-r', '--rho', type=float,  default=0.0, help='Density in mol/L')
    parser.add_argument('-g', action='store_true', help='Generate additional GROMACS files')
    parser.add_argument('-pdb', '--pdb_box', action='store_true',
                        help='Modify the fifth line of the PDB file for box dimensions')

    args = parser.parse_args()

    nmols = [int(n) for n in args.infiles[::2]]  # even elements are numbers of molecules

    files = args.infiles[1::2]  # odd elements are molecule files

    if args.box and args.rho != 0.0:
        raise RuntimeError('supply density or box dimensions, not both')

    if args.box:
        a = b = c = float(args.box)
        alpha = beta = gamma = 90.0
    elif args.rho != 0.0:
        a = b = c = calculate_box_size(args.rho, args.molar_masses, nmols)
    else:
        raise RuntimeError('density or box dimensions need to be supplied')

    with open('packmol.inp', 'w') as file:
        file.write("# created by BoxGo\n")
        file.write("tolerance 2.5\nfiletype pdb\noutput output.pdb\n\n")

        for nmol, file_name in zip(nmols, files):
            file.write(f"structure {file_name}\n  number {nmol}\n")
            file.write(f"  inside box 0.0 0.0 0.0 {a:.2f} {b:.2f} {c:.2f}\nend structure\n\n")

    if args.g:
        write_mdp_files()  # 根据 -g 选项生成额外的文件

    if args.pdb_box:
        # 假设第一个文件是你想修改的pdb文件
        replace_fifth_line('output.pdb', a)  # 用计算出的盒子尺寸修改第一个文件

if __name__ == "__main__":
    main()


