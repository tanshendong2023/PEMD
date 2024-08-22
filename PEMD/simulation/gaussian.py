# Copyright (c) 2024. PEMD developers. All rights reserved.
# Distributed under the terms of the MIT License.

# ******************************************************************************
# Module Docstring
# ******************************************************************************

import os

class PEMDGaussian():
    def __init__(
            self,
            work_dir,
            xyz_filename,
            outfile_headname,
            epsilon=80.4,
            chg=0,
            mult=1
    ):

        self.work_dir = work_dir
        self.xyz_filename = xyz_filename
        self.xyz_filepath = os.path.join(work_dir, xyz_filename)
        self.epsilon = epsilon
        self.chg = chg
        self.mult = mult
        self.outfile_headname = outfile_headname

