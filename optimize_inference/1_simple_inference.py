"""
History:
- Rangsiman Ketkaew [15.04.2026]
"""

from ase import units
from ase.md.langevin import Langevin
from ase.io import read, write
import numpy as np
import time

from mace.calculators import MACECalculator

# Download a pretrained model from 
# https://github.com/ACEsuit/mace-foundations/releases#release-mace_mh_1

calculator = MACECalculator(
    model_path='./mace-mh-1.model', 
    device='cpu',
    head="matpes_r2scan"
    )
init_conf = read('Naphthalene.xyz', '0')
init_conf.set_calculator(calculator)

dyn = Langevin(init_conf, 0.5*units.fs, temperature_K=310, friction=5e-3)
def write_frame():
        dyn.atoms.write('md_3bpa.xyz', append=True)
dyn.attach(write_frame, interval=20)
dyn.run(100)
print("MD finished!")
