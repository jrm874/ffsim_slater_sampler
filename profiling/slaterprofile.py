import ffsim
from ffsim.states.bitstring import BitstringType
import numpy as np


norb = 1000
nelec = 100

shots = 3

rng = np.random.default_rng(1234)
rotation = ffsim.random.random_unitary(norb, seed=rng)
rdm = ffsim.slater_determinant_rdms(norb, range(nelec), rotation, rank=1)
samples = ffsim.sample_slater(
            rdm, norb, nelec, shots=shots, bitstring_type=BitstringType.INT, seed=rng
        )


