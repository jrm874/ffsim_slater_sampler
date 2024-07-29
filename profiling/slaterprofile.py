import ffsim
from ffsim.states.bitstring import BitstringType
import numpy as np


norb = 100
nelec = 50

shots = 1000

rng = np.random.default_rng(1234)
rotation = ffsim.random.random_unitary(norb, seed=rng)
rdm = ffsim.slater_determinant_rdms(norb, range(nelec), rotation, rank=1)

from ffsim.states.slater import _autoregressive_slater_vec

samples = _autoregressive_slater_vec(rdm, norb, nelec, shots, seed = rng)

#samples = ffsim.sample_slater(
#            rdm, norb, nelec, shots=shots, bitstring_type=BitstringType.INT, seed=rng
#        )


