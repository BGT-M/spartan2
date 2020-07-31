import warnings

import numpy as np
from datasketch import MinHash
from datasketch.hashfunc import sha1_hash32

_mersenne_prime = (1 << 61) - 1
_max_hash = (1 << 32) - 1
_hash_range = (1 << 32)

class NeiMinHash(MinHash):
    def __init__(self, num_perm=128, seed=1,
            hashfunc=sha1_hash32,
            hashobj=None, # Deprecated.
            hashvalues=None, permutations=None):
        if hashvalues is not None:
            num_perm = len(hashvalues)
        if num_perm > _hash_range:
            # Because 1) we don't want the size to be too large, and
            # 2) we are using 4 bytes to store the size value
            raise ValueError("Cannot have more than %d number of\
                    permutation functions" % _hash_range)
        self.seed = seed
        # Check the hash function.
        if not callable(hashfunc):
            raise ValueError("The hashfunc must be a callable.")
        self.hashfunc = hashfunc
        # Check for use of hashobj and issue warning.
        if hashobj is not None:
            warnings.warn("hashobj is deprecated, use hashfunc instead.",
                    DeprecationWarning)
        # Initialize hash values
        if hashvalues is not None:
            self.hashvalues = self._parse_hashvalues(hashvalues)
        else:
            self.hashvalues = self._init_hashvalues(num_perm)
        # Initalize permutation function parameters
        if permutations is not None:
            self.permutations = permutations
        else:
            generator = np.random.RandomState(self.seed)
            # Create parameters for a random bijective permutation function
            # that maps a 32-bit hash value to another 32-bit hash value.
            # http://en.wikipedia.org/wiki/Universal_hashing
            a = generator.randint(1, _mersenne_prime, num_perm, dtype=np.uint64)
            b = generator.randint(0, _mersenne_prime, num_perm, dtype=np.uint64)
            self.a, self.b = a, b.reshape(len(b), 1)
            self.permutations = np.vstack([a, b])
        if len(self) != len(self.permutations[0]):
            raise ValueError("Numbers of hash values and permutations mismatch")

    def updates(self, values):
        values = np.array(values, dtype=np.uint64)

        phv = (np.outer(self.a, values) + self.b) % _mersenne_prime
        phv = np.bitwise_and(phv, np.uint64(_max_hash)).min(axis=1)
        self.hashvalues = np.minimum(self.hashvalues, phv, dtype=np.uint64)