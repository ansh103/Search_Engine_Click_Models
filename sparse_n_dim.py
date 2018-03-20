import numpy as np
from numbers import Number

class Ndsparse:
    """
    N-dimensional sparse matrix.
    entries in the form of dict of positions and values in matrix
    key as N-tuple of positions (i,k,...)
    val is value at position
    d stands for dimension
    """

    def __init__(self, *args):
        """
        Constructor
        Ndsparse(scalar)
        Ndsparse(dict of (pos):val pairs, optional list of dims for shape)
        Ndsparse(other Ndsparse)
        Ndsparse(numpy.array)
        Ndsparse(nested lists)
        """
        # Blank Ndsparse
        if len(args) == 0:
            self.entries = {}
            self.d = 0
            self.shape = ()

        # From a single scalar
        elif isinstance(args[0], (int, float, complex)):
            self.entries = {(): args[0]}
            self.d = 0
            self.shape = ()

        # From dict of pos,val pairs
        #   Make sure a new dict is produced for the Ndsparse
        elif args[0].__class__.__name__ == 'dict' or args[0].__class__.__name__ == 'Ndsparse':
            if args[0].__class__.__name__ == 'Ndsparse':
                entries = args[0].entries
            else:
                entries = args[0]

            self.entries = {}
            ctr = 0
            for pos, val in entries.items():
                if ctr == 0:
                    d = len(pos)
                if not all(isinstance(x, int) for x in pos):
                    raise IndexError('Position indices must be integers')
                if len(pos) != d:
                    raise IndexError('Position index dimension mismatch')
                if not isinstance(val, Number):
                    raise ValueError('Values must be numbers')
                self.entries[pos] = val
                ctr += 1

            self.d = d

            if len(args) > 1:
                self.shape = args[1]
            else:
                self.shape = get_entries_shape(entries)

        # From numpy array or list of lists (of lists...) dense format
        #   1st dim = rows, 2nd dim = cols, 3rd dim = pages, ...
        #   Uses a numpy array as an intermediate when constructing from lists for convenience
        #   Note that for now, values are converted to floats
        elif args[0].__class__.__name__ == 'ndarray' or args[0].__class__.__name__ == 'list':
            if args[0].__class__.__name__ == 'list':
                array = np.array(args[0])
            else:
                array = args[0]

            self.entries = {}
            it = np.nditer(array, flags=['multi_index'])
            while not it.finished:
                self.entries[it.multi_index] = float(it[0])
                it.iternext()
            self.shape = array.shape
            self.d = len(self.shape)

        # Catch unsupported initialization
        else:
            raise TypeError("Unknown type for Ndsparse construction.")

        # Cleanup
        self.remove_zeros()

    def copy(self):
        """
        Copy "constructor"
        """
        return Ndsparse(self.entries)

    def __repr__(self):
        """
        String representation of Ndsparse class
        """
        rep = [''.join([str(self.d), '-d sparse tensor with ', str(self.nnz()), ' nonzero entries\n'])]
        poss = list(self.entries.keys())
        poss.sort()
        for pos in poss:
            rep.append(''.join([str(pos), '\t', str(self.entries[pos]), '\n']))
        return ''.join(rep)

    def nnz(self):
        """
        Number of nonzero entries. Number of indexed entries if no explicit 0's allowed.
        """
        return len(self.entries)

    def merge_positions(self, other):
        """
        Return (overlap, self_free, other_free)
            overlap: set of tuples of positions where self and other overlap
            self_free: set of tuples of positions where only self is nonzero
            other_free: set of tuples of positions where only other is nonzero
        """
        self_keys = set(self.entries.keys())
        other_keys = set(other.entries.keys())
        overlap = self_keys & other_keys
        self_free = self_keys.difference(other_keys)
        other_free = other_keys.difference(self_keys)
        return overlap, self_free, other_free

    def remove_zeros(self):
        """
        Remove explicit 0 entries in Ndsparse matrix
        """
        new_entries = {}
        for pos, val in self.entries.items():
            if val != 0:
                new_entries[pos] = val
        self.entries = new_entries

    def __getitem__(self, index):
        """Get value at tuple item"""
        if len(index) != self.d:
            raise IndexError('Wrong number of indices specified')
        for i, ind in enumerate(index):
            if ind > self.shape[i] or ind < 0:
                raise IndexError('%i-th index is out of bounds' % (i,))

        try:
            return self.entries[index]
        except KeyError:
            return 0

    def __setitem__(self, index, value):
        """Set value at tuple """
        if len(index) != self.d:
            raise IndexError('Wrong number of indices specified')
        for i, ind in enumerate(index):
            if ind > self.shape[i] or ind < 0:
                raise IndexError('%i-th index is out of bounds' % (i,))

        if value == 0:  # Special case adds structural 0
            del self.entries[index]
        else:
            self.entries[index] = value

    def to_np(self):
        """Convert to dense numpy.array"""
        array = np.zeros(self.shape)
        for pos, val in self.entries.items():
            array[pos] = val
        return array

    def __eq__(self, other):
        """
        Test equality of 2 Ndsparse objects by value. Must have the same nonzero elements, rank, and dimensions.
        """
        if self.d == other.d and self.shape == other.shape and self.entries == other.entries:
            return True
        else:
            return False

    def __add__(self, other):
        """
        Element-wise addition of self + other.
        """
        # Error handling: make sure Dims are same
        overlap, self_free, other_free = self.merge_positions(other)
        out = {}

        for pos in overlap:
            out[pos] = self.entries[pos] + other.entries[pos]
        for pos in self_free:
            out[pos] = self.entries[pos]
        for pos in other_free:
            out[pos] = other.entries[pos]

        return Ndsparse(out, self.shape)

    def __sub__(self, other):
        """
        Element-wise subtraction of self - other.
        """
        # Error handling: make sure Dims are same
        overlap, self_free, other_free = self.merge_positions(other)
        out = {}

        for pos in overlap:
            out[pos] = self.entries[pos] - other.entries[pos]
        for pos in self_free:
            out[pos] = self.entries[pos]
        for pos in other_free:
            out[pos] = -other.entries[pos]

        return Ndsparse(out, self.shape)

    def __mul__(self, other):
        """
        Element-wise multiplication of self .* other.
        """
        # Error handling: make sure Dims are same
        overlap, self_free, other_free = self.merge_positions(other)
        out = {}

        for pos in overlap:
            out[pos] = self.entries[pos] * other.entries[pos]

        return Ndsparse(out, self.shape)

    def transpose(self, permutation=None):
        """
        Transpose Ndsparse matrix in place
        permutation: tuple of new indices
        Matrix starts out as (0,1,...,N) and can be transposed according to 
           the permutation (N,1,....0) or whatever, with N! possible permutations
        Note indexing starts at 0
        """
        if permutation is None:
            if self.d == 2:
                permutation = (1, 0)
            else:
                raise ValueError('A permutation must be supplied for ndim > 2')

        if len(permutation) != self.d:
            raise ValueError('The permutation must match the matrix dimensions')

        out = {}
        for key, value in self.entries.items():
            out[permute(key, permutation)] = value
        self.entries = out
        self.shape = permute(self.shape, permutation)

    def reshape(self, shapemat):
        """
        Like the MATLAB reshape. http://www.mathworks.com/help/matlab/ref/reshape.html
        """
        raise NotImplementedError


def permute(vec, permutation):
    """
    Permute vec tuple according to permutation tuple.
    """
    return tuple([vec[permutation[i]] for i in range(len(vec))])


def get_entries_shape(entries):
    """
    Get dimensions corresponding to max indices in entries
    """
    max_inds = [0] * len(next(iter(entries.keys())))
    for pos in entries.keys():
        for i, ind in enumerate(pos):
            if ind > max_inds[i]:
                max_inds[i] = ind
    return tuple([ind + 1 for ind in max_inds])


def matrix_product(mat1, mat2):
    """
    Standard 2-d matrix multiply
    """
    if mat1.d != 2 or mat2.d != 2:
        raise ValueError('Matrices must both be 2-d for usual matrix multiplication')
    return ttt(mat1, (0, -1), mat2, (-1, 1))


def inner_product(mat1, mat2):
    """
    Standard 1-d vector inner product
    """
    if mat1.d != 1 or mat2.d != 1:
        raise ValueError('Matrices must both be 1-d for usual matrix inner product')
    return ttt(mat1, (-1,), mat2, (-1,))


def outer_product(mat1, mat2):
    """
    Standard 1-d vector outer product
    """
    if mat1.d != 1 or mat2.d != 1:
        raise ValueError('Matrices must both be 1-d for usual matrix outer product')
    return ttt(mat1, (0,), mat2, (1,))


def kronecker_product(mat1, mat2):
    """
    Kronecker product of 2-d matrices using ttt and rearranging indices
    """
    if mat1.d != 2 or mat2.d != 2:
        raise ValueError('Matrices must both be 2-d for usual matrix kronecker product')
    prod = ttt(mat1, (0, 1), mat2, (2, 3))

    # Dimensions: self(m x n) (x) other(p x q) = kprod(mp x nq)
    m = mat1.shape[0]
    n = mat1.shape[1]
    p = mat2.shape[0]
    q = mat2.shape[1]

    # Modify indices
    kprod = {}
    for pos, val in prod.entries.items():
        i = p * pos[0] + pos[2] + 1
        j = q * pos[1] + pos[3] + 1
        kprod[(i - 1, j - 1)] = val

    return Ndsparse(kprod, (m * p, n * q))


def ttt(mat1, spec1, mat2, spec2):
    """
    Tensor x tensor generalized multiplication/contraction. Contract along the corresponding negative
    indices in spec1 and spec2, and arrange the product according to the positive indices in spec1 and spec2. Like
    Einstein notation
    Special cases:
        Outer product: Contract on no dims
        Inner product: Contract on all dims (specify order)
    Time complexity: O(mat1.nnz * mat2.nnz)
    """
    # Validate dimensions
    if mat1.d != len(spec1):
        raise ValueError("ndims of 1st tensor ({mat1d}) doesn't match specified dims ({spec1d})"
                         .format(mat1d=mat1.d, spec1d=len(spec1)))
    if mat2.d != len(spec2):
        raise ValueError("ndims of 2nd tensor ({mat2d}) doesn't match specified dims ({spec2d})"
                         .format(mat2d=mat2.d, spec2d=len(spec2)))

    # Validate contracted dims
    con_dims_1 = [i for i in spec1 if i < 0]
    con_dims_2 = [i for i in spec2 if i < 0]
    if set(con_dims_1) != set(con_dims_2):
        raise ValueError('Contracted dims (negative indices) must match in both tensors')

    # Validate kept dims
    keep_dims_1 = [i for i in spec1 if i >= 0]
    keep_dims_2 = [i for i in spec2 if i >= 0]
    keep_dims = keep_dims_1 + keep_dims_2
    if sorted(keep_dims) != list(range(len(keep_dims))):
        raise ValueError('Product dims (positive indices) must contain sequential indices from 0 to remaining dims')

    # Make mapping of contracted dims
    #   Produce 2 lists: the n-th entry from each list is the n-th dim of that tensor to contract together
    con_map_1 = []
    con_map_2 = []
    for dim in con_dims_1:
        con_map_1.append(spec1.index(dim))
        con_map_2.append(spec2.index(dim))

    spec = spec1 + spec2
    all_shape = mat1.shape + mat2.shape
    prod_d = len(keep_dims)
    prod_shape = []
    for i in range(prod_d):
        prod_shape.append(all_shape[spec.index(i)])
    prod_shape = tuple(prod_shape)

    # Accumulate nonzero positions
    terms = []  # list of tuples of (pos tuple, val) to sum

    keep_pos = [0]*prod_d  # keeps track of current pos
    for pos1, val1 in mat1.entries.items():

        con_ind_1s = [pos1[i] for i in con_map_1]  # mat1's contracted indices
        for i, ind in enumerate(pos1):  # mat1's kept indices
            if spec1[i] >= 0:
                keep_pos[spec1[i]] = ind

        for pos2, val2 in mat2.entries.items():

            con_ind_2s = [pos2[i] for i in con_map_2]  # mat2's contracted indices
            for i, ind in enumerate(pos2):  # mat2's kept indices
                if spec2[i] >= 0:
                    keep_pos[spec2[i]] = ind

            if con_ind_1s == con_ind_2s:  # Match entries that share contraction index (including none)
                terms.append((tuple(keep_pos), val1 * val2))
                # (make sure that the keep_pos list is copied into a tuple, instead of its reference being copied)

    # Sum entries
    prod_entries = {}
    for term in terms:
        pos = term[0]
        val = term[1]
        if pos not in prod_entries:
            prod_entries[pos] = val
        else:
            prod_entries[pos] += val

return Ndsparse(prod_entries, prod_shape)
