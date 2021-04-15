import numbers

import torch
from evocraftsearch.spaces import Space


class MultiBinarySpace(Space):
    """
    An n-shape binary space.

    The argument to MultiBinarySpace defines n, which could be a number or a `list` of numbers.

    Example Usage:

    >> self.genome_space = spaces.MultiBinarySpace(5)

    >> self.genome_space.sample()

        array([0,1,0,1,0], dtype =int8)

    >> self.genome_space = spaces.MultiBinarySpace([3,2])

    >> self.genome_space.sample()

        array([[0, 0],
               [0, 1],
               [1, 1]], dtype=int8)

    """

    def __init__(self, n, indpb=1.0):
        self.n = n
        if type(n) in [tuple, list, torch.tensor]:
            input_n = n
        else:
            input_n = (n,)
        if isinstance(indpb, numbers.Number):
            indpb = torch.full(input_n, indpb, dtype=torch.float64)
        self.indpb = torch.as_tensor(indpb, dtype=torch.float64)

        super(MultiBinarySpace, self).__init__(input_n, torch.int8)

    def sample(self):
        return torch.randint(low=0, high=2, size=(self.n,), dtype=self.dtype)

    def mutate(self, x):
        mutate_mask = torch.rand(self.shape) < self.indpb
        x = torch.where(mutate_mask, (~x.bool()).type(self.dtype), x)
        if not self.contains(x):
            return self.clamp(x)
        else:
            return x

    def contains(self, x):
        if isinstance(x, list) or isinstance(x, tuple):
            x = torch.tensor(x)  # Promote list to array for contains check
        if self.shape != x.shape:
            return False
        return ((x == 0) | (x == 1)).all()

    def clamp(self, x):
        # TODO?
        return x

    def __repr__(self):
        return "MultiBinarySpace({})".format(self.n)

    def __eq__(self, other):
        return isinstance(other, MultiBinarySpace) and self.n == other.n
