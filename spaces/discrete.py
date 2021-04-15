import torch
from evocraftsearch.spaces import Space


class DiscreteSpace(Space):
    r"""A discrete space in :math:`\{ 0, 1, \\dots, n-1 \}`.
    /!\ mutation is gaussian by default: please create custom space inheriting from discrete space for custom mutation functions

    Example::

        >>> DiscreteSpace(2)

    """

    def __init__(self, n, mutation_mean=0.0, mutation_std=1.0, indpb=1.0):
        assert n >= 0
        self.n = n

        # mutation_mean: mean for the gaussian addition mutation
        # mutation_std: std for the gaussian addition mutation
        # indpb – independent probability for each attribute to be mutated.
        self.mutation_mean = torch.as_tensor(mutation_mean, dtype=torch.float64)
        self.mutation_std = torch.as_tensor(mutation_std, dtype=torch.float64)
        self.indpb = torch.as_tensor(indpb, dtype=torch.float64)
        super(DiscreteSpace, self).__init__((), torch.int64)

    def sample(self):
        return torch.randint(self.n, ())

    def mutate(self, x):
        mutate_mask = torch.rand(self.shape) < self.indpb
        noise = torch.normal(self.mutation_mean, self.mutation_std, ())
        x = x.type(torch.float64) + mutate_mask * noise
        x = torch.floor(x).type(self.dtype)
        if not self.contains(x):
            return self.clamp(x)
        else:
            return x

    def contains(self, x):
        if isinstance(x, int):
            as_int = x
        elif not x.dtype.is_floating_point and (x.shape == ()):  # integer or size 0
            as_int = int(x)
        else:
            return False
        return 0 <= as_int < self.n

    def clamp(self, x):
        x = torch.max(x, torch.as_tensor(0, dtype=self.dtype, device=x.device))
        x = torch.min(x, torch.as_tensor(self.n - 1, dtype=self.dtype, device=x.device))
        return x

    def __repr__(self):
        return "DiscreteSpace(%d)" % self.n

    def __eq__(self, other):
        return isinstance(other, DiscreteSpace) and self.n == other.n
