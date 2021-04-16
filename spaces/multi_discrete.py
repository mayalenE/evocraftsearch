import numbers
from copy import deepcopy
import torch
from evocraftsearch.spaces import Space


class MultiDiscreteSpace(Space):
    """
    - The multi-discrete space consists of a series of discrete spaces with different number of possible instances in eachs
    - Can be initialized as

        MultiDiscreteSpace([ 5, 2, 2 ])

    """

    def __init__(self, nvec, mutation_mean=0.0, mutation_std=1.0, indpb=1.0):

        """
        nvec: vector of counts of each categorical variable
        """
        assert (torch.tensor(nvec) > 0).all(), 'nvec (counts) have to be positive'
        self.nvec = torch.as_tensor(nvec, dtype=torch.int64)
        self.mutation_std = mutation_std

        # mutation_mean: mean for the gaussian addition mutation
        # mutation_std: std for the gaussian addition mutation
        # indpb – independent probability for each attribute to be mutated.
        if isinstance(mutation_mean, numbers.Number):
            mutation_mean = torch.full(self.nvec.shape, mutation_mean, dtype=torch.float64)
        self.mutation_mean = torch.as_tensor(mutation_mean, dtype=torch.float64)
        if isinstance(mutation_std, numbers.Number):
            mutation_std = torch.full(self.nvec.shape, mutation_std, dtype=torch.float64)
        self.mutation_std = torch.as_tensor(mutation_std, dtype=torch.float64)
        if isinstance(indpb, numbers.Number):
            indpb = torch.full(self.nvec.shape, indpb, dtype=torch.float64)
        self.indpb = torch.as_tensor(indpb, dtype=torch.float64)

        super(MultiDiscreteSpace, self).__init__(self.nvec.shape, torch.int64)

    def sample(self):
        return (torch.rand(self.nvec.shape) * self.nvec).type(self.dtype)

    def mutate(self, x):
        mutate_mask = torch.rand(self.shape) < self.indpb
        noise = torch.normal(self.mutation_mean, self.mutation_std)
        x = x.type(torch.float64) + mutate_mask * noise
        x = torch.floor(x).type(self.dtype)
        if not self.contains(x):
            return self.clamp(x)
        else:
            return x

    def crossover(self, x1, x2):
        child_1 = deepcopy(x1)
        child_2 = deepcopy(x2)
        if self.shape != ():
            crossover_mask = (torch.rand(self.shape) < self.indpb)
            switch_parent_mask = crossover_mask & torch.randint(2, self.shape, dtype=torch.bool)
            # mix parents
            child_1[switch_parent_mask] = x2[switch_parent_mask]
            child_2[switch_parent_mask] = x1[switch_parent_mask]
        return child_1, child_2

    def contains(self, x):
        if isinstance(x, list):
            x = torch.tensor(x)  # Promote list to array for contains check
        # if nvec is uint32 and space dtype is uint32, then 0 <= x < self.nvec guarantees that x
        # is within correct bounds for space dtype (even though x does not have to be unsigned)
        return x.shape == self.shape and (0 <= x).all() and (x < self.nvec).all()

    def clamp(self, x):
        x = torch.max(x, torch.as_tensor(0, dtype=self.dtype, device=x.device))
        x = torch.min(x, torch.as_tensor(self.nvec - 1, dtype=self.dtype, device=x.device))
        return x

    def __repr__(self):
        return "MultiDiscreteSpace({})".format(self.nvec)

    def __eq__(self, other):
        return isinstance(other, MultiDiscreteSpace) and torch.all(self.nvec == other.nvec)
