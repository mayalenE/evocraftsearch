import numbers
from copy import deepcopy
import torch
from evocraftsearch.spaces import Space


class BoxSpace(Space):
    """
    A (possibly unbounded) box in R^n. Specifically, a Box represents the
    Cartesian product of n closed intervals. Each interval has the form of one
    of [a, b], (-oo, b], [a, oo), or (-oo, oo).

    There are two common use cases:

    * Identical bound for each dimension::
        >>> BoxSpace(low=-1.0, high=2.0, shape=(3, 4), dtype=torch.float32)
        Box(3, 4)

    * Independent bound for each dimension::
        >>> BoxSpace(low=torch.tensor([-1.0, -2.0]), high=torch.tensor([2.0, 4.0]), dtype=torch.float32)
        Box(2,)

    """

    def __init__(self, low, high, shape=None, dtype=torch.float32, mutation_mean=0.0, mutation_std=1.0, indpb=1.0):
        assert dtype is not None, 'dtype must be explicitly provided. '
        self.dtype = dtype

        # determine shape if it isn't provided directly
        if shape is not None:
            shape = tuple(shape)
            assert isinstance(low, numbers.Number) or low.shape == shape, "low.shape doesn't match provided shape"
            assert isinstance(high, numbers.Number) or high.shape == shape, "high.shape doesn't match provided shape"
        elif not isinstance(low, numbers.Number):
            shape = low.shape
            assert isinstance(high, numbers.Number) or high.shape == shape, "high.shape doesn't match low.shape"
        elif not isinstance(high, numbers.Number):
            shape = high.shape
            assert isinstance(low, numbers.Number) or low.shape == shape, "low.shape doesn't match high.shape"
        else:
            raise ValueError("shape must be provided or inferred from the shapes of low or high")

        if isinstance(low, numbers.Number):
            low = torch.full(shape, low, dtype=dtype)

        if isinstance(high, numbers.Number):
            high = torch.full(shape, high, dtype=dtype)

        self.shape = shape
        self.low = low.type(self.dtype)
        self.high = high.type(self.dtype)

        # Boolean arrays which indicate the interval type for each coordinate
        self.bounded_below = ~torch.isneginf(self.low)
        self.bounded_above = ~torch.isposinf(self.high)

        # mutation_mean: mean for the gaussian addition mutation
        # mutation_std: std for the gaussian addition mutation
        # indpb – independent probability for each attribute to be mutated.
        if isinstance(mutation_mean, numbers.Number):
            mutation_mean = torch.full(self.shape, mutation_mean, dtype=torch.float64)
        self.mutation_mean = torch.as_tensor(mutation_mean, dtype=torch.float64)
        if isinstance(mutation_std, numbers.Number):
            mutation_std = torch.full(self.shape, mutation_std, dtype=torch.float64)
        self.mutation_std = torch.as_tensor(mutation_std, dtype=torch.float64)
        if isinstance(indpb, numbers.Number):
            indpb = torch.full(self.shape, indpb, dtype=torch.float64)
        self.indpb = torch.as_tensor(indpb, dtype=torch.float64)

        super(BoxSpace, self).__init__(self.shape, self.dtype)

    def is_bounded(self, manner="both"):
        below = torch.all(self.bounded_below)
        above = torch.all(self.bounded_above)
        if manner == "both":
            return below and above
        elif manner == "below":
            return below
        elif manner == "above":
            return above
        else:
            raise ValueError("manner is not in {'below', 'above', 'both'}")

    def sample(self):
        """
        Generates a single random sample inside of the Box.

        In creating a sample of the box, each coordinate is sampled according to
        the form of the interval:

        * [a, b] : uniform distribution
        * [a, oo) : shifted exponential distribution
        * (-oo, b] : shifted negative exponential distribution
        * (-oo, oo) : normal distribution
        """
        high = self.high.type(torch.float64) if self.dtype.is_floating_point else self.high.type(torch.int64) + 1
        sample = torch.empty(self.shape, dtype=torch.float64)

        # Masking arrays which classify the coordinates according to interval
        # type
        unbounded = ~self.bounded_below & ~self.bounded_above
        upp_bounded = ~self.bounded_below & self.bounded_above
        low_bounded = self.bounded_below & ~self.bounded_above
        bounded = self.bounded_below & self.bounded_above

        # Vectorized sampling by interval type
        sample[unbounded] = torch.randn(unbounded[unbounded].shape, dtype=torch.float64)

        sample[low_bounded] = (-torch.rand(low_bounded[low_bounded].shape, dtype=torch.float64)).exponential_() + \
                              self.low[low_bounded]

        sample[upp_bounded] = self.high[upp_bounded] - (
            -torch.rand(upp_bounded[upp_bounded].shape, dtype=torch.float64)).exponential_()

        sample[bounded] = (self.low[bounded] - high[bounded]) * torch.rand(bounded[bounded].shape,
                                                                           dtype=torch.float64) + high[bounded]

        if not self.dtype.is_floating_point:  # integer
            sample = torch.floor(sample)

        return sample.type(self.dtype)

    def mutate(self, x):
        mutate_mask = (torch.rand(self.shape) < self.indpb).type(torch.float64)
        noise = torch.normal(self.mutation_mean, self.mutation_std)
        x = x.type(torch.float64) + mutate_mask * noise
        if not self.dtype.is_floating_point:  # integer
            x = torch.floor(x)
        x = x.type(self.dtype)
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
        return x.shape == self.shape and torch.all(x >= self.low) and torch.all(x <= self.high)

    def clamp(self, x):
        if self.is_bounded(manner="below"):
            x = torch.max(x, torch.as_tensor(self.low, dtype=self.dtype, device=x.device))
        if self.is_bounded(manner="above"):
            x = torch.min(x, torch.as_tensor(self.high, dtype=self.dtype, device=x.device))
        return x

    def __repr__(self):
        return "BoxSpace({}, {}, {}, {})".format(self.low.min(), self.high.max(), self.shape, self.dtype)

    def __eq__(self, other):
        return isinstance(other, BoxSpace) and (self.shape == other.shape) and torch.allclose(self.low,
                                                                                              other.low) and torch.allclose(
            self.high, other.high)
