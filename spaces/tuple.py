from evocraftsearch.spaces import Space


class TupleSpace(Space):
    """
    A tuple (i.e., product) of simpler spaces

    Example usage:
    self.observation_space = spaces.Tuple((spaces.Discrete(2), spaces.Discrete(3)))
    """

    def __init__(self, spaces):
        self.spaces = spaces
        for space in spaces:
            assert isinstance(space, Space), "Elements of the tuple must be instances of evocraftsearch.Space"
        super(TupleSpace, self).__init__(None, None)

    def sample(self):
        return tuple([space.sample() for space in self.spaces])

    def mutate(self, x):
        return tuple([space.mutate(part) for (space, part) in zip(self.spaces, x)])

    def crossover(self, x1, x2):
        child_list = [space.crossover(x1,x2) for (space, x1, x2) in zip(self.spaces, x1, x2)]
        child_1_tuple = tuple()
        child_2_tuple = tuple()
        for (child_1, child_2) in child_list:
            child_1_tuple += (child_1, )
            child_2_tuple += (child_2, )
        return child_1_tuple, child_2_tuple

    def contains(self, x):
        if isinstance(x, list):
            x = tuple(x)  # Promote list to tuple for contains check
        return isinstance(x, tuple) and len(x) == len(self.spaces) and all(
            space.contains(part) for (space, part) in zip(self.spaces, x))

    def clamp(self, x):
        return tuple([space.clamp(x) for space in self.spaces])

    def __repr__(self):
        return "Tuple(" + ", ".join([str(s) for s in self.spaces]) + ")"

    def __getitem__(self, index):
        return self.spaces[index]

    def __len__(self):
        return len(self.spaces)

    def __eq__(self, other):
        return isinstance(other, TupleSpace) and self.spaces == other.spaces
