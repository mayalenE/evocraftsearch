class Space(object):
    """
    Defines the init_space, genome_space and intervention_space of a system
    """

    def __init__(self, shape=None, dtype=None):
        self.shape = None if shape is None else tuple(shape)
        self.dtype = dtype

    def sample(self):
        """
        Randomly sample an element of this space.
        Can be uniform or non-uniform sampling based on boundedness of space."""
        raise NotImplementedError

    def mutate(self, x):
        """
        Randomly mutate an element of this space.
        """
        raise NotImplementedError

    def crossover(self, x1, x2):
        """
        Mate 2 elements of this space
        """
        pass

    def contains(self, x):
        """
        Return boolean specifying if x is a valid
        member of this space
        """
        raise NotImplementedError

    def clamp(self, x):
        """
        Return a valid clamped value of x inside space's bounds
        """
        raise NotImplementedError

    def __contains__(self, x):
        return self.contains(x)
