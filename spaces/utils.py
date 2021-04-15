import torch
from addict import Dict
from evocraftsearch.spaces import BoxSpace
from evocraftsearch.spaces import DictSpace
from evocraftsearch.spaces import DiscreteSpace
from evocraftsearch.spaces import MultiBinarySpace
from evocraftsearch.spaces import MultiDiscreteSpace
from evocraftsearch.spaces import TupleSpace


def flatdim(space):
    """Return the number of dimensions a flattened equivalent of this space
    would have.

    Accepts a space and returns an integer. Raises ``NotImplementedError`` if
    the space is not defined in ``evocraftsearch.spaces``.
    """
    if isinstance(space, BoxSpace):
        return int(torch.prod(torch.tensor(space.shape)))
    elif isinstance(space, DiscreteSpace):
        return int(space.n)
    elif isinstance(space, TupleSpace):
        return int(sum([flatdim(s) for s in space.spaces]))
    elif isinstance(space, DictSpace):
        return int(sum([flatdim(s) for s in space.spaces.values()]))
    elif isinstance(space, MultiBinarySpace):
        return int(space.n)
    elif isinstance(space, MultiDiscreteSpace):
        return int(torch.prod(torch.tensor(space.shape)))
    else:
        raise NotImplementedError


def flatten(space, x):
    """Flatten a data point from a space.

    This is useful when e.g. points from spaces must be passed to a neural
    network, which only understands flat arrays of floats.

    Accepts a space and a point from that space. Always returns a 1D array.
    Raises ``NotImplementedError`` if the space is not defined in
    ``evocraftsearch.spaces``.
    """
    if isinstance(space, BoxSpace):
        return torch.as_tensor(x, dtype=torch.float32).flatten()
    elif isinstance(space, DiscreteSpace):
        onehot = torch.zeros(space.n, dtype=torch.float32)
        onehot[x] = 1.0
        return onehot
    elif isinstance(space, TupleSpace):
        return torch.cat(
            [flatten(s, x_part) for x_part, s in zip(x, space.spaces)])
    elif isinstance(space, DictSpace):
        return torch.cat(
            [flatten(s, x[key]) for key, s in space.spaces.items()])
    elif isinstance(space, MultiBinarySpace):
        return torch.as_tensor(x).flatten()
    elif isinstance(space, MultiDiscreteSpace):
        return torch.as_tensor(x).flatten()
    else:
        raise NotImplementedError


def unflatten(space, x):
    """Unflatten a data point from a space.

    This reverses the transformation applied by ``flatten()``. You must ensure
    that the ``space`` argument is the same as for the ``flatten()`` call.

    Accepts a space and a flattened point. Returns a point with a structure
    that matches the space. Raises ``NotImplementedError`` if the space is not
    defined in ``evocraftsearch.spaces``.
    """
    if isinstance(space, BoxSpace):
        return torch.as_tensor(x, dtype=torch.float32).reshape(space.shape)
    elif isinstance(space, DiscreteSpace):
        return torch.nonzero(x)[0][0].type(space.dtype)
    elif isinstance(space, TupleSpace):
        dims = torch.tensor([flatdim(s) for s in space.spaces])
        list_flattened = torch.split(x, dims.tolist())
        list_unflattened = [
            unflatten(s, flattened)
            for flattened, s in zip(list_flattened, space.spaces)
        ]
        return tuple(list_unflattened)
    elif isinstance(space, DictSpace):
        dims = torch.tensor([flatdim(s) for s in space.spaces.values()])
        list_flattened = torch.split(x, torch.cumsum(dims, 0)[:-1])
        list_unflattened = [
            (key, unflatten(s, flattened))
            for flattened, (key, s) in zip(list_flattened, space.spaces.items())
        ]
        return Dict(list_unflattened)
    elif isinstance(space, MultiBinarySpace):
        return torch.as_tensor(x).reshape(space.shape)
    elif isinstance(space, MultiDiscreteSpace):
        return torch.as_tensor(x).reshape(space.shape)
    else:
        raise NotImplementedError


def flatten_space(space):
    """Flatten a space into a single ``Box``.

    This is equivalent to ``flatten()``, but operates on the space itself. The
    result always is a `Box` with flat boundaries. The box has exactly
    ``flatdim(space)`` dimensions. Flattening a sample of the original space
    has the same effect as taking a sample of the flattenend space.

    Raises ``NotImplementedError`` if the space is not defined in
    ``morphosearch.spaces``.

    Example::

        >>> box = BoxSpace(0.0, 1.0, shape=(3, 4, 5))
        >>> box
        Box(3, 4, 5)
        >>> flatten_space(box)
        Box(60,)
        >>> flatten(box, box.sample()) in flatten_space(box)
        True

    Example that flattens a discrete space::

        >>> discrete = DiscreteSpace(5)
        >>> flatten_space(discrete)
        Box(5,)
        >>> flatten(box, box.sample()) in flatten_space(box)
        True

    Example that recursively flattens a dict::

        >>> space = DictSpace({"position": DiscreteSpace(2),
        ...               "velocity": BoxSpace(0, 1, shape=(2, 2))})
        >>> flatten_space(space)
        Box(6,)
        >>> flatten(space, space.sample()) in flatten_space(space)
        True
    """
    if isinstance(space, BoxSpace):
        return BoxSpace(space.low.flatten(), space.high.flatten())
    if isinstance(space, DiscreteSpace):
        return BoxSpace(low=0, high=1, shape=(space.n,))
    if isinstance(space, TupleSpace):
        space = [flatten_space(s) for s in space.spaces]
        return BoxSpace(
            low=torch.cat([s.low for s in space]),
            high=torch.cat([s.high for s in space]),
        )
    if isinstance(space, DictSpace):
        space = [flatten_space(s) for s in space.spaces.values()]
        return BoxSpace(
            low=torch.cat([s.low for s in space]),
            high=torch.cat([s.high for s in space]),
        )
    if isinstance(space, MultiBinarySpace):
        return BoxSpace(low=0, high=1, shape=(space.n,))
    if isinstance(space, MultiDiscreteSpace):
        return BoxSpace(
            low=torch.zeros_like(space.nvec),
            high=space.nvec,
            dtype=space.dtype
        )
    raise NotImplementedError
