from unittest import TestCase

import torch
from addict import Dict
from evocraftsearch.spaces import TupleSpace, BoxSpace, DiscreteSpace, MultiDiscreteSpace, MultiBinarySpace, DictSpace
from evocraftsearch.spaces import utils


class TestSpace(TestCase):
    def test_flatdim(self):
        space_flatdim_tuples = [
            (DiscreteSpace(3), 3),
            (BoxSpace(low=0., high=float('inf'), shape=(2, 2)), 4),
            (TupleSpace([DiscreteSpace(5), DiscreteSpace(10)]), 15),
            (TupleSpace(
                [DiscreteSpace(5), BoxSpace(low=torch.tensor([0, 0]), high=torch.tensor([1, 5]), dtype=torch.float32)]),
             7),
            (TupleSpace((DiscreteSpace(5), DiscreteSpace(2), DiscreteSpace(2))), 9),
            (MultiDiscreteSpace([2, 2, 100]), 3),
            (MultiBinarySpace(10), 10),
            (DictSpace({"position": DiscreteSpace(5),
                        "velocity": BoxSpace(low=torch.tensor([0, 0]), high=torch.tensor([1, 5]),
                                             dtype=torch.float32)}), 7),
        ]
        for space, flatdim in space_flatdim_tuples:
            dim = utils.flatdim(space)
            assert dim == flatdim, "Expected {} to equal {}".format(dim, flatdim)

    def test_flatten_space_boxes(self):
        spaces = [
            DiscreteSpace(3),
            BoxSpace(low=0., high=float('inf'), shape=(2, 2)),
            TupleSpace([DiscreteSpace(5), DiscreteSpace(10)]),
            TupleSpace(
                [DiscreteSpace(5), BoxSpace(low=torch.tensor([0, 0]), high=torch.tensor([1, 5]), dtype=torch.float32)]),
            TupleSpace((DiscreteSpace(5), DiscreteSpace(2), DiscreteSpace(2))),
            MultiDiscreteSpace([2, 2, 100]),
            MultiBinarySpace(10),
            DictSpace({"position": DiscreteSpace(5),
                       "velocity": BoxSpace(low=torch.tensor([0, 0]), high=torch.tensor([1, 5]), dtype=torch.float32)}),
        ]

        for space in spaces:
            flat_space = utils.flatten_space(space)
            assert isinstance(flat_space, BoxSpace), "Expected {} to equal {}".format(type(flat_space), BoxSpace)
            flatdim = utils.flatdim(space)
            (single_dim,) = flat_space.shape
            assert single_dim == flatdim, "Expected {} to equal {}".format(single_dim, flatdim)

    def test_flat_space_contains_flat_points(self):
        spaces = [
            DiscreteSpace(3),
            BoxSpace(low=0., high=float('inf'), shape=(2, 2)),
            TupleSpace([DiscreteSpace(5), DiscreteSpace(10)]),
            TupleSpace(
                [DiscreteSpace(5), BoxSpace(low=torch.tensor([0, 0]), high=torch.tensor([1, 5]), dtype=torch.float32)]),
            TupleSpace((DiscreteSpace(5), DiscreteSpace(2), DiscreteSpace(2))),
            MultiDiscreteSpace([2, 2, 100]),
            MultiBinarySpace(10),
            DictSpace({"position": DiscreteSpace(5),
                       "velocity": BoxSpace(low=torch.tensor([0, 0]), high=torch.tensor([1, 5]), dtype=torch.float32)}),
        ]
        for space in spaces:
            some_samples = [space.sample() for _ in range(10)]
            flattened_samples = [utils.flatten(space, sample) for sample in some_samples]
            flat_space = utils.flatten_space(space)
            for i, flat_sample in enumerate(flattened_samples):
                assert flat_sample in flat_space, 'Expected sample #{} {} to be in {}'.format(i, flat_sample,
                                                                                              flat_space)

    def test_flatten_dim(self):
        spaces = [
            DiscreteSpace(3),
            BoxSpace(low=0., high=float('inf'), shape=(2, 2)),
            TupleSpace([DiscreteSpace(5), DiscreteSpace(10)]),
            TupleSpace(
                [DiscreteSpace(5), BoxSpace(low=torch.tensor([0, 0]), high=torch.tensor([1, 5]), dtype=torch.float32)]),
            TupleSpace((DiscreteSpace(5), DiscreteSpace(2), DiscreteSpace(2))),
            MultiDiscreteSpace([2, 2, 100]),
            MultiBinarySpace(10),
            DictSpace({"position": DiscreteSpace(5),
                       "velocity": BoxSpace(low=torch.tensor([0, 0]), high=torch.tensor([1, 5]), dtype=torch.float32)}),
        ]
        for space in spaces:
            sample = utils.flatten(space, space.sample())
            (single_dim,) = sample.shape
            flatdim = utils.flatdim(space)
            assert single_dim == flatdim, "Expected {} to equal {}".format(single_dim, flatdim)

    def test_flatten_roundtripping(self):
        spaces = [
            DiscreteSpace(3),
            BoxSpace(low=0., high=float('inf'), shape=(2, 2)),
            TupleSpace([DiscreteSpace(5), DiscreteSpace(10)]),
            TupleSpace(
                [DiscreteSpace(5), BoxSpace(low=torch.tensor([0, 0]), high=torch.tensor([1, 5]), dtype=torch.float32)]),
            TupleSpace((DiscreteSpace(5), DiscreteSpace(2), DiscreteSpace(2))),
            MultiDiscreteSpace([2, 2, 100]),
            MultiBinarySpace(10),
            DictSpace({"position": DiscreteSpace(5),
                       "velocity": BoxSpace(low=torch.tensor([0, 0]), high=torch.tensor([1, 5]), dtype=torch.float32)}),
        ]
        for space in spaces:
            some_samples = [space.sample() for _ in range(10)]
            flattened_samples = [utils.flatten(space, sample) for sample in some_samples]
            roundtripped_samples = [utils.unflatten(space, sample) for sample in flattened_samples]
            for i, (original, roundtripped) in enumerate(zip(some_samples, roundtripped_samples)):
                assert compare_nested(original, roundtripped), \
                    'Expected sample #{} {} to equal {}'.format(i, original, roundtripped)


def compare_nested(left, right):
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return torch.allclose(left, right)

    elif isinstance(left, Dict) and isinstance(right, Dict):
        res = len(left) == len(right)
        for ((left_key, left_value), (right_key, right_value)) in zip(left.items(), right.items()):
            if not res:
                return False
            res = left_key == right_key and compare_nested(left_value, right_value)
        return res
    elif isinstance(left, (tuple, list)) and isinstance(right, (tuple, list)):
        res = len(left) == len(right)
        for (x, y) in zip(left, right):
            if not res:
                return False
            res = compare_nested(x, y)
        return res
    else:
        return left == right
