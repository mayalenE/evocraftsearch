from copy import copy
from unittest import TestCase

import torch
from evocraftsearch.spaces import TupleSpace, BoxSpace, DiscreteSpace, MultiDiscreteSpace, MultiBinarySpace, DictSpace


class TestSpace(TestCase):
    def test_contain(self):
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
            sample_1 = space.sample()
            mutate_1 = space.mutate(sample_1)
            sample_2 = space.sample()
            mutate_2 = space.mutate(sample_2)
            assert space.contains(sample_1) and space.contains(mutate_1)
            assert space.contains(sample_2) and space.contains(mutate_2)

    def test_equality(self):

        spaces = [
            DiscreteSpace(3),
            BoxSpace(low=torch.tensor([-10, 0]), high=torch.tensor([10, 10]), dtype=torch.float32),
            BoxSpace(low=-float('inf'), high=float('inf'), shape=(1, 3)),
            TupleSpace([DiscreteSpace(5), DiscreteSpace(10)]),
            TupleSpace(
                [DiscreteSpace(5), BoxSpace(low=torch.tensor([0, 0]), high=torch.tensor([1, 5]), dtype=torch.float32)]),
            TupleSpace((DiscreteSpace(5), DiscreteSpace(2), DiscreteSpace(2))),
            MultiDiscreteSpace([2, 2, 100]),
            MultiBinarySpace(6),
            DictSpace({"position": DiscreteSpace(5),
                       "velocity": BoxSpace(low=torch.tensor([0, 0]), high=torch.tensor([1, 5]), dtype=torch.float32)}),
        ]

        for space in spaces:
            space1 = space
            space2 = copy(space)
            assert space1 == space2, "Expected {} to equal {}".format(space1, space2)

    def test_inequality(self):
        space_tuples = [
            (DiscreteSpace(3), DiscreteSpace(4)),
            (MultiDiscreteSpace([2, 2, 100]), MultiDiscreteSpace([2, 2, 8])),
            (MultiBinarySpace(8), MultiBinarySpace(7)),
            (BoxSpace(low=torch.tensor([-10, 0]), high=torch.tensor([10, 10]), dtype=torch.float32),
             BoxSpace(low=torch.tensor([-10, 0]), high=torch.tensor([10, 9]), dtype=torch.float32)),
            (BoxSpace(low=-float("inf"), high=0., shape=(2, 1)),
             BoxSpace(low=0., high=float("inf"), shape=(2, 1))),
            (TupleSpace([DiscreteSpace(5), DiscreteSpace(10)]), TupleSpace([DiscreteSpace(1), DiscreteSpace(10)])),
            (DictSpace({"position": DiscreteSpace(5)}), DictSpace({"position": DiscreteSpace(4)})),
            (DictSpace({"position": DiscreteSpace(5)}), DictSpace({"speed": DiscreteSpace(5)})),
        ]

        for space_tuple in space_tuples:
            space1, space2 = space_tuple
            assert space1 != space2, "Expected {} != {}".format(space1, space2)

    def test_sample(self):
        spaces = [
            DiscreteSpace(5),
            BoxSpace(low=0, high=255, shape=(2,), dtype=torch.uint8),
            BoxSpace(low=-float('inf'), high=float('inf'), shape=(3, 3)),
            BoxSpace(low=1., high=float('inf'), shape=(3, 3)),
            BoxSpace(low=-float('inf'), high=2., shape=(3, 3)),
        ]

        for space in spaces:
            n_trials = 100
            samples = torch.stack([space.sample() for _ in range(n_trials)])
            if isinstance(space, BoxSpace):
                if space.is_bounded():
                    expected_mean = (space.high + space.low) / 2
                elif space.is_bounded("below"):
                    expected_mean = 1 + space.low
                elif space.is_bounded("above"):
                    expected_mean = -1 + space.high
                else:
                    expected_mean = torch.full(space.shape, 0.)
            elif isinstance(space, DiscreteSpace):
                expected_mean = torch.tensor(space.n / 2)
            else:
                raise NotImplementedError
            torch.testing.assert_allclose(expected_mean, samples.type(expected_mean.dtype).mean(0),
                                          atol=3.0 * samples.type(expected_mean.dtype).std(), rtol=0.0)

    def test_mutate(self):
        spaces = [
            DiscreteSpace(5, mutation_mean=2.0, mutation_std=1.0, indpb=0.9),
            BoxSpace(low=0, high=255, shape=(2,), dtype=torch.uint8, mutation_mean=0.0, mutation_std=1.0, indpb=0.9),
            BoxSpace(low=0, high=255, shape=(2,), dtype=torch.float64, mutation_mean=0.0, mutation_std=1.0, indpb=0.9),
            MultiDiscreteSpace(nvec=(5, 2, 2), mutation_mean=(0.0, 0.4, 0.0), mutation_std=(1.0, 1.4, 0.0),
                               indpb=(0.9, 0.8, 1.0)),
            MultiBinarySpace(n=10, indpb=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9))
        ]

        for space in spaces:
            n_trials = 100
            samples = torch.stack([space.sample() for _ in range(n_trials)])
            mutated_samples = torch.stack([space.mutate(sample) for sample in samples])
            assert samples.dtype == mutated_samples.dtype
            assert samples.shape == mutated_samples.shape
            for mutated_sample in mutated_samples:
                assert space.contains(mutated_sample)
            delta_mutation = mutated_samples.double() - samples.double()
            if not isinstance(space, MultiBinarySpace):
                assert torch.all((delta_mutation.mean(0) - space.mutation_mean).abs() <= 3.0 * space.mutation_std)
            else:
                assert torch.all(((mutated_samples - samples).abs().sum(0).double() - (
                            n_trials * space.indpb)).abs() <= n_trials / 10)

    def test_class_inequality(self):
        space_tuples = [
            (DiscreteSpace(5), MultiBinarySpace(5)),
            (BoxSpace(low=torch.tensor([-10, 0]), high=torch.tensor([10, 10]), dtype=torch.float32),
             MultiDiscreteSpace([2, 2, 8])),
            (BoxSpace(low=0, high=255, shape=(64, 64, 3), dtype=torch.uint8),
             BoxSpace(low=0, high=255, shape=(32, 32, 3), dtype=torch.uint8)),
            (DictSpace({"position": DiscreteSpace(5)}), TupleSpace([DiscreteSpace(5)])),
            (DictSpace({"position": DiscreteSpace(5)}), DiscreteSpace(5)),
            (TupleSpace((DiscreteSpace(5),)), DiscreteSpace(5)),
            (BoxSpace(low=torch.tensor([-float('inf'), 0.]), high=torch.tensor([0., float('inf')])),
             BoxSpace(low=torch.tensor([-float('inf'), 1.]), high=torch.tensor([0., float('inf')])))
        ]
        for space_tuple in space_tuples:
            assert space_tuple[0] == space_tuple[0]
            assert space_tuple[1] == space_tuple[1]
            assert space_tuple[0] != space_tuple[1]
            assert space_tuple[1] != space_tuple[0]

    def test_bad_space_calls(self):
        space_fns = [
            lambda: DictSpace(space1='abc'),
            lambda: DictSpace({'space1': 'abc'}),
            lambda: TupleSpace(['abc'])
        ]

        for space_fn in space_fns:
            with self.assertRaises(AssertionError):
                space_fn()
