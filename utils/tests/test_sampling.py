from unittest import TestCase

import torch
from evocraftsearch.utils import sample_value


class TestSampling(TestCase):
    def test_sample_value(self):

        num_of_test = 10

        # check scalars
        for i in range(num_of_test):
            val = sample_value(1)
            assert val == 1 and len(val.shape) == 0

            val = sample_value(1.3)
            assert val == 1.3 and len(val.shape) == 0

            val = sample_value(-1.3)
            assert val == -1.3 and len(val.shape) == 0

        # check sampling of continuous (min, max)
        for i in range(num_of_test):
            min_max = (0, 1)
            val = sample_value(min_max)
            assert min_max[0] <= val <= min_max[1] and len(val.shape) == 0

            min_max = (-1, 1)
            val = sample_value(min_max)
            assert min_max[0] <= val <= min_max[1] and len(val.shape) == 0

            min_max = (40, 50)
            val = sample_value(min_max)
            assert min_max[0] <= val <= min_max[1] and len(val.shape) == 0

        # check sampling of continuous ('continous', min, max)
        for i in range(num_of_test):
            min_max = ('continous', 0, 1)
            val = sample_value(min_max)
            assert min_max[1] <= val <= min_max[2] and int(val) != val and len(val.shape) == 0

            min_max = ('continous', -1, 1)
            val = sample_value(min_max)
            assert min_max[1] <= val <= min_max[2] and int(val) != val and len(val.shape) == 0

            min_max = ('continous', 40, 50)
            val = sample_value(min_max)
            assert min_max[1] <= val <= min_max[2] and int(val) != val and len(val.shape) == 0

        # check sampling of discrete ('discrete', min, max)
        for i in range(num_of_test):
            min_max = ('discrete', 0, 1)
            val = sample_value(min_max)
            assert val in [0, 1] and len(val.shape) == 0

            min_max = ('discrete', -1, 1)
            val = sample_value(min_max)
            assert val in [-1, 0, 1] and len(val.shape) == 0

            min_max = ('discrete', 40, 50)
            val = sample_value(min_max)
            assert val in [40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50] and len(val.shape) == 0

        # check sampling of item from list
        for i in range(num_of_test):
            lst = [0, 1, 2, 3]
            val = sample_value(lst)
            assert val in lst and len(val.shape) == 0

            lst = ['a', 'bb', 'ccc', 'dddd']
            val = sample_value(lst)
            assert val in lst

            lst = [3, '22', (33, 34)]
            val = sample_value(lst)
            assert val in lst

        for i in range(num_of_test):
            descr = dict()
            descr['type'] = 'continuous'
            descr['min'] = 0
            descr['max'] = 1
            val = sample_value(descr)
            assert descr['min'] <= val <= descr['max'] and int(val) != val and len(val.shape) == 0

            descr['type'] = 'continuous'
            descr['min'] = -10
            descr['max'] = 10
            val = sample_value(descr)
            assert descr['min'] <= val <= descr['max'] and int(val) != val and len(val.shape) == 0

            descr['type'] = 'discrete'
            descr['min'] = 0
            descr['max'] = 1
            val = sample_value(descr)
            assert val in [0, 1] and len(val.shape) == 0

            descr['type'] = 'discrete'
            descr['min'] = 10
            descr['max'] = 15
            val = sample_value(descr)
            assert val in [10, 11, 12, 13, 14, 15] and len(val.shape) == 0

        # userdefined function
        my_func = lambda a, b: torch.tensor(a - b)
        descr = ('function', my_func, 100, 50)
        val = sample_value(descr)
        assert val == descr[2] - descr[3] and len(val.shape) == 0
