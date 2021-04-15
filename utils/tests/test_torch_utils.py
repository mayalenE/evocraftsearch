from unittest import TestCase

import matplotlib.pyplot as plt
from evocraftsearch.utils.torch_utils import get_regions_masks


class Test(TestCase):
    def test_get_regions_masks(self):
        regions = get_regions_masks((1024, 512), 2, 4, include_out_of_maxradius=True)
        for i in range(8):
            plt.imshow(regions[i].double().to_dense().transpose(0, 1), cmap='gray')
            plt.show()
