from unittest import TestCase
import os
from evocraftsearch.evocraft.utils import load_arena, save_arena
class TestEvocraftUtils(TestCase):

    def test_load_save_nbt(self):
        for nbt_idx, nbt_filename in enumerate(os.listdir("/home/mayalen/code/08-EvoCraft/structures")[4:5]):
            nbt_filepath = os.path.join("/home/mayalen/code/08-EvoCraft/structures", nbt_filename)
            if os.path.isfile(nbt_filepath) and '.nbt' in nbt_filepath:
                load_arena(nbt_filepath, arena_bbox=((nbt_idx % 10)*16,4,(nbt_idx // 10)*16,16,16,16))
                save_arena(os.path.join('test_nbt', nbt_filename), arena_bbox=((nbt_idx % 10)*16,4,(nbt_idx // 10)*16,16,16,16), name='main', author='mayalen')
                load_arena(os.path.join('test_nbt', nbt_filename), arena_bbox=(((nbt_idx+1) % 10) * 16, 4, ((nbt_idx+1) // 10) * 16, 16, 16, 16))
