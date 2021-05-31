from evocraftsearch import OutputFitness
import torch
from evocraftsearch.evocraft.minecraft_pb2 import *
import time

class DisplacementFitness(OutputFitness):

    @staticmethod
    def default_config():
        default_config = OutputFitness.default_config()
        return default_config

    def __init__(self, client, arena_bbox=(0,4,0,32,32,32), block_orientation="NORTH", blocks_list=['AIR', 'CLAY', 'SLIME', 'PISTON', 'STICKY_PISTON', 'REDSTONE_BLOCK'], config={}, **kwargs):
        super().__init__(config=config, **kwargs)
        self.client = client
        self.arena_bbox = arena_bbox # test arena to spawn blocks and read back state
        self.block_orientation = block_orientation
        self.blocks_list = blocks_list

    def clear_arena(self):
        self.client.fillCube(FillCubeRequest(
            cube=Cube(
                min=Point(x=self.arena_bbox[0]-20, y=4, z=self.arena_bbox[2]-20),
                max=Point(x=self.arena_bbox[0] + self.arena_bbox[3] + 20, y=self.arena_bbox[1] + self.arena_bbox[4], z=self.arena_bbox[2] + self.arena_bbox[5] + 20)
            ),
            type=AIR
        ))

    def map(self, observations, reduction="mean"):
        """
            Maps the observations of a system to an embedding vector
        """
        last_potential = observations.potentials[-1]
        n_channels, SZ, SY, SX = last_potential.shape

        coords = []
        feats = []
        for i in range(SZ):
            for j in range(SY):
                for k in range(SX):
                    block_id = last_potential[:, i, j, k].cpu().detach().argmax()
                    if block_id > 0:
                        coords.append(torch.tensor([i, j, k], dtype=torch.float64))
                        feats.append(block_id)

        if len(coords) > 0:
            blocks = []
            for block_idx, block_pos in enumerate(coords):
                # translate block pos to fit in area
                world_x = int(block_pos[0] + self.arena_bbox[0])
                world_y = int(block_pos[1] + self.arena_bbox[1])
                world_z = int(block_pos[2] + self.arena_bbox[2])

                if (world_x < self.arena_bbox[0] + self.arena_bbox[3]) and (world_y < self.arena_bbox[1] + self.arena_bbox[4]) and (world_z < self.arena_bbox[2] + self.arena_bbox[5]):
                    cur_pos = Point(x=world_x, y=world_y, z=world_z)
                    cur_block = Block(position=cur_pos, type=self.blocks_list[feats[block_idx]], orientation='NORTH')
                    blocks.append(cur_block)

            center_of_mass = torch.stack(coords).mean(0)
            # Draw the loaded blocks
            self.clear_arena()
            time.sleep(5)
            self.client.spawnBlocks(Blocks(blocks=blocks))

            # Read back the state after some time
            time.sleep(5)
            arena_blocks = self.client.readCube(Cube(
                min=Point(x=self.arena_bbox[0] - 20, y=self.arena_bbox[1], z=self.arena_bbox[2] - 20),
                max=Point(x=self.arena_bbox[0] + self.arena_bbox[3] + 20, y=self.arena_bbox[1] + self.arena_bbox[4] - 1, z=self.arena_bbox[2] + self.arena_bbox[5] + 20)
            ))

            coords = []
            for block in arena_blocks.blocks:
                if BlockType.keys()[block.type] != "AIR":
                    i = int(block.position.x - self.arena_bbox[0])
                    j = int(block.position.y - self.arena_bbox[1])
                    k = int(block.position.z - self.arena_bbox[2])
                    coords.append(torch.tensor([i, j, k], dtype=torch.float64))

            if len(coords) > 0:
                new_center_of_mass = torch.stack(coords).mean(0)
                fitness = torch.norm(new_center_of_mass - center_of_mass)

            else:
                fitness = torch.tensor(100.0) # went outside of arena means found a flying machine!


        else:
            fitness = torch.tensor(0.0)

        return fitness

