import grpc
from exputils.seeding import set_seed
import time
import torch
from torch import nn
import pytorchneat
import neat

from evocraftsearch.evocraft import minecraft_pb2_grpc
from evocraftsearch.evocraft.minecraft_pb2 import *
channel = grpc.insecure_channel('localhost:5001')
client = minecraft_pb2_grpc.MinecraftServiceStub(channel)


class CPPNSystem(nn.Module):

    def __init__(self, env_size=(32,32,32), cppn_n_passes=2, tau=0.01, device=torch.device('cpu')):

        nn.Module.__init__(self)
        self.env_size = env_size
        self.n_blocks = 6
        self.R = 1
        self.cppn_n_passes = cppn_n_passes
        self.tau = tau
        self.device = device


    def reset(self, genome, neat_config):
        self.step_idx = 0
        self.cppn = pytorchneat.rnn.RecurrentNetwork.create(genome, neat_config)
        self.cppn.to(self.device)
        self.state = torch.zeros(self.env_size, dtype=torch.int64) #H,W,D matrix between 0 and N_blocks-1 (253)
        self.state[self.env_size[0]//2-1:self.env_size[0]//2+1, self.env_size[1]//2-1:self.env_size[1]//2+1, self.env_size[2]//2-1:self.env_size[2]//2+1] = 1

        obs = self.state
        return obs

    def step(self, action=None):

        cppn_input = torch.zeros(self.env_size + ((2*self.R+1)**3, self.n_blocks,))
        neighbor_idx = 0
        scattered_inds = torch.zeros(self.env_size + ((2*self.R+1)**3, )).long()
        for i in range(-self.R, self.R+1):
            for j in range(-self.R, self.R + 1):
                for k in range(-self.R, self.R + 1):
                    shifted_state = torch.roll(self.state, shifts=(i,j,k), dims=(0,1,2))
                    scattered_inds[:,:,:,neighbor_idx] = shifted_state
                    neighbor_idx += 1
        cppn_input.scatter_(-1, scattered_inds.unsqueeze(-1), 1)
        cppn_input = cppn_input.view(-1, (2*self.R+1)**3 * self.n_blocks)
        cppn_output = self.cppn.activate(cppn_input, self.cppn_n_passes)
        blocktype_logits = cppn_output
        print(blocktype_logits)
        blocktype = nn.functional.gumbel_softmax(blocktype_logits, hard=True, tau=self.tau).argmax(-1)
        print(blocktype.sum())
        self.state = blocktype.reshape(self.env_size)
        self.step_idx += 1

        obs = self.state
        reward = 0.0
        done = False
        info = ''
        return obs, reward, done, info

    def forward(self):
        obs, reward, done, info = self.step()
        return obs

    def render(self, mode="human"):
        raise NotImplementedError



if __name__ == '__main__':

    set_seed(1)

    cppn_system = CPPNSystem()

    neat_config = neat.Config(
        pytorchneat.selfconnectiongenome.SelfConnectionGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        'test_neat_cppnsystem.cfg'
    )

    genome = neat_config.genome_type(0)
    genome.configure_new(neat_config.genome_config)
    obs = cppn_system.reset(genome, neat_config)


    for step_idx in range(20):
        print(f"step: {step_idx}")

        blocks = []
        blocktypes_list = ['AIR', 'CLAY', 'SLIME', 'PISTON', 'STICKY_PISTON', 'REDSTONE_BLOCK']
        for cur_y in range(cppn_system.env_size[0]):
            for cur_x in range(cppn_system.env_size[1]):
                for cur_z in range(cppn_system.env_size[2]):
                    cur_pos = Point(x=cur_x, y=4+cur_y, z=cur_z)
                    cur_type = blocktypes_list[obs[cur_y, cur_x, cur_z]]
                    cur_orientation = 'UP'
                    cur_block = Block(position=cur_pos, type=cur_type, orientation=cur_orientation)
                    blocks.append(cur_block)

        client.spawnBlocks(Blocks(blocks=blocks))
        time.sleep(5)

        obs, _, _, _ = cppn_system.step()



