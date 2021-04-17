from evocraftsearch import OutputFitness
import torch
import torch.nn.functional as F

class ReconstructionFitness(OutputFitness):

    @staticmethod
    def default_config():
        default_config = OutputFitness.default_config()
        default_config.a_CE = 1.0
        default_config.a_IoU = 1.0
        return default_config

    def __init__(self, target, config=None, **kwargs):
        super().__init__(config={}, **kwargs)
        self.target = target


    def calc(self, observations, reduction="mean"):
        """
            Maps the observations of a system to an embedding vector
            Return a torch tensor
            # TODO: check neural3DCA loss when code release
        """
        last_potential = observations.potentials[-1]
        SX, SY, SZ, n_channels = last_potential.shape
        air_potential = last_potential[0,0,0,0]
        air_one_hot = F.one_hot(torch.tensor(0),n_channels)

        output_presence = F.relu(last_potential - air_potential).max(-1)[0] #float where presence, 0 elsewhere
        target_presence = F.relu(self.target - air_one_hot).sum(-1) #1 where presence, 0 elsewhere

        intersection = (output_presence * target_presence)
        union = (output_presence + target_presence)

        iou = (union.sum() - intersection.sum()) / (union.sum() + 1e-8)  # We smooth our devision to avoid 0/0

        ce_apply_mask = target_presence.detach().bool()
        ce = F.cross_entropy(last_potential[ce_apply_mask].view(-1,n_channels), self.target[ce_apply_mask].argmax(-1).view(-1))

        fitness = 4.0 / (self.config.a_CE * ce + self.config.a_IoU * iou)

        if reduction == "mean":
            fitness = fitness.mean()
        elif reduction == "sum":
            fitness = fitness.sum()
        else:
            raise NotImplementedError

        return fitness
