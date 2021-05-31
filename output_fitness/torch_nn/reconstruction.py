from evocraftsearch import OutputFitness
import torch.nn.functional as F

class ReconstructionFitness(OutputFitness):

    @staticmethod
    def default_config():
        default_config = OutputFitness.default_config()
        default_config.a_CE = 10.0
        default_config.a_IoU = 10.0
        return default_config

    def __init__(self, target, config={}, **kwargs):
        super().__init__(config=config, **kwargs)
        self.target_presence = (self.target.argmax(0) > 0).float()  # 1 where presence, 0 elsewhere
        self.target = target.argmax(0)


    def calc(self, observations, reduction="mean"):
        """
            Maps the observations of a system to an embedding vector
            Return a torch tensor
            # TODO: check neural3DCA loss when code release
        """
        last_potential = observations.potentials[-1]
        n_channels, SZ, SY, SX = last_potential.shape
        air_potential = last_potential[0, 0, 0, 0]



        fitness = 0

        for potential in observations.potentials:

            output_presence = potential.detach().argmax(0) > 0

            output_presence_logits = F.relu(potential - air_potential).max(0)[0]  # potential value - air_value where presence, 0 elsewhere

            ce = F.cross_entropy(potential[:, output_presence].view(n_channels, -1).transpose(0,1), self.target[output_presence].view(-1), reduction=reduction)

            iou = F.binary_cross_entropy_with_logits(output_presence_logits, self.target_presence)

            fitness -= (self.config.a_CE * ce + self.config.a_IoU * iou)

        return fitness
