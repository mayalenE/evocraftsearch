from evocraftsearch import OutputFitness
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

        ce = F.cross_entropy(last_potential.view(-1,n_channels), self.target.argmax(-1).view(-1), reduction=reduction)
        # output_presence = last_potential.max(-1)[1] > 0
        # target_presence = self.target.argmax(-1) > 0
        # intersection = output_presence & target_presence
        # union = output_presence | target_presence

        air_potential = last_potential[0,0,0,0]
        output_presence_logits = F.relu(last_potential - air_potential).max(-1)[0] #potential value - air_value where presence, 0 elsewhere
        target_presence = (self.target.argmax(-1) > 0).float()  #1 where presence, 0 elsewhere
        # intersection = (output_presence_logits * target_presence)
        # union = (output_presence_logits + target_presence)
        # iou = (union.sum() - intersection.sum()) / (union.sum() + 1e-8)  # We smooth our devision to avoid 0/0

        iou = F.binary_cross_entropy_with_logits(output_presence_logits, target_presence)


        # ce_apply_mask = target_presence.detach().bool()
        # ce = F.cross_entropy(last_potential[ce_apply_mask].view(-1,n_channels), self.target[ce_apply_mask].argmax(-1).view(-1))

        fitness = - (self.config.a_CE * ce + self.config.a_IoU * iou)

        return fitness
