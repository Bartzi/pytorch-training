import torch.nn.functional as F

from pytorch_training import Updater
from pytorch_training.reporter import get_current_reporter
from pytorch_training.updater import GradientApplier


class MNISTUpdater(Updater):

    def update_core(self):
        # get the network we want to optimize
        net = self.networks['net']

        # GradientApplier helps us save some boilerplate code
        with GradientApplier([net], self.optimizers.values()):
            # get the batch and transfer it to the training device
            batch = next(self.iterators['images'])
            batch = {k: v.to(self.device) for k, v in batch.items()}

            # perform forward pass through network
            prediction = net(batch['images'])

            # calculate loss
            loss = F.nll_loss(prediction, batch['labels'])

            # log the loss
            reporter = get_current_reporter()
            reporter.add_observation({"loss": loss}, prefix='loss')

            # perform backward pass for later weight update
            loss.backward()
