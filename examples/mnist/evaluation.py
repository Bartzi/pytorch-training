import torch
import torch.nn.functional as F
from torch import nn

from pytorch_training.reporter import get_current_reporter


class MNISTEvaluator:

    def __init__(self, network: nn.Module):
        self.network = network

    def __call__(self, batch):
        reporter = get_current_reporter()

        # since we only evaluate, we do not need to save the computational graph
        with torch.no_grad():
            output = self.network(batch['images'])

            loss = F.nll_loss(output, batch['labels'])
            # calculate accuracy by taking most probable predictions
            predictions = output.argmax(dim=1, keepdim=True)
            accuracy = predictions.eq(batch['labels'].view_as(predictions)).sum().item() / len(batch['images'])

            reporter.add_observation({"test_loss": loss}, prefix='loss')
            reporter.add_observation({"accuracy": accuracy}, prefix='accuracy')
