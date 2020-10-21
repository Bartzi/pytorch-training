from typing import Union, Tuple

from pytorch_training.trigger import Trigger
from pytorch_training.triggers.interval_trigger import IntervalTrigger


def get_trigger(trigger: Union[Trigger, Tuple[int, str]]) -> Trigger:
    if isinstance(trigger, Trigger):
        return trigger
    else:
        return IntervalTrigger(*trigger)

