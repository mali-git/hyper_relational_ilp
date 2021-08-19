"""Local closed world assumption training loop."""
from typing import Any, Mapping, Optional, Tuple

import torch
import torch.nn as nn
from pykeen.stoppers import Stopper
from pykeen.trackers import ResultTracker
from pykeen.training import TrainingLoop
from pykeen.triples import CoreTriplesFactory, Instances

# backward compatability
try:
    from pykeen.training.utils import apply_label_smoothing
except ImportError:
    from pykeen.losses import apply_label_smoothing

from ilp.data.statement_factory import StatementsFactory
from ilp.data.statement_factory import LCWAInstances as InductiveLCWAInstances

__all__ = [
    "train_lcwa",
]

InductiveLCWASampleType = Tuple[torch.LongTensor, torch.FloatTensor]
InductiveLCWABatchType = InductiveLCWASampleType


class InductiveLCWATrainingLoop(TrainingLoop[InductiveLCWASampleType, InductiveLCWABatchType]):
    """Inductive variant of the LCWATrainingLoop."""

    @staticmethod
    def _get_batch_size(batch: InductiveLCWABatchType) -> int:  # noqa: D102
        return batch[0].shape[0]

    def _create_instances(self, triples_factory: CoreTriplesFactory) -> InductiveLCWAInstances:  # noqa: D102
        assert isinstance(triples_factory, StatementsFactory)
        # TODO: Fix typing by subclassing from CoreTriplesFactory / Instances?
        return triples_factory.create_lcwa_instances()

    def _process_batch(
        self,
        batch: InductiveLCWABatchType,
        start: int,
        stop: int,
        label_smoothing: float = 0.0,
        slice_size: Optional[int] = None,
    ) -> torch.FloatTensor:  # noqa: D102
        batch_pairs, batch_labels_full = batch

        # Send batch to device
        batch_pairs = batch_pairs.to(device=self.device)
        batch_labels_full = batch_labels_full.to(device=self.device)

        scores = self.model(
            h_indices=batch_pairs[:, 0],
            r_indices=batch_pairs[:, 1],
            t_indices=None,
            qualifier_indices=batch_pairs[:, 2:],
        )

        # apply label smoothing if necessary (TODO: do this in data loader?)
        if label_smoothing > 0.:
            batch_labels_full = apply_label_smoothing(
                labels=batch_labels_full,
                epsilon=label_smoothing,
                num_classes=self.model.num_entities,
            )

        # Exclude scores obtained by scoring against qualifier-only entities
        non_qualifier_only_entities = self.model.statement_factory.non_qualifier_only_entities
        entity_mask = torch.zeros(self.model.num_entities, dtype=torch.bool)
        entity_mask[non_qualifier_only_entities] = True

        return self.model.loss(
            scores[:, entity_mask],
            batch_labels_full[:, entity_mask],
        )

    def _slice_size_search(self, *, triples_factory: CoreTriplesFactory, training_instances: Instances, batch_size: int, sub_batch_size: int, supports_sub_batching: bool) -> int:
        raise NotImplementedError


def train_lcwa(
    model: nn.Module,
    num_epochs: int,
    batch_size: int,
    optimizer,
    label_smoothing: float,
    use_tqdm: bool = True,
    use_tqdm_batch: bool = True,
    tqdm_kwargs: Optional[Mapping[str, Any]] = None,
    stopper: Optional[Stopper] = None,
    result_tracker: Optional[ResultTracker] = None,
):
    """LCWA training loop."""
    loop = InductiveLCWATrainingLoop(
        model=model,
        triples_factory=model.statement_factory,
        optimizer=optimizer,
    )
    return loop.train(
        triples_factory=model.statement_factory,
        num_epochs=num_epochs,
        batch_size=batch_size,
        label_smoothing=label_smoothing,
        stopper=stopper,
        result_tracker=result_tracker,
    )
