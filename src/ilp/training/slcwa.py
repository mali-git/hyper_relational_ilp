"""Stochastic local closed world assumption training loop."""
from typing import Any, Mapping, Optional

import pykeen.losses
import torch
from pykeen.stoppers import Stopper
from pykeen.trackers import ResultTracker
from pykeen.training import TrainingLoop
from pykeen.triples import CoreTriplesFactory, Instances

from ..data.statement_factory import StatementsFactory
from ..models import QualifierModel

__all__ = [
    "train_slcwa",
]

InductiveSLCWASampleType = torch.LongTensor
InductiveSLCWABatchType = torch.LongTensor


class InductiveSLCWATrainingLoop(TrainingLoop[InductiveSLCWASampleType, InductiveSLCWABatchType]):
    """Inductive variant of the SLCWA training loop."""

    def __init__(self, num_negs_per_pos: int, **kwargs):
        super().__init__(**kwargs)
        self.num_negs_per_pos = num_negs_per_pos

    @staticmethod
    def _get_batch_size(batch: InductiveSLCWABatchType) -> int:
        return batch.shape[0]

    def _create_instances(self, triples_factory: CoreTriplesFactory) -> Instances:
        assert isinstance(triples_factory, StatementsFactory)
        instances = triples_factory.create_slcwa_instances()
        self.entities = instances.entities
        return instances

    def _process_batch(self, batch: InductiveSLCWABatchType, start: int, stop: int, label_smoothing: float = 0.0, slice_size: Optional[int] = None) -> torch.FloatTensor:
        assert isinstance(self.model, QualifierModel)
        # compute positive scores
        pos_batch = batch.to(device=self.device)
        h_indices = pos_batch[:, 0]
        r_indices = pos_batch[:, 1]
        t_indices = pos_batch[:, 2]
        qualifier_indices = pos_batch[:, 3:]
        positive_scores = self.model(
            h_indices=h_indices,
            r_indices=r_indices,
            t_indices=t_indices,
            qualifier_indices=qualifier_indices,
        )

        if self.model.statement_factory.create_inverse_triples:
            # only tail corruption (since we are using inverse triples, we do not need to corrupt heads)
            negative_scores = _get_negative_scores(
                model=self.model,
                h_indices=h_indices,
                r_indices=r_indices,
                t_indices=t_indices,
                qualifier_indices=qualifier_indices,
                num_negs_per_pos=self.num_negs_per_pos,
                corruption="t",
                entities=self.entities,
            )
        else:
            # equal corruption head / tail
            negative_scores = torch.cat(
                [
                    _get_negative_scores(
                        model=self.model,
                        h_indices=h_indices,
                        r_indices=r_indices,
                        t_indices=t_indices,
                        qualifier_indices=qualifier_indices,
                        num_negs_per_pos=self.num_negs_per_pos // 2,
                        corruption=corruption,
                        entities=self.model.statement_factory.non_qualifier_only_entities,
                    )
                    for corruption in ("h", "t")
                ],
                dim=-1
            )
        return compute_slcwa_loss(model=self.model, positive=positive_scores, negative=negative_scores)

    def _slice_size_search(self, *, triples_factory: CoreTriplesFactory, training_instances: Instances, batch_size: int, sub_batch_size: int, supports_sub_batching: bool) -> int:
        raise NotImplementedError


def _get_negative_scores(
    model: QualifierModel,
    h_indices: torch.LongTensor,
    r_indices: torch.LongTensor,
    t_indices: torch.LongTensor,
    qualifier_indices: torch.LongTensor,
    num_negs_per_pos: int,
    corruption: str,
    entities: torch.LongTensor,
) -> torch.FloatTensor:
    if corruption not in {"h", "t"}:
        raise ValueError(corruption)

    # prepare kwargs
    kwargs = dict(
        h_indices=h_indices,
        r_indices=r_indices,
        t_indices=t_indices,
    )

    # remove positive indices
    key = f"{corruption}_indices"
    pos_indices = kwargs.pop(key)

    # generate negative indices
    neg_indices = entities[torch.randint(
        entities.shape[0],
        size=(pos_indices.shape[0], num_negs_per_pos),
        device=entities.device,
    )].to(pos_indices.device)

    # add negative indices
    kwargs[key] = neg_indices

    # compute scores
    return model(
        **kwargs,
        qualifier_indices=qualifier_indices,
    )


def train_slcwa(
    model: QualifierModel,
    num_epochs: int,
    batch_size: int,
    num_negs_per_pos: int,
    optimizer,
    label_smoothing=None,
    use_tqdm: bool = True,
    use_tqdm_batch: bool = True,
    tqdm_kwargs: Optional[Mapping[str, Any]] = None,
    stopper: Optional[Stopper] = None,
    result_tracker: Optional[ResultTracker] = None,
):
    loop = InductiveSLCWATrainingLoop(
        model=model,
        triples_factory=model.statement_factory,
        optimizer=optimizer,
        num_negs_per_pos=num_negs_per_pos,
    )
    return loop.train(
        triples_factory=model.statement_factory,
        num_epochs=num_epochs,
        batch_size=batch_size,
        label_smoothing=label_smoothing,
        stopper=stopper,
        result_tracker=result_tracker,
    )


def compute_slcwa_loss(
    model: torch.nn.Module,
    positive: torch.FloatTensor,
    negative: torch.FloatTensor,
) -> torch.FloatTensor:
    if isinstance(model.loss, pykeen.losses.BCEWithLogitsLoss):
        # pointwise
        return model.loss.process_slcwa_scores(positive, negative)
    elif isinstance(model.loss, pykeen.losses.MarginRankingLoss):
        # MarginRankingLoss supports broadcasting
        positive = positive.view(*positive.shape, 1)
        negative = negative.view(*positive.shape, -1)
        return model.loss.process_slcwa_scores(positive, negative)
    else:
        raise NotImplementedError(model.loss)
