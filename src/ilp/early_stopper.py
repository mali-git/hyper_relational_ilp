# -*- coding: utf-8 -*-

"""
Implementation of early stopping based upon pykeen.

Replaces triples factories by statement factories, and uses a different evaluation method.
"""

import dataclasses
import logging
from dataclasses import dataclass
from typing import Any, List, Mapping, Optional

import torch
import torch.nn as nn
from pykeen.stoppers import Stopper
from pykeen.stoppers.early_stopping import is_improvement
from pykeen.trackers import ResultTracker
from torch_geometric.data import Data

from .evaluation import evaluate
from .data.statement_factory import StatementsFactory

logger = logging.getLogger(__name__)


@dataclass
class EarlyStopper(Stopper):
    """A harness for early stopping."""

    #: The model
    model: nn.Module = dataclasses.field(repr=False)
    #: The statements to use for evaluation; this is different to pykeen.
    evaluation_statements_factory: StatementsFactory = dataclasses.field(init=True)
    #: Size of the evaluation batches
    evaluation_batch_size: Optional[int] = None
    #: Slice size of the evaluation batches
    evaluation_slice_size: Optional[int] = None
    #: The number of epochs after which the model is evaluated on validation set
    frequency: int = 10
    #: The number of iterations (one iteration can correspond to various epochs)
    #: with no improvement after which training will be stopped.
    patience: int = 2
    #: The name of the metric to use
    metric: str = 'hits_at_k'
    #: The minimum relative improvement necessary to consider it an improved result
    relative_delta: float = 0.01
    #: The best result so far
    best_metric: Optional[float] = None
    #: The epoch at which the best result occurred
    best_epoch: Optional[int] = None
    #: The remaining patience
    remaining_patience: int = dataclasses.field(init=False)
    #: The metric results from all evaluations
    results: List[float] = dataclasses.field(default_factory=list, repr=False)
    #: Whether a larger value is better, or a smaller
    larger_is_better: bool = True
    #: The result tracker
    result_tracker: Optional[ResultTracker] = None
    #: Did the stopper ever decide to stop?
    stopped: bool = False
    #: The validation graph which is required in the inductive setting
    data_geometric: Optional[Data] = None
    #: In the inductive setting, the statements to filter out are not part of the training graph
    mapped_training_statements: Optional[torch.LongTensor] = None
    #: Indicates, whether to restrict evaluation to non-qualifier-only entities
    restricted_evaluation: bool = False

    def __post_init__(self):
        """Run after initialization and check the metric is valid."""
        if self.evaluation_statements_factory is None:
            raise ValueError('Must specify a validation_triples_factory or a dataset for using early stopping.')

        self.remaining_patience = self.patience

        # Dummy result tracker
        if self.result_tracker is None:
            self.result_tracker = ResultTracker()

    def should_evaluate(self, epoch: int) -> bool:
        """Decide if evaluation should be done based on the current epoch and the internal frequency."""
        return epoch > 0 and epoch % self.frequency == 0

    @property
    def number_results(self) -> int:
        """Count the number of results stored in the early stopper."""
        return len(self.results)

    def should_stop(self, epoch: int) -> bool:
        """Evaluate on a metric and compare to past evaluations to decide if training should stop."""
        # Evaluate
        if torch.is_tensor(self.restricted_evaluation):
            restrict_entities_to = self.restricted_evaluation
        elif self.restricted_evaluation:
            restrict_entities_to = self.evaluation_statements_factory.non_qualifier_only_entities
        else:
            restrict_entities_to = None

        # this evaluation method is different to pykeen
        metric_results = evaluate(
            model=self.model,
            mapped_statements=self.evaluation_statements_factory.mapped_statements,
            mapped_training_statements=self.mapped_training_statements,
            use_tqdm=False,
            batch_size=self.evaluation_batch_size,
            data_geometric=self.data_geometric,
            restrict_entities_to=restrict_entities_to,
        )

        self.result_tracker.log_metrics(
            metrics=metric_results.to_flat_dict(),
            step=epoch,
            prefix='validation',
        )
        result = metric_results.get_metric(self.metric)

        # Append to history
        self.results.append(result)

        # check for improvement
        if self.best_metric is None or is_improvement(
            best_value=self.best_metric,
            current_value=result,
            larger_is_better=self.larger_is_better,
            relative_delta=self.relative_delta,
        ):
            self.best_epoch = epoch
            self.best_metric = result
            self.remaining_patience = self.patience
        else:
            self.remaining_patience -= 1

        # Stop if the result did not improve more than delta for patience evaluations
        if self.remaining_patience <= 0:
            logger.info(
                f'Stopping early after {self.number_results} evaluations at epoch {epoch}. The best result '
                f'{self.metric}={self.best_metric} occurred at epoch {self.best_epoch}.',
            )
            self.stopped = True
            return True

        return False

    def get_summary_dict(self) -> Mapping[str, Any]:
        """Get a summary dict."""
        return dict(
            frequency=self.frequency,
            patience=self.patience,
            relative_delta=self.relative_delta,
            metric=self.metric,
            larger_is_better=self.larger_is_better,
            results=self.results,
            stopped=self.stopped,
            best_epoch=self.best_epoch,
            best_metric=self.best_metric,
        )
