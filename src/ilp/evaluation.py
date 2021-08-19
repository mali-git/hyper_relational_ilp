"""Evaluation methods based upon pykeen."""
import logging
import pathlib
from typing import Collection, Mapping, Optional, Union

import pandas
import torch
from pykeen.evaluation import RankBasedMetricResults
from pykeen.evaluation.evaluator import create_sparse_positive_filter_, filter_scores_, optional_context_manager
from pykeen.evaluation.rank_based_evaluator import RankBasedEvaluator, SIDES
from pykeen.utils import split_list_in_batches_iter
from torch_geometric.data import Data
from tqdm.auto import tqdm

# backward compatability
try:
    from pykeen.evaluation.rank_based_evaluator import RANK_AVERAGE
except ImportError:
    from pykeen.evaluation.rank_based_evaluator import RANK_REALISTIC as RANK_AVERAGE

from .models.base import QualifierModel

logger = logging.getLogger(__name__)


class ExtendedRankBasedEvaluator(RankBasedEvaluator):
    """Extension of rank-based evaluator which adds an export to pandas."""

    def get_df(self, **kwargs) -> pandas.DataFrame:
        data = {
            f"{side}_rank": self._get_ranks(side=side, rank_type=RANK_AVERAGE)
            for side in SIDES
            if side != "both"
        }
        for k, v in kwargs.items():
            data[k] = v
        return pandas.DataFrame(data=data)


@torch.no_grad()
def evaluate(
    model: QualifierModel,
    mapped_statements: torch.LongTensor,  # statements
    mapped_training_statements: Optional[torch.LongTensor] = None,
    data_geometric: Optional[Data] = None,
    ks: Optional[Collection[int]] = None,
    filtered: bool = True,
    only_size_probing: bool = False,
    batch_size: Optional[int] = None,
    device: Optional[torch.device] = None,
    use_tqdm: bool = True,
    tqdm_kwargs: Optional[Mapping[str, str]] = None,
    restrict_entities_to: Optional[torch.LongTensor] = None,
    store_ranks: Union[None, str, pathlib.Path] = None,
) -> RankBasedMetricResults:
    """Adapter to allow storing ranks into file before finalizing."""
    evaluator = _evaluate(
        model=model,
        mapped_statements=mapped_statements,
        mapped_training_statements=mapped_training_statements,
        data_geometric=data_geometric,
        ks=ks,
        filtered=filtered,
        only_size_probing=only_size_probing,
        batch_size=batch_size,
        device=device,
        use_tqdm=use_tqdm,
        tqdm_kwargs=tqdm_kwargs,
        restrict_entities_to=restrict_entities_to,
    )

    if store_ranks:
        store_ranks = pathlib.Path(store_ranks).expanduser().resolve()
        logger.info(f"Storing ranks to {store_ranks.as_uri()}")
        num_qualifiers = (mapped_statements[:, 3::2] != model.statement_factory.padding_idx).sum(dim=-1).cpu().numpy()
        evaluator.get_df(num_qualifiers=num_qualifiers).to_csv(store_ranks, sep="\t", index_label="statement_id")

    # Finalize
    return evaluator.finalize()


def _evaluate(
    model: QualifierModel,
    mapped_statements: torch.LongTensor,  # statements
    mapped_training_statements: Optional[torch.LongTensor] = None,
    data_geometric: Optional[Data] = None,
    ks: Optional[Collection[int]] = None,
    filtered: bool = True,
    only_size_probing: bool = False,
    batch_size: Optional[int] = None,
    device: Optional[torch.device] = None,
    use_tqdm: bool = True,
    tqdm_kwargs: Optional[Mapping[str, str]] = None,
    restrict_entities_to: Optional[torch.LongTensor] = None,
) -> ExtendedRankBasedEvaluator:
    """The actual evaluation."""
    evaluator = ExtendedRankBasedEvaluator(ks=ks, filtered=filtered)
    # Send to device
    if device is not None:
        model = model.to(device)
    device = model.device
    # Ensure evaluation mode
    model.eval()
    # Prepare for result filtering
    if filtered:
        if mapped_training_statements is None:
            mapped_training_statements = model.statement_factory.mapped_statements
        all_pos_triples = torch.cat([mapped_training_statements[:, :3], mapped_statements[:, :3]], dim=0)
        all_pos_triples = all_pos_triples.to(device=device)
    else:
        all_pos_triples = None
    # Send tensors to device
    mapped_statements = mapped_statements.to(device=device)
    if restrict_entities_to is not None:
        restrict_entities_to = restrict_entities_to.to(device=device)
        if data_geometric is None:
            ignore_entity_mask = torch.ones(model.statement_factory.num_entities, dtype=torch.bool)
        else:
            ignore_entity_mask = torch.ones(data_geometric.entities.shape[0], dtype=torch.bool)
        ignore_entity_mask[restrict_entities_to] = False
        ignore_entity_mask = ignore_entity_mask.to(device=device)
    else:
        ignore_entity_mask = None
    # Prepare batches
    if batch_size is None:
        batch_size = 1
    batches = split_list_in_batches_iter(input_list=mapped_statements, batch_size=batch_size)
    # Show progressbar
    num_statements = mapped_statements.shape[0]
    # Flag to check when to quit the size probing
    evaluated_once = False
    # Disable gradient tracking
    _tqdm_kwargs = dict(
        desc=f'Evaluating on {model.device}',
        total=num_statements,
        unit='statement',
        unit_scale=True,
        # Choosing no progress bar (use_tqdm=False) would still show the initial progress bar without disable=True
        disable=not use_tqdm,
    )
    if tqdm_kwargs:
        _tqdm_kwargs.update(tqdm_kwargs)
    with optional_context_manager(use_tqdm, tqdm(**_tqdm_kwargs)) as progress_bar:
        # batch-wise processing
        for batch in batches:
            batch_size = batch.shape[0]
            relation_filter = None
            for column in (0, 2):
                relation_filter = _evaluate_batch(
                    batch=batch,
                    model=model,
                    data_geometric=data_geometric,
                    column=column,
                    evaluator=evaluator,
                    all_pos_triples=all_pos_triples,
                    relation_filter=relation_filter,
                    filtering_necessary=filtered,
                    padding_idx=model.statement_factory.padding_idx,
                    ignore_entity_mask=ignore_entity_mask,
                )

            # If we only probe sizes we do not need more than one batch
            if only_size_probing and evaluated_once:
                break

            evaluated_once = True

            if use_tqdm:
                progress_bar.update(batch_size)
    # Empty buffers
    # Fixme: change name?
    model.post_parameter_update()
    return evaluator


def _evaluate_batch(
    batch: torch.LongTensor,  # statements
    model: QualifierModel,
    column: int,
    evaluator: RankBasedEvaluator,
    all_pos_triples: Optional[torch.LongTensor],  # statements?
    relation_filter: Optional[torch.BoolTensor],
    filtering_necessary: bool,
    data_geometric: Optional[Data] = None,
    padding_idx: int = 0,
    ignore_entity_mask: Optional[torch.BoolTensor] = None,
) -> torch.BoolTensor:
    """Evaluate on a single batch."""
    if column not in {0, 2}:
        raise ValueError(f'column must be either 0 or 2, but is column={column}')

    # Predict scores once
    if column == 2:  # tail scores
        batch_scores_of_corrupted = model.score_t(batch[:, 0:2], qualifiers=batch[:, 3:], data_geometric=data_geometric)
    else:
        batch_scores_of_corrupted = model.score_h(batch[:, 1:3], qualifiers=batch[:, 3:], data_geometric=data_geometric)
    assert (batch[:, column] < batch_scores_of_corrupted.shape[1]).all()

    # Select scores of true
    batch_scores_of_true = batch_scores_of_corrupted[
        torch.arange(0, batch.shape[0]),
        batch[:, column],
    ]

    # Create positive filter for all corrupted
    if filtering_necessary:
        # Needs all positive triples
        if all_pos_triples is None:
            raise ValueError('If filtering_necessary of positive_masks_required is True, all_pos_triples has to be '
                             'provided, but is None.')

        # Create filter
        positive_filter, relation_filter = create_sparse_positive_filter_(
            hrt_batch=batch,
            all_pos_triples=all_pos_triples,
            relation_filter=relation_filter,
            filter_col=column,
        )

        batch_scores_of_corrupted = filter_scores_(
            scores=batch_scores_of_corrupted,
            filter_batch=positive_filter,
        )

        # The scores for the true triples have to be rewritten to the scores tensor
        batch_scores_of_corrupted[
            torch.arange(0, batch.shape[0]),
            batch[:, column],
        ] = batch_scores_of_true

    # mask padding idx
    batch_scores_of_corrupted[:, padding_idx] = float("nan")

    # Restrict evaluation to certain entities, e.g. non-qualifier entities
    if ignore_entity_mask is not None:
        batch_scores_of_corrupted = torch.masked_fill(batch_scores_of_corrupted, ignore_entity_mask.unsqueeze(dim=0), value=float("nan"))

    evaluator._update_ranks_(
        true_scores=batch_scores_of_true[:, None],
        all_scores=batch_scores_of_corrupted,
        side="head" if column == 0 else "tail",
    )

    return relation_filter
