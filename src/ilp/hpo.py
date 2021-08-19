# -*- coding: utf-8 -*-

"""Implementation of the hyper-parameter optimization pipeline."""
import logging
from typing import Sequence, Tuple

import nevergrad as ng
from class_resolver import Hint
from pykeen.losses import BCEWithLogitsLoss, MarginRankingLoss

from .models import QualifierModel, StarE, model_resolver
from .models.layers import rotate
from .pipeline import pipeline

logger = logging.getLogger(__name__)

__all__ = [
    "objective",
    "hpo_pipeline",
]

# search spaces
hpo_ranges = dict(
    embedding_dim=ng.p.TransitionChoice(list(range(128, 256 + 1, 32))),
    batch_size=ng.p.TransitionChoice(list(range(128, 1025, 64))),
    learning_rate=ng.p.Log(lower=0.0001, upper=1.0),
    label_smoothing=ng.p.TransitionChoice([0.1, 0.15]),
    early_stopping_relative_delta=0.003,
)
transformer_hpo_ranges = dict(
    dim_transformer_hidden=ng.p.TransitionChoice([512, 1024]),
    num_transformer_heads=ng.p.TransitionChoice([2, 4]),
    num_transformer_layers=ng.p.TransitionChoice([2, 4]),
    affine_transformation=ng.p.Choice([True, False]),
)
dropout = [0.1, 0.2, 0.3, 0.4, 0.5]
gnn_hpo_ranges = dict(
    use_learnable_x=False,
    num_layers=ng.p.TransitionChoice([2, 3]),
    hid_drop=ng.p.Choice(dropout),
    gcn_drop=ng.p.Choice(dropout),
    attention_slope=ng.p.TransitionChoice([0.1, 0.2, 0.3, 0.4]),
    triple_qual_weight=0.8,
    attention_drop=ng.p.Choice(dropout),
    num_attention_heads=ng.p.TransitionChoice([2, 4]),
    composition_function=rotate,
    qualifier_aggregation=ng.p.TransitionChoice(["sum", "attn"]),
    qualifier_comp_function=rotate,
    use_bias=False,
    use_attention=False,
)


def objective(
    kwargs,
    raise_on_error: bool = False,
    verbose: bool = False,
) -> Tuple[float, int, Sequence[float]]:
    """Optimization objective to minimize."""
    try:
        return pipeline(
            is_hpo=True,
            verbose=verbose,
            **kwargs,
        )
    except Exception as e:
        logger.error(f"ERROR: {e}")
        if 'CUDA out of memory.' in e.args[0]:
            return 1000.0, None, None
        if raise_on_error:
            raise e
        return 1000.0, None, None


def hpo_pipeline(
    *,
    training_approach: str,
    num_epochs: int,
    early_stopping_patience: int,
    num_hpo_iterations: int,
    model_cls: Hint[QualifierModel],
    **kwargs,
):
    """Optimize hyperparameters using nevergrad."""
    # normalize model class
    model_cls = model_resolver.lookup(query=model_cls)

    # model specific search spaces
    model_kwargs = dict()
    if issubclass(model_cls, QualifierModel):
        model_kwargs.update(transformer_hpo_ranges)
    if issubclass(model_cls, StarE):
        model_kwargs.update(gnn_hpo_ranges)

    # training approach specific parameters
    training_approach_hpo_ranges = dict(
        training_approach=training_approach,
    )
    if training_approach == "slcwa":
        training_approach_hpo_ranges.update(
            loss=MarginRankingLoss(margin=1),
            num_negs_per_pos=ng.p.TransitionChoice([k for k in range(1, 65, 1)]),
        )
    else:
        training_approach_hpo_ranges.update(
            loss=BCEWithLogitsLoss(),
        )

    # construct search space
    kwargs = ng.p.Dict(
        num_epochs=num_epochs,
        model_name=model_cls,
        early_stopping_patience=early_stopping_patience,
        # static kwargs
        **kwargs,
        # hpo ranges
        **training_approach_hpo_ranges,
        **hpo_ranges,
        **model_kwargs,
    )

    track_epochs = []
    track_performance = []
    track_params = []
    optimizer = ng.optimizers.RandomSearch(parametrization=kwargs, budget=num_hpo_iterations)
    for _ in range(optimizer.budget):
        x = optimizer.ask()
        loss, current_num_epochs, _ = objective(*x.args, **x.kwargs)
        track_epochs.append(current_num_epochs)
        track_performance.append(loss)
        track_params.append(x.value)
        optimizer.tell(x, loss)
    # recommendation = optimizer.minimize(objective, verbosity=2)
    recommendation = optimizer.provide_recommendation()
    config = dict(recommendation.value)
    config.pop("num_epochs")
    id_best = track_performance.index(min(track_performance))
    pipeline(
        **config,
        is_hpo=False,
        num_epochs=track_epochs[id_best],
    )
