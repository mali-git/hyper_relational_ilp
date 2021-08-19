"""Trainings pipeline."""
import logging
import pathlib
from typing import Any, Mapping, Optional

import torch
from class_resolver import Hint
from pykeen.losses import Loss, loss_resolver
from pykeen.trackers import CSVResultTracker, ResultTracker, WANDBResultTracker

from .data.statement_factory import get_factories
from .early_stopper import EarlyStopper
from .evaluation import evaluate
from .models import QualifierModel, model_resolver
from .training.lcwa import train_lcwa
from .training.slcwa import train_slcwa

logger = logging.getLogger(__name__)


def pipeline(
    *,
    # dataset
    inductive_setting: str,
    dataset_name: str,
    dataset_version: Optional[str] = None,
    max_num_qualifier_pairs: int,
    create_inverse_triples: bool = True,
    # training
    learning_rate: float,
    num_epochs: int = 1000,
    batch_size: int,
    label_smoothing: float,
    is_hpo: bool = False,
    num_negs_per_pos: Optional[int] = None,
    loss: Optional[Loss] = None,
    training_approach: str = "lcwa",
    save_path: Optional[pathlib.Path] = None,
    # logging
    use_wandb: bool = False,
    experiment_name: Optional[str] = None,
    verbose: bool = False,
    # evaluation
    evaluation_batch_size: int,
    early_stopping_relative_delta: float = 0.03,
    early_stopping_patience: int = 200,
    # model
    model_name: Hint[QualifierModel],
    **kwargs,
):
    """Train and evaluate a model."""
    training_fct_kwargs = dict()
    keys_to_remove = []
    # TODO: Hotfix
    for key, value in kwargs.items():
        if 'training_kwargs_' in key:
            training_fct_kwargs[key[len('training_kwargs_'):]] = value
            keys_to_remove.append(key)
    for key in keys_to_remove:
        kwargs.pop(key)

    # data
    inference_sf, test_sf, train_sf, validation_sf = get_factories(
        dataset_name=dataset_name,
        dataset_version=dataset_version,
        inductive_setting=inductive_setting,
        max_num_qualifier_pairs=max_num_qualifier_pairs,
        create_inverse_triples=create_inverse_triples,
    )

    # training loop
    if training_approach == "lcwa":
        training_fct = train_lcwa
        loss = loss or "BCEWithLogitsLoss"
    elif training_approach == "slcwa":
        training_fct = train_slcwa
        training_fct_kwargs["num_negs_per_pos"] = num_negs_per_pos
        loss = loss or "MarginRanking"
    else:
        raise ValueError(f"Training approach {training_approach} not supported.")
    loss = loss_resolver.make(query=loss)

    # model
    model = model_resolver.make(
        query=model_name,
        statement_factory=train_sf,
        pos_kwargs=kwargs,
        loss=loss,
    )
    if verbose:
        logger.info(f"Model:\n{model}")

    tracker = get_tracker(
        use_wandb=use_wandb,
        experiment_name=experiment_name,
        config=dict(
            # dataset
            inductive_setting=inductive_setting,
            dataset_name=dataset_name,
            dataset_version=dataset_version,
            max_num_qualifier_pairs=max_num_qualifier_pairs,
            create_inverse_triples=create_inverse_triples,
            # training
            learning_rate=learning_rate,
            num_epochs=num_epochs,
            batch_size=batch_size,
            label_smoothing=label_smoothing,
            num_negs_per_pos=num_negs_per_pos,
            loss=loss_resolver.normalize_inst(loss),
            training_approach=training_approach,
            # evaluation
            evaluation_batch_size=evaluation_batch_size,
            early_stopping_relative_delta=early_stopping_relative_delta,
            early_stopping_patience=early_stopping_patience,
            # model
            model_cls=model_resolver.normalize_inst(model),
            **kwargs,
        ),
    )

    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], learning_rate)

    inference_data_geometric = inference_sf.create_data_object()
    mapped_training_statements = inference_sf.mapped_statements
    restrict_entities_to = inference_sf.mapped_statements[:, :3:2].unique()

    early_stopper = EarlyStopper(
        model=model,
        mapped_training_statements=mapped_training_statements,
        relative_delta=early_stopping_relative_delta,
        evaluation_statements_factory=validation_sf,
        frequency=1,
        patience=early_stopping_patience if is_hpo else num_epochs,
        result_tracker=tracker,
        evaluation_batch_size=evaluation_batch_size,
        data_geometric=inference_data_geometric,
        metric='hits_at_10',
        restricted_evaluation=restrict_entities_to,
    )
    training_fct(
        model=model,
        batch_size=batch_size,
        num_epochs=num_epochs,
        label_smoothing=label_smoothing,
        optimizer=optimizer,
        use_tqdm=True,
        use_tqdm_batch=True,
        stopper=early_stopper,
        result_tracker=tracker,
        **training_fct_kwargs,
    )
    if is_hpo:
        # Return best value determined by early stopping
        return -early_stopper.best_metric, early_stopper.best_epoch, early_stopper.results

    metric_results = evaluate(
        model=model,
        mapped_statements=test_sf.mapped_statements,
        mapped_training_statements=mapped_training_statements,
        use_tqdm=False,
        batch_size=evaluation_batch_size,
        # Validation and tests sets are inductive
        data_geometric=inference_data_geometric,
        restrict_entities_to=restrict_entities_to,
    )

    tracker.log_metrics(
        metrics=metric_results.to_flat_dict(),
        step=num_epochs + 1,
        prefix='test',
    )

    if save_path:
        torch.save(
            dict(
                model=model.cpu(),
                factories=dict(
                    training=train_sf,
                    inference=inference_sf if id(inference_sf) != id(train_sf) else None,
                    validation=validation_sf,
                    test=test_sf,
                ),
            ),
            save_path,
        )

    return -metric_results.get_metric('hits_at_10'), None, None


def get_tracker(use_wandb: bool, experiment_name: str, config: Mapping[str, Any]) -> ResultTracker:
    if use_wandb:
        return WANDBResultTracker(
            project=PROJECT_NAME,
            experiment=experiment_name,
            config=config,
        )
    else:
        return CSVResultTracker()


PROJECT_NAME = "Improving Inductive Link Prediction Using Hyper-Relational Facts"
