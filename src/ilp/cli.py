# -*- coding: utf-8 -*-

"""Command-line interface with click."""

import click

from ilp.data.features import V1, V2, dataset_resolver
from ilp.hpo import hpo_pipeline
from ilp.models.blp import BLP
from ilp.models.layers import rotate
from ilp.models.qblp import QBLP
from ilp.models.stare import StarE
from ilp.pipeline import pipeline

# logging
option_wandb = click.option(
    "-wb",
    "--use-wandb",
    type=bool,
    default=False,
)

# dataset options
option_inductive_setting = click.option(
    "-is",
    "--inductive-setting",
    type=click.Choice(["semi", "full"], case_sensitive=False),
    default="full",
)

option_dataset = click.option(
    "-dn",
    "--dataset-name",
    type=click.Choice(sorted(dataset_resolver.lookup_dict.keys()), case_sensitive=False),
    default="wd50100",
)

option_dataset_version = click.option(
    "-dv",
    "--dataset-version",
    type=click.Choice([V1, V2], case_sensitive=False),
    default=V1,
)

option_create_inverse_triples = click.option(
    "-ci",
    "--create-inverse-triples",
    type=bool,
    default=True,
)

option_max_pairs = click.option(
    "-mp",
    "--max-num-qualifier-pairs",
    type=int,
    default=6,
)

# training options
option_training_approach = click.option(
    "-ta",
    "--training-approach",
    type=click.Choice(["lcwa", "slcwa"], case_sensitive=False),
    default="lcwa",
)

option_epochs = click.option(
    "-ne",
    "--num-epochs",
    type=int,
    default=1_000,
)

option_batch_size = click.option(
    "-bs",
    "--batch-size",
    type=int,
    default=128,
)

option_num_negatives = click.option(
    "-nn",
    "--num-negatives",
    type=int,
    default=10,
)

option_learning_rate = click.option(
    "-lr",
    "--learning-rate",
    type=float,
    default=0.0001,
)

option_label_smoothing = click.option(
    "-ls",
    "--label-smoothing",
    type=float,
    default=0.1,
)

# evaluation
option_eval_batch_size = click.option(
    "-eb",
    "--eval-batch-size",
    type=int,
    default=10,
)

option_embedding_dim = click.option(
    "-ed",
    "--embedding-dimension",
    type=int,
    default=256,
)

# transformer decoder (QBLP / StarE)
option_transformer_hidden = click.option(
    "-td",
    "--transformer-hidden-dimension",
    type=int,
    default=512,
)

option_transformer_heads = click.option(
    "-th",
    "--transformer-num-heads",
    type=int,
    default=2,
)

option_transformer_layers = click.option(
    "-tl",
    "--transformer-num-layers",
    type=int,
    default=2,
)


@click.group()
def main():
    """The main entry point."""


@main.group()
def run():
    """Run individual settings."""


@run.command(name="blp")
@option_inductive_setting
@option_dataset
@option_dataset_version
@option_create_inverse_triples
@option_max_pairs
@option_batch_size
@option_num_negatives
@option_epochs
@option_learning_rate
@option_wandb
@option_eval_batch_size
@option_label_smoothing
@option_training_approach
@option_embedding_dim
def run_blp(
    inductive_setting: str,
    dataset_name: str,
    dataset_version: str,
    embedding_dimension: int,
    create_inverse_triples: bool,
    max_num_qualifier_pairs: int,
    batch_size: int,
    num_negatives: int,
    num_epochs: int,
    learning_rate: float,
    use_wandb: bool,
    eval_batch_size: int,
    label_smoothing: float,
    training_approach: str,
):
    """Run BLP."""
    pipeline(
        model_name=BLP,
        dataset_name=dataset_name,
        dataset_version=dataset_version,
        max_num_qualifier_pairs=max_num_qualifier_pairs,
        create_inverse_triples=create_inverse_triples,
        inductive_setting=inductive_setting,
        learning_rate=learning_rate,
        batch_size=batch_size,
        evaluation_batch_size=eval_batch_size,
        label_smoothing=label_smoothing,
        is_hpo=False,
        use_wandb=use_wandb,
        num_epochs=num_epochs,
        verbose=False,
        num_negs_per_pos=num_negatives,
        loss=None,
        training_approach=training_approach,
        # model kwargs
        embedding_dim=embedding_dimension,
    )


@run.command(name="qblp")
@option_inductive_setting
@option_dataset
@option_dataset_version
@option_create_inverse_triples
@option_max_pairs
@option_batch_size
@option_num_negatives
@option_epochs
@option_learning_rate
@option_wandb
@option_eval_batch_size
@option_label_smoothing
@option_training_approach
@option_embedding_dim
@option_transformer_hidden
@option_transformer_heads
@option_transformer_layers
@click.option("-at", "--affine-transformation", type=bool, default=False)
def run_qblp(
    inductive_setting: str,
    dataset_name: str,
    dataset_version: str,
    embedding_dimension: int,
    transformer_hidden_dimension: int,
    create_inverse_triples: bool,
    max_num_qualifier_pairs: int,
    batch_size: int,
    num_negatives: int,
    num_epochs: int,
    learning_rate: float,
    use_wandb: bool,
    eval_batch_size: int,
    label_smoothing: float,
    transformer_num_heads: int,
    transformer_num_layers: int,
    training_approach: str,
    affine_transformation: bool,
):
    """Run QBLP."""
    pipeline(
        model_name=QBLP,
        dataset_name=dataset_name,
        dataset_version=dataset_version,
        max_num_qualifier_pairs=max_num_qualifier_pairs,
        create_inverse_triples=create_inverse_triples,
        inductive_setting=inductive_setting,
        learning_rate=learning_rate,
        batch_size=batch_size,
        evaluation_batch_size=eval_batch_size,
        label_smoothing=label_smoothing,
        is_hpo=False,
        use_wandb=use_wandb,
        num_epochs=num_epochs,
        verbose=False,
        num_negs_per_pos=num_negatives,
        loss=None,
        training_approach=training_approach,
        # model kwargs
        embedding_dim=embedding_dimension,
        dim_transformer_hidden=transformer_hidden_dimension,
        num_transformer_heads=transformer_num_heads,
        num_transformer_layers=transformer_num_layers,
        affine_transformation=affine_transformation,
    )


@run.command(name="stare")
@option_inductive_setting
@option_dataset
@option_dataset_version
@option_create_inverse_triples
@option_max_pairs
@option_batch_size
@option_epochs
@option_learning_rate
@option_wandb
@option_eval_batch_size
@option_label_smoothing
@option_training_approach
@option_embedding_dim
@option_transformer_hidden
@option_transformer_heads
@option_transformer_layers
@click.option("-ad", "--attention-dropout", type=float, default=0.1)
@click.option("-as", "--attention-slope", type=float, default=0.2)
@click.option("-ah", "--attention-num-heads", type=int, default=2)
@click.option("-gd", "--gcn-dropout", type=float, default=0.2)
@click.option("-hd", "--hidden-dropout", type=float, default=0.3)
@click.option("-td", "--transformer-dropout", type=float, default=0.1)
@click.option("-qa", "--qualifier-aggregation", type=str, default="attn")
@click.option("-qw", "--triple-qual-weight", type=float, default=0.8)
@click.option("-nl", "--num-layers", type=int, default=2)
@click.option("-ub", "--use-bias", type=bool, default=False)
def run_stare(
    inductive_setting: str,
    dataset_name: str,
    dataset_version: str,
    embedding_dimension: int,
    transformer_hidden_dimension: int,
    attention_dropout: float,
    attention_slope: float,
    gcn_dropout: float,
    hidden_dropout: float,
    label_smoothing: float,
    attention_num_heads: int,
    transformer_num_heads: int,
    transformer_num_layers: int,
    num_layers: int,
    qualifier_aggregation: str,
    triple_qual_weight: float,
    create_inverse_triples: bool,
    max_num_qualifier_pairs: int,
    batch_size: int,
    num_epochs: int,
    learning_rate: float,
    use_wandb: bool,
    eval_batch_size: int,
    transformer_dropout: float,
    training_approach: str,
    use_bias: bool,
):
    """Run StarE."""
    pipeline(
        model_name=StarE,
        dataset_name=dataset_name,
        dataset_version=dataset_version,
        max_num_qualifier_pairs=max_num_qualifier_pairs,
        create_inverse_triples=create_inverse_triples,
        inductive_setting=inductive_setting,
        learning_rate=learning_rate,
        batch_size=batch_size,
        evaluation_batch_size=eval_batch_size,
        label_smoothing=label_smoothing,
        is_hpo=False,
        use_wandb=use_wandb,
        num_epochs=num_epochs,
        verbose=False,
        num_negs_per_pos=None,
        loss=None,
        training_approach=training_approach,
        # model kwargs
        num_layers=num_layers,
        embedding_dim=embedding_dimension,
        dim_transformer_hidden=transformer_hidden_dimension,
        use_bias=use_bias,
        hid_drop=hidden_dropout,  # Used in GNN encoder
        gcn_drop=gcn_dropout,  # Used in GNN conv-layer
        transformer_drop=transformer_dropout,
        composition_function=rotate,
        qualifier_aggregation=qualifier_aggregation,
        qualifier_comp_function=rotate,
        use_attention=False,  # For attending messages
        num_attention_heads=attention_num_heads,  # For attending messages
        num_transformer_heads=transformer_num_heads,
        num_transformer_layers=transformer_num_layers,
        attention_slope=attention_slope,  # For transformer and GNN
        attention_drop=attention_dropout,  # For transformer and GNN
        triple_qual_weight=triple_qual_weight,  #
        use_learnable_x=False,
    )


@main.command()
@option_inductive_setting
@option_dataset
@option_dataset_version
@option_create_inverse_triples
@option_max_pairs
@option_create_inverse_triples
@option_training_approach
@click.option(
    "-patience",
    "--early-stopping-patience",
    type=int,
    default=200,
)
@option_epochs
@click.option(
    "-hi",
    "--num-hpo-iterations",
    type=int,
    default=100,
)
@option_wandb
@option_eval_batch_size
@click.option(
    "-m",
    "--model",
    type=click.Choice(["blp", "qblp", "stare"], case_sensitive=False),
    default="stare",
)
def tune(
    inductive_setting: str,
    dataset_name: str,
    dataset_version: str,
    max_num_qualifier_pairs: int,
    create_inverse_triples: bool,
    training_approach: str,
    num_epochs: int,
    num_hpo_iterations: int,
    early_stopping_patience: int,
    use_wandb: bool,
    eval_batch_size: int,
    model: str,
):
    """Tune hyperparameters using nevergrad."""
    hpo_pipeline(
        training_approach=training_approach,
        num_epochs=num_epochs,
        early_stopping_patience=early_stopping_patience,
        num_hpo_iterations=num_hpo_iterations,
        model_cls=model,
        # forwarded to pipeline
        dataset_name=dataset_name,
        dataset_version=dataset_version,
        inductive_setting=inductive_setting,
        max_num_qualifier_pairs=max_num_qualifier_pairs,
        create_inverse_triples=create_inverse_triples,
        evaluation_batch_size=eval_batch_size,
        use_wandb=use_wandb,
    )


if __name__ == '__main__':
    main()
