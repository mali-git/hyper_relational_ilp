"""Create command line scripts from configuration JSONs."""
import json
import pathlib
from textwrap import dedent
from typing import Any, Iterable, Mapping, Optional, Sequence, Tuple

import click

from ilp.models import model_resolver

_convert_to_int = {
    "batch-size",
    "eval-batch-size",
    "embedding-dimension",
    "transformer-hidden-dimension",
    "transformer-num-heads",
    "transformer-num-layers",
    "num-layers",
    "attention-num-heads",
}


def _iter_config(
    config: Mapping[str, Any],
    key_pairs: Iterable[Tuple[str, str]],
    ignore: Iterable[str] = tuple(),
) -> Iterable:
    all_keys = set(config.keys())
    for old_key, new_key in key_pairs:
        if old_key in config:
            all_keys.remove(old_key)
            yield f"--{new_key}"
            value = config[old_key]
            if new_key in _convert_to_int:
                value = int(value)
            yield value
    all_keys = all_keys.difference([
        'GPU Count',  # does not matter
        'GPU Type',  # does not matter
        'Name',  # wandb
        'Notes',  # wandb
        'State',  # wandb
        'loss',  # chosen based on training assumption
        'loss_name',  # chosen based on training assumption
        'max_num_qualifier_pairs',  # already inferred
        'optimizer',  # always adam
        'test_path',  # automatically chosen
        'validation_path',  # automatically chosen
        'device',  # does not matter
        'model_name',  # already inferred
        'restricted_evaluation',  # legacy
    ])
    all_keys = all_keys.difference(ignore)
    all_keys = [
        k
        for k in all_keys
        if not (
            k.startswith("test") or
            k.startswith("validation") or
            k.startswith("train") or
            k.startswith("inference")
        )
    ]
    for k in all_keys:
        print(f"WARNING: Ignoring: {k}={config[k]}")
    if all_keys:
        raise ValueError(all_keys)


_common_key_pairs = [
    # batch_size
    ("batch_size", "batch-size"),
    # create_inverse_triples
    ("create_inverse_triples", "create-inverse-triples"),
    # eval_batch_size
    ("evaluation_batch_size", "eval-batch-size"),
    # label_smoothing
    ("label_smoothing", "label-smoothing",),
    # learning_rate
    ("learning_rate", "learning-rate"),
    # num_epochs
    ("num_epochs", "num-epochs"),
    # embedding_dim
    ("embedding_dim", "embedding-dimension"),
]
_transformer_key_pairs = [
    # dim_transformer_hidden
    ("dim_transformer_hidden", "transformer-hidden-dimension"),
    # num_transformer_heads
    ("num_transformer_heads", "transformer-num-heads"),
    # num_transformer_layers
    ("num_transformer_layers", "transformer-num-layers"),
]


def _iter_slcwa_from_config(config: Mapping[str, Any]) -> Iterable:
    use_slcwa = config.get("use_slcwa")
    if use_slcwa:
        yield "--training-approach"
        yield "slcwa"
        yield "--num-negatives"
        yield config["num_negs_per_pos"]
    else:
        yield "--training-approach"
        yield "lcwa"


def _iter_blp_config(config: Mapping[str, Any]) -> Iterable:
    yield from _iter_config(config=config, key_pairs=_common_key_pairs, ignore=[
        # handled later
        "num_negs_per_pos",
        "use_slcwa",
        # always transe
        "scoring_fct_name",
    ])
    yield from _iter_slcwa_from_config(config)


def _iter_qblp_config(config: Mapping[str, Any]) -> Iterable:
    yield from _iter_config(config=config, key_pairs=_common_key_pairs + _transformer_key_pairs + [
        # affine_transformation
        ("affine_transformation", "affine-transformation"),
    ], ignore=[
        # handled later
        "num_negs_per_pos",
        "use_slcwa",
    ])
    yield from _iter_slcwa_from_config(config)


def _iter_stare_config(config: Mapping[str, Any]) -> Iterable:
    yield from _iter_config(
        config=config,
        key_pairs=_common_key_pairs + _transformer_key_pairs + [
            # attention_drop
            ("attention_drop", "attention-dropout"),
            # attention_slope
            ("attention_slope", "attention-slope"),
            # feature_dim
            # TODO: not configurable
            # gcn_drop
            ("gcn_drop", "gcn-dropout"),
            # hid_drop
            ("hid_drop", "hidden-dropout"),
            # hidden_dimension -> same as transformer-hidden-dimension
            ("hidden_dimension", "transformer-hidden-dimension"),
            # loss
            # TODO: not configurable
            # num_attention_heads
            ("num_attention_heads", "attention-num-heads"),
            # num_layers
            ("num_layers", "num-layers"),
            # qualifier_aggregation
            ("qualifier_aggregation", "qualifier-aggregation"),
            # qualifier_comp_function
            # TODO: not configurable
            # ("qualifier_comp_function", ""),
            # triple_qual_weight
            ("triple_qual_weight", "triple-qual-weight"),
            # use_bias
            ("use_bias", "use-bias"),
        ],
        ignore=[
            'composition_function',  # always rotate
            'qualifier_comp_function',  # always rotate
            'feature_dim',  # inferred from data
            'use_attention',  # always False
            'use_learnable_x',  # always False
        ]
    )


def _command_from_config(
    config_path: pathlib.Path,
) -> str:
    parts = config_path.name.split(sep="_")

    # model = config["model_name"]  # not available for all configurations :-/
    # normalize name
    model = model_resolver.normalize_cls(model_resolver.lookup(query=parts[0]))

    dataset = parts[1:-2]
    if dataset[0] == "semi":
        inductive_setting = "semi"
        dataset = "".join(dataset[1:])
        remap = dict(
            wd2033="wd50100",
            wd2025="wd5066",
        )
        dataset = remap[dataset]
        version = None
    else:
        inductive_setting = "full"
        *dataset, version = dataset
        dataset = "".join(dataset).replace("wd20", "wd50")
    max_num_qualifier = int(parts[-1].replace(".json", ""))

    command = ["ilp", "run", model]
    command.extend((
        "--inductive-setting", inductive_setting,
        "--dataset-name", dataset,
        "--max-num-qualifier-pairs", max_num_qualifier,
        "--use-wandb", True,
    ))
    if version is not None:
        command.extend(("--dataset-version", version))

    with config_path.open("r") as json_file:
        config = json.load(json_file)
    if model == "stare":
        command.extend(_iter_stare_config(config))
    elif model == "blp":
        command.extend(_iter_blp_config(config))
    elif model == "qblp":
        command.extend(_iter_qblp_config(config))
    else:
        raise ValueError

    return " ".join(map(str, command))


@click.command()
@click.option("-i", "--input-root", type=pathlib.Path, default=pathlib.Path(__file__).resolve().parents[1].joinpath("best_configs"))
@click.option("-o", "--output-path", type=pathlib.Path, default=pathlib.Path(__file__).parent.joinpath("experiments.sh"))
@click.option("-g", "--gpu", type=int, multiple=True, default=[])
@click.option("-v", "--venv", type=pathlib.Path, default=None)
@click.option("-w", "--working-directory", type=pathlib.Path, default=None)
def main(
    input_root: pathlib.Path,
    output_path: pathlib.Path,
    venv: Optional[pathlib.Path],
    working_directory: Optional[pathlib.Path],
    gpu: Sequence[int],
):
    """Create run commands from config."""
    input_root = input_root.expanduser().resolve()
    print(f"Reading from {input_root.as_uri()}")
    output_path = output_path.expanduser().resolve()

    if not gpu:
        assert venv
        for i, path in enumerate(input_root.rglob("*.json")):
            try:
                output_path.parent.joinpath(f"experiment_{i}.sh").write_text(dedent(f"""\
                                #!/bin/bash
                                #SBATCH --job-name=ilp-{i}
                                #SBATCH --ntasks=1
                                #SBATCH --cpus-per-task=4
                                #SBATCH --ntasks-per-node=1
                                #SBATCH --time=12:00:00
                                #SBATCH --gres=gpu:1

                                # activate venv
                                source {venv.as_posix()}
                                
                                # generated from {path.relative_to(input_root).as_posix()}
                                
                                {_command_from_config(config_path=path)}
                            """))
            except ValueError:
                print(f"Skipping {path.relative_to(input_root)}")
    else:
        lines = []
        for path in input_root.rglob("*.json"):
            try:
                lines.append(f"# {path.relative_to(input_root).as_posix()}\n{_command_from_config(config_path=path)}")
            except ValueError:
                print(f"Skipping {path.relative_to(input_root)}")
        n_gpu = len(gpu)
        for i, gpu_id in enumerate(gpu):
            this_output_path = output_path.with_name(name=output_path.with_suffix("").name + f"_{i}").with_suffix(output_path.suffix)
            tasks = lines[i::n_gpu]
            preamble = [f"export CUDA_VISIBLE_DEVICES={i}"]
            if working_directory:
                preamble.append(f"cd {working_directory.as_posix()}")
            if venv:
                preamble.append(f"source {venv.as_posix()}")
            this_output_path.write_text("\n\n".join(preamble + tasks))
            print(f"Written {len(tasks)} tasks to {this_output_path.as_uri()}")


if __name__ == '__main__':
    main()
