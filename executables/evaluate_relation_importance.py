import logging
import pathlib
import re
from typing import Mapping

import click
import numpy
import pandas
import requests
import torch
from matplotlib import cm, pyplot as plt
from torch_geometric.data import Data as DataGeometric
from tqdm.auto import tqdm

from ilp.data.statement_factory import PADDING, StatementsFactory
from ilp.evaluation import _evaluate
from ilp.models.base import QualifierModel
from ilp.utils import get_device


def _get_labels(
    relation_df: pandas.DataFrame,
    buffer_path: pathlib.Path = None,
) -> pandas.DataFrame:
    if buffer_path is None:
        buffer_path = pathlib.Path("relation_names.tsv")
    buffer_path = buffer_path.expanduser().resolve()
    if buffer_path.is_file():
        return pandas.read_csv(buffer_path, sep="\t")

    logging.info("Loading relation names from Wikidata.")
    label_data = []
    for label in tqdm(relation_df["relation_label"].unique(), unit="relation", desc="Retrieving labels"):
        r = requests.get(url=f"https://www.wikidata.org/wiki/Property:{label}")
        title = re.search(r"<title>(.*) - Wikidata</title>", r.text).group(1)
        label_data.append((label, title))
    df = pandas.DataFrame(data=label_data, columns=["relation_label", "name"])
    df.to_csv(buffer_path, sep="\t", index=False)
    return df


@click.group()
def main():
    pass


@main.command()
@click.option("-i", "--input-path", type=str, default="../src/result.pt")
@click.option("-o", "--output-path", type=str, default="../data/degree.tsv")
def degree(
    input_path: str,
    output_path: str,
):
    logging.basicConfig(level=logging.INFO)

    # input normalization
    input_path = pathlib.Path(input_path).expanduser().resolve()
    output_path = pathlib.Path(output_path).expanduser().resolve()

    output_path = output_path.parent.joinpath(output_path.name.rsplit(".", maxsplit=1)[0] + "_" + input_path.name.replace("result_", "")).with_suffix(".tsv")
    logging.info(f"Updated output path to {output_path.as_uri()}")

    logging.info(f"Loading results from {input_path.as_uri()}")
    results = torch.load(input_path)

    inference: StatementsFactory = results["factories"]["inference"]
    statements = inference.mapped_statements

    # compute entity degree
    entity_degree = torch.zeros(inference.num_entities, dtype=torch.long)
    idx, count = statements[:, [0, 2]].unique(return_counts=True)
    entity_degree[idx] = count
    logging.info("Computed entity degrees")

    # get average degree of entity for each qualifier relation
    count = torch.zeros(inference.num_relations, dtype=torch.long)
    degree_sum = torch.zeros(inference.num_relations, dtype=torch.long)
    qr = statements[:, 3::2]
    qe = statements[:, 4::2]
    degree_sum.index_add_(dim=0, index=qr.flatten(), source=entity_degree[qe].flatten())
    count.index_add_(dim=0, index=qr.flatten(), source=torch.ones_like(qr).flatten())
    avg_degree = degree_sum / count.clamp_min(1)
    logging.info("Computed average entity degree for each qualifier relation")

    # create df
    data = []
    for relation_label, relation_id in inference.relation_to_id.items():
        data.append((
            relation_id,
            relation_label,
            avg_degree[relation_id].item(),
        ))
    df = pandas.DataFrame(data=data, columns=["relation_id", "relation_label", "avg_entity_degree"])
    logging.info(f"Created dataframe of shape {df.shape}")

    labels = _get_labels(relation_df=df)
    df = pandas.merge(df, labels, on="relation_label")
    logging.info(f"Added relation names")
    df.to_csv(output_path, sep="\t", index=False)


@main.command()
@click.option("-i", "--input-path", type=str, default="../src/result.pt")
@click.option("-o", "--output-path", type=str, default="../data/relation_importance.tsv")
@click.option("-b", "--eval-batch-size", type=int, default=128)
def measure(
    input_path: str,
    output_path: str,
    eval_batch_size: int,
):
    logging.basicConfig(level=logging.INFO)

    # input normalization
    input_path = pathlib.Path(input_path).expanduser().resolve()
    output_path = pathlib.Path(output_path).expanduser().resolve()

    logging.info(f"Loading results from {input_path.as_uri()}")
    results = torch.load(input_path)

    device = get_device()
    logging.info(f"Using device: {device}")

    # get model
    model: QualifierModel = results["model"]
    model = model.to(device=device)
    logging.info(f"Loaded model:\n{model}")

    # get inference & test statements
    factories: Mapping[str, StatementsFactory] = results["factories"]
    inference = factories["inference"] if "inference" in factories else None
    logging.info(f"Loaded statement factory: {inference}")

    testing = factories["testing"]
    logging.info(f"Loaded statement factory: {testing}")

    # send data to device
    inference_data_geometric: DataGeometric = None if inference is None else inference.create_data_object().to(device=device)
    old_qualifier_index: torch.LongTensor = None if inference is None else inference_data_geometric.qualifier_index
    testing_statements: torch.LongTensor = testing.mapped_statements.to(device=device)
    inference_statements = None if inference is None else inference.mapped_statements.to(device=device)

    # we need to mask relation + inverse
    relation_to_id = testing.relation_to_id
    id_to_relation = {v: k for k, v in relation_to_id.items()}
    full_df = []
    for relation_id in tqdm(range(0, max(relation_to_id.values()) + 1, 2), desc="relation", total=len(relation_to_id) // 2):
        # mask testing qualifier pairs
        this_testing_statements = testing_statements.clone()
        # special case for padding_idx
        if relation_id > 0:
            relation_id = relation_id - 1
        # get mask for qualifier pairs to mask
        mask = (this_testing_statements[:, :3:2] == relation_id) | (this_testing_statements[:, :3:2] == (relation_id + 1))
        # mask relation and entity of the pair
        this_testing_statements[:, :3:2][mask] = testing.padding_idx
        this_testing_statements[:, :4:2][mask] = testing.padding_idx

        # there are no inverse relations in qualifier index, so we only check for relation_id
        if inference_data_geometric is not None:
            q_mask = old_qualifier_index[0] == relation_id
            # only use qualifier pairs without this id
            inference_data_geometric.qualifier_index = old_qualifier_index[:, ~q_mask]

        # evaluate
        evaluator = _evaluate(
            model=model,
            mapped_statements=this_testing_statements,
            mapped_training_statements=inference_statements,
            filtered=inference_statements is not None,
            use_tqdm=False,
            batch_size=eval_batch_size,
            # test uses the inference edge index (aka "train inductive")
            data_geometric=inference_data_geometric,
        )
        df = evaluator.get_df(
            relation_id=relation_id,
            relation_label=id_to_relation[relation_id],
            masked=mask.any(dim=-1).cpu().numpy(),
            statement_id=numpy.arange(testing_statements.shape[0]),
        )
        full_df.append(df)
    full_df = pandas.concat(full_df, ignore_index=True)
    full_df.to_csv(output_path, sep="\t", index=False)
    logging.info(f"Stored results to {output_path.as_uri()}")


@main.command()
@click.option("-i", "--input-path", type=str, default="../data/relation_importance.tsv")
@click.option("-d", "--diff-path", type=str, default="../data/relation_importance_diff.tsv")
@click.option("-m", "--restrict-to-masked", is_flag=True)
def analyze(
    input_path: str,
    diff_path: str,
    restrict_to_masked: bool,
):
    logging.basicConfig(level=logging.INFO)

    # input normalization
    input_path = pathlib.Path(input_path).expanduser().resolve()
    diff_path = pathlib.Path(diff_path).expanduser().resolve()
    if restrict_to_masked:
        diff_path = diff_path.parent.joinpath(diff_path.name.replace(".tsv", "_restricted.tsv"))

    df = pandas.read_csv(input_path, sep="\t")
    logging.info(f"Loaded {df.shape[0]:,} measurements from {input_path.as_uri()}")

    relation_id_to_label = df[["relation_id", "relation_label"]].drop_duplicates()

    df["rank"] = 0.5 * (df["head_rank"] + df["tail_rank"])

    original_rank = df.loc[df["relation_id"] == 0, ["statement_id", "rank"]]

    if restrict_to_masked:
        df = df[df["masked"]]
    masked = df.loc[:, ["statement_id", "relation_id", "rank"]]
    diff_data = []
    for relation_id, group in masked.groupby(by="relation_id"):
        s = pandas.merge(
            left=original_rank,
            right=group,
            on="statement_id",
            suffixes=("_original", "_masked"),
        )
        s["rank_diff"] = s["rank_original"] - s["rank_masked"]
        diff_data.append(s)
    diff_df = pandas.concat(diff_data, ignore_index=True)
    diff_df = pandas.merge(left=diff_df, right=relation_id_to_label, on="relation_id", how="left")
    diff_df.to_csv(diff_path, sep="\t", index=False)
    logging.info(f"Written differences to {diff_path.as_uri()}")


@main.command()
@click.option("-d", "--diff-path", type=str, default="../data/relation_importance_diff.tsv")
@click.option("-k", "--top-k", type=int, default=5)
@click.option("-m", "--restrict-to-masked", is_flag=True)
def visualize(
    diff_path: str,
    top_k: int,
    restrict_to_masked: bool,
):
    logging.basicConfig(level=logging.INFO)
    diff_path = pathlib.Path(diff_path).expanduser().resolve()
    if restrict_to_masked:
        diff_path = diff_path.parent.joinpath(diff_path.name.replace(".tsv", "_restricted.tsv"))

    # create plot
    diff_df = pandas.read_csv(diff_path, sep="\t")
    description: pandas.DataFrame = diff_df.groupby(by="relation_id")["rank_diff"].describe().loc[:, ["count", "mean"]].reset_index().sort_values(by="mean", ascending=False)

    relation_id_to_label = diff_df[["relation_id", "relation_label"]].drop_duplicates()
    label_df = _get_labels(relation_df=relation_id_to_label)
    relation_id_to_label = pandas.merge(left=relation_id_to_label, right=label_df, on="relation_label", how="left")
    description = pandas.merge(left=description, right=relation_id_to_label, on="relation_id", how="left")

    # throw out padding
    description = description[description["relation_label"] != PADDING]

    top_k = pandas.concat([
        description.nlargest(n=top_k, columns=["mean"]),
        description.nsmallest(n=top_k, columns=["mean"]),
    ], axis=0, ignore_index=True)[["relation_label", "name", "mean"]].sort_values(by="mean", ascending=False)
    print(top_k.to_latex(index=False, float_format="{0:2.2f}".format))
    print(top_k.to_markdown(index=False))

    description["i"] = numpy.arange(description.shape[0])
    max_count = description["count"].max()
    golden = (1 + 5 ** 0.5) / 2
    height = 3
    plt.rc("font", **dict(size=13))
    f, ax = plt.subplots(figsize=(height * golden, height))
    ax: plt.Axes
    cmap = cm.get_cmap("RdBu_r")
    good = cmap(1.0)
    bad = cmap(-1.0)
    color = numpy.empty(shape=(description.shape[0], 4))
    improvement = description["mean"]
    color[improvement < 0, :3] = bad[:3]
    color[improvement >= 0, :3] = good[:3]
    color[:, 3] = (description["count"] / max_count) ** 0.2
    ax.bar(
        description["i"],
        height=improvement,
        color=color,
    )
    ax.ticklabel_format(axis="y", style="scientific", scilimits=(0, 0), useMathText=True)
    ax.grid()
    ax.set_xlim(-1, description.shape[0])

    # add annotation
    idx = pandas.DataFrame(data=[
        # best
        description.iloc[0],
        # worst
        description.iloc[-1],
        # most frequent improvement
        description[improvement > 0].iloc[description[improvement > 0]["count"].argmax()],
        # most frequent decrease
        description[improvement < 0].iloc[description[improvement < 0]["count"].argmax()],
    ])[["relation_id", "i"]].astype(int).drop_duplicates()
    poi = pandas.merge(relation_id_to_label, idx, on="relation_id")
    poi = pandas.merge(left=poi, right=description[["relation_id", "mean", "count"]], on="relation_id")

    for i, (_, row) in enumerate(poi.sort_values(by="mean").iterrows()):
        mean_diff = row["mean"]
        # i: 0-3
        yo = 0.2 + i * 0.19
        xo = 0.6 - 0.1 * i
        if i == 1:
            yo -= 0.1
            xo -= 0.1
        name = row["name"]
        count = int(row["count"])
        ax.annotate(
            text=f"{name}",
            xy=[row["i"], mean_diff],
            # xytext=(-15, 5),
            # textcoords="offset pixels",
            textcoords="figure fraction",
            xytext=[xo, yo],
            # horizontalalignment='center',
            verticalalignment='center',
            arrowprops=dict(arrowstyle="->"),
            # fontsize=14,
        )
    ax.set_xlabel("Qualifying relation")
    ax.set_ylabel(r"$\Delta MR$")
    ax.axhline(0, color="black")
    ax.set_xticks([])
    plt.tight_layout()
    plt.savefig("qualifier_relation_importance.pdf")
    plt.show()


if __name__ == '__main__':
    main()
