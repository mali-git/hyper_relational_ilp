"""Evaluate link prediction ranks for different number of qualifiers."""
import logging
import pathlib
from typing import Mapping, Optional

import click
import pandas
import scipy.stats
import seaborn
import torch
from matplotlib import pyplot as plt
from torch_geometric.data.data import Data as DataGeometric
from tqdm.auto import tqdm

from ilp.data.statement_factory import StatementsFactory
from ilp.evaluation import _evaluate
from ilp.models.base import QualifierModel
from ilp.utils import get_device


@click.group()
def main():
    pass


@main.command()
@click.option('-p', '--input-path', type=str, default='~/qualifier_ranks.tsv')
@click.option('-a', '--output-path', type=str, default=None)
@click.option('-f', '--figure-path', type=str, default=None)
@click.option("-l", "--log-scale", is_flag=True)
def first(
    input_path: str,
    output_path: Optional[str],
    figure_path: Optional[str],
    log_scale: bool,
):
    logging.basicConfig(level=logging.INFO)

    # Input parameter normalization
    input_path = pathlib.Path(input_path).expanduser().resolve()
    name, ext = input_path.name.rsplit(sep=".", maxsplit=1)
    if output_path is None:
        output_path = input_path.with_name(name=name + "_aggregated." + ext)
    output_path = pathlib.Path(output_path).expanduser().resolve()
    if figure_path is None:
        figure_path = input_path.with_name(name=name + "_aggregated." + "pdf")

    # Load rank data
    df = pandas.read_csv(input_path, sep="\t")
    logging.info(f"Loaded rank data from {input_path.as_uri()}: {df.shape}")

    print(df.nlargest(n=10, columns=["head_rank"]))
    print(df.nlargest(n=10, columns=["tail_rank"]))

    # create plot of rank distribution
    df = pandas.concat([
        df.loc[:, ["num_qualifiers", "head_rank"]].rename(columns=dict(head_rank="rank")),
        df.loc[:, ["num_qualifiers", "tail_rank"]].rename(columns=dict(tail_rank="rank")),
    ], ignore_index=True)
    # df["rank"] = 0.5 * (df["head_rank"] + df["tail_rank"])
    golden = (1 + 5 ** 0.5) / 2
    seaborn.catplot(data=df, y="rank", x="num_qualifiers", kind="box", height=3, aspect=golden)
    plt.xlabel("number of qualifier pairs")
    y_label = "rank"
    if log_scale:
        y_label += " [log]"
        plt.yscale("log")
    plt.ylabel(y_label)  # , fontdict=dict(size=16))
    plt.tight_layout()
    plt.savefig(figure_path)
    logging.info(f"Written figure to {figure_path.as_uri()}")
    plt.show()

    # aggregate ranks grouped by num_qualifiers
    data = []
    no_qualifier = df.loc[df["num_qualifiers"] == 0, "rank"]
    for nq, sdf in df.groupby(by="num_qualifiers"):
        ranks = sdf["rank"]
        # TODO: the difference here can be either caused by having less qualifiers OR by having more difficult triples.
        _, pval = scipy.stats.ttest_ind(no_qualifier, ranks, equal_var=False)
        res = dict(
            num_qualifiers=nq,
            count=len(ranks),
            mrr=(1 / ranks).mean(),
            mr=ranks.mean(),
            p=pval,
        )
        for k in (1, 3, 5, 10):
            res[f"hits_at_{k}"] = (ranks <= 10).mean()
        data.append(res)
    df = pandas.DataFrame(data=data).set_index("num_qualifiers")
    df.to_csv(output_path, sep="\t")
    logging.info(f"Written summary to {output_path.as_uri()}")


@main.command()
@click.option("-i", "--input-path", type=pathlib.Path, default="../src/result.pt")
@click.option("-o", "--output-path", type=pathlib.Path, default="../data/num_qualifier_data.tsv")
@click.option("-b", "--eval-batch-size", type=int, default=128)
def trim(
    input_path: pathlib.Path,
    output_path: pathlib.Path,
    eval_batch_size: int,
):
    logging.basicConfig(level=logging.INFO)

    input_path = input_path.expanduser().resolve()
    output_path = output_path.expanduser().resolve()

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
    testing_statements: torch.LongTensor = testing.mapped_statements.to(device=device)
    mapped_training_statements = None if inference is None else inference.mapped_statements.to(device=device)

    full_df = []
    for i in tqdm(range(testing.max_num_qualifier_pairs + 1), total=testing.max_num_qualifier_pairs, unit="max_qualifier"):
        this_testing_statements = testing_statements[:, :3 + 2 * i]
        # evaluate
        evaluator = _evaluate(
            model=model,
            mapped_statements=this_testing_statements,
            mapped_training_statements=mapped_training_statements,
            filtered=mapped_training_statements is not None,
            use_tqdm=False,
            batch_size=eval_batch_size,
            # test uses the inference edge index (aka "train inductive")
            data_geometric=inference_data_geometric,
        )
        df = evaluator.get_df(max_num_qualifier_pairs=i)
        full_df.append(df)
    full_df = pandas.concat(full_df, ignore_index=True)
    full_df.to_csv(output_path, sep="\t", index=False)
    logging.info(f"Stored results to {output_path.as_uri()}")


@main.command()
@click.option("-i", "--input-root", type=pathlib.Path, default="../data")
@click.option("-f", "--figure-path", type=pathlib.Path, default="../data/num_qualifiers.pdf")
def viz(
    input_root: pathlib.Path,
    figure_path: pathlib.Path,
):
    logging.basicConfig(level=logging.INFO)
    input_root = input_root.expanduser().resolve()
    figure_path = figure_path.expanduser().resolve()

    # Load rank data
    dfs = []
    for p in input_root.glob("*num_qualifier_data.tsv"):
        df = pandas.read_csv(p, sep="\t")
        logging.info(f"Loaded rank data from {p.as_uri()}: {df.shape}")
        df["dataset"] = p.name.replace("_num_qualifier_data.tsv", "")
        dfs.append(df)
    df = pandas.concat(dfs, ignore_index=True)

    df = pandas.concat([
        df.loc[:, ["max_num_qualifier_pairs", "dataset", "head_rank"]].rename(columns=dict(head_rank="rank")),
        df.loc[:, ["max_num_qualifier_pairs", "dataset", "tail_rank"]].rename(columns=dict(tail_rank="rank")),
    ], ignore_index=True)

    dataset_normalization = {
        "wd50_100_v1": "FI WD20K (100) V1",
        "wd50_66_v2": "FI WD20K (66) V2",
        "si_qblp_100_v1": "SI WD20K (33)",
    }

    df["model"] = df["dataset"].apply(lambda x: "QBLP" if "qblp" in x else "StarE")
    df["dataset"] = df["dataset"].apply(dataset_normalization.__getitem__)

    f = seaborn.relplot(
        data=df,
        x="max_num_qualifier_pairs",
        y="rank",
        kind="line",
        hue="dataset",
        style="model",
        height=3,
        aspect=1,
    )
    f.set(
        ylim=(0, None),
        xticks=range(df["max_num_qualifier_pairs"].max() + 1),
        xlabel="maximum number of qualifiers",
        xlim=(0, 3),
        # yscale="log",
    )
    plt.tight_layout()
    plt.grid()
    plt.savefig(figure_path)
    logging.info(f"Written figure to {figure_path.as_uri()}")
    plt.show()


if __name__ == '__main__':
    main()
