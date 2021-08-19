"""Utilities to download precomputed features."""
import functools
import logging
import pathlib
import pickle
from dataclasses import dataclass
from typing import Mapping, Optional, Tuple, Union

import numpy as np
from class_resolver import Resolver
from google_drive_downloader import GoogleDriveDownloader

from .split import generate_semi_inductive_split, get_semi_inductiveness_violations
from ..utils import PADDING, download_if_necessary, load_rdf_star_statements, resolve_cache_root

__all__ = [
    "get_node_features",
    "get_remapped_node_features",
    "dataset_resolver",
]

logger = logging.getLogger(__name__)
V1 = 'v1'
V2 = 'v2'


class Dataset:
    """A collection of Google Drive file IDs."""

    #: The name
    name: str

    #: The entity features: index file
    features_index: str

    #: The entity features: embedding file
    features_emb: str

    #: The statements
    inductive_statements: str

    #: Full transductive statements
    transductive_statements: str

    def get_path(
        self,
        split: str,
        part: str,
        version: Optional[str] = None,
        force: bool = False,
    ) -> pathlib.Path:
        """
        Get path to the file with statements.

        :param split:
            The dataset split. From {"inductive", "transductive"}.
        :param part:
            The part. From {"train", "inference", "validation", "test"}
        :param version:
            The dataset version. Only effective for inductive split.
        :param force:
            Whether to enforce re-downloading.
        """
        if split == "transductive":
            file_id = self.transductive_statements
            root = download_statements(dataset=self.name, file_id=file_id, force=force)
            transductive_file_names = dict(
                train="train.txt",
                validation="valid.txt",
                test="test.txt",
            )
            if part not in transductive_file_names.keys():
                raise ValueError(f"Invalid part: {part}. Allowed are: {sorted(transductive_file_names.keys())}")
            file_name = transductive_file_names[part]
            return root.joinpath("transductive", "statements", file_name)

        if split == 'semi_inductive':
            root = resolve_cache_root(self.name)

            # verify existence of split
            if len(list(root.joinpath('semi_inductive').iterdir())) != 3 or force:
                logger.warning("Did not find semi-inductive split - creating one.")
                create_semi_inductive_split(dataset_name=self.name)
                # raise FileNotFoundError("Semi-inductive set not available. Please re-run split.")

            # verify part and extract file name
            semi_inductive_file_names = dict(
                train="si_train.txt",
                test="si_test.txt",
                validation="si_valid.txt",
            )
            if part not in semi_inductive_file_names.keys():
                raise ValueError(f"Invalid part: {part}. Allowed are: {sorted(semi_inductive_file_names.keys())}")
            file_name = semi_inductive_file_names[part]

            return root.joinpath("semi_inductive", file_name)

        if split == "inductive":
            # verify version
            if version not in {V1, V2}:
                raise ValueError(f"Dataset version must be either {V1} or {V2}, but received {version}")

            # verify part and extract file name
            inductive_file_names = dict(
                train="transductive_train.txt",
                inference="inductive_train.txt",
                validation="inductive_val.txt",
                test="inductive_ts.txt",
            )
            if part not in inductive_file_names:
                raise ValueError(f"Invalid part: {part}. Allowed are: {sorted(inductive_file_names.keys())}")
            file_name = inductive_file_names[part]

            # ensure that files are downloaded
            file_id = self.inductive_statements
            cache_root = download_statements(dataset=self.name, file_id=file_id, force=force)

            # return path to file
            return cache_root.joinpath("inductive", "statements", version, file_name)

        raise ValueError(f"Unknown split: {split}")


@dataclass
class WD50(Dataset):
    """The original dataset."""
    name = "wd50"
    features_index = "1FrDuTkN2-eGhw5OTlyZ33PSvpQhtbEhU"
    features_emb = "1lu4DtBukbB4JejRxY_yqTO0ZPorKSdJg"
    inductive_statements = "16r96dCIR1jwL3J5JK5HCpDCY1ac2jMo7"
    transductive_statements = "1XQgcRji-T_uwGf4-R1AYXKUI0nTcSU_f"


@dataclass
class WD50_33(Dataset):
    name = "wd50_33"
    features_index = "1AuVUq2QepQ-xkKIdscEc3Cn-G2BoSIAf"
    features_emb = "1usHz9oryz8ShMg6jyz4jgjMnXN5uva3u"
    inductive_statements = "14OfkLfOWCUZ4XU-7Dr8aTkdPaREX5Xqi"
    transductive_statements = "1KJTzHUcHs6f92nWxpdmJOBIj_56FJPLr"


@dataclass
class WD50_66(Dataset):
    name = "wd50_66"
    features_index = "1CbpONOWXb5rWWlTtZJUPJeZpvlV69S2I"
    features_emb = "1yFDrAnYOWYmXDebm_mkgdGaVFyQhTwgZ"
    inductive_statements = "1jTP7Lrrs0B92wT6Fv2OSo7Jt-MfZgotW"
    transductive_statements = "1838eqrsCpxCX-0QcJvOUFKvCIs_tFP_X"


@dataclass
class WD50_100(Dataset):
    name = "wd50_100"
    features_index = "1BIXyH8Svxa-cBrraTxL6kz_OWHfK9qnu"
    features_emb = "1My2SIkk19G6nGHR5irDvxnK8AEDKQLi7"
    inductive_statements = "1-uhu7DkNs1fYyYgDqb5nS1GVDvsRmCV9"
    transductive_statements = "1eQ7ZKydEAUQMpZOfG3msxMmJHfUAYZ8-"


dataset_resolver = Resolver.from_subclasses(
    base=Dataset,
)

DATASET_NAMES = dataset_resolver.lookup_dict.keys()


def download_statements(
    dataset: str,
    force: bool = False,
    file_id: str = None,
) -> pathlib.Path:
    """Download statements for the given dataset."""
    dataset = dataset_resolver.lookup(dataset).name
    cache_root = resolve_cache_root(dataset)
    zip_file_path = cache_root.joinpath(dataset)
    if not cache_root.joinpath('inductive').exists() or len(list(cache_root.joinpath('inductive').iterdir())) == 0 or force:
        GoogleDriveDownloader.download_file_from_google_drive(
            file_id=file_id,
            dest_path=str(zip_file_path),
            overwrite=force,
            unzip=True,
            showsize=True,
        )

    # TODO: Why?
    if zip_file_path.is_file():
        zip_file_path.unlink()

    return cache_root


# cache
@functools.lru_cache(maxsize=1)
def get_node_features(
    dataset: str,
    cache_root: Union[None, str, pathlib.Path] = None,
    force: bool = False,
) -> Tuple[Mapping[str, int], np.ndarray]:
    """
    Get node features.

    Downloads file to cache root if necessary.

    :param dataset:
        The dataset name. Must be in DATASET_NAMES.
    :param cache_root:
        The root directory for caching.
    :param force:
        Enforce re-downloading even if files exist.

    :return:
        A tuple (label_to_id, embeddings) of the mapping from labels to corresponding IDs, and the actual embeddings.
    """
    # resolve dataset
    dataset = dataset_resolver.make(dataset)

    # get Google Drive file IDs
    index_id, emb_id = dataset.features_index, dataset.features_emb

    # Load index
    index_file_path = download_if_necessary(
        dataset.name,
        "index.txt",
        cache_root=cache_root,
        file_id=index_id,
        force=force,
    )
    with index_file_path.open("r") as index_file:
        label_to_id = {
            label: i
            for i, label in enumerate(
                line.strip() for line in index_file.readlines()
            )
        }
    logger.info(f"Loaded index file with {len(label_to_id)} entries.")

    # Load embeddings
    emb_file_path = download_if_necessary(
        dataset.name,
        "embeddings.pkl",
        cache_root=cache_root,
        file_id=emb_id,
        force=force,
    )
    # Load embeddings
    with emb_file_path.open("rb") as pickle_file:
        emb: np.ndarray = pickle.load(pickle_file)
    logger.info(f"Loaded embeddings of shape {emb.shape}")
    if len(emb) != len(label_to_id):
        raise AssertionError(
            f"The number of embeddings ({len(emb)}) does not correspond to the number of labels ({len(label_to_id)})."
        )

    return label_to_id, emb


def get_remapping_index(
    old_label_to_id: Mapping[str, int],
    new_label_to_id: Mapping[str, int],
) -> np.ndarray:
    """
    Create a remapping index based on two label-to-ID mappings for the same labels.

    :param old_label_to_id:
        The old mapping which is expected to be a superset of new_label_to_id (without the 'padding' entry).
    :param new_label_to_id:
        The new mapping.

    :return: shape: (n,)
        An index to remap embeddings from the old IDs to the new IDs by embeddings[index].
    """

    num = len(new_label_to_id)
    if set(new_label_to_id.values()) != set(range(num)):
        raise ValueError("The IDs are not consecutive.")

    result = np.empty(shape=(num,), dtype=np.int32)
    for label, new_id in new_label_to_id.items():
        result[new_id] = old_label_to_id[label]
    return result


def get_remapped_node_features(
    dataset: str,
    label_to_id: Mapping[str, int],
    cache_root: Union[None, str, pathlib.Path] = None,
    drive_label_to_id: Optional[Mapping[str, int]] = None,
    emb: np.ndarray = None,
    force: bool = False,
) -> np.ndarray:
    """
    Get node features and ensures that they use the provided label-to-id mapping.

    Downloads file to cache root if necessary.

    :param dataset:
        The dataset name. Must be in DATASET_NAMES.
    :param label_to_id:
        The label-to-id mapping which shall be used. The translation uses the labels to convert from old IDs to new IDs.
    :param cache_root:
        The root directory for caching.
    :param force:
        Enforce re-downloading even if files exist.

    :return: shape: (n, d)
        The embeddings.
    """
    if drive_label_to_id is None or emb is None:
        drive_label_to_id, emb = get_node_features(
            dataset=dataset,
            cache_root=cache_root,
            force=force,
        )
    if PADDING not in drive_label_to_id:
        drive_label_to_id[PADDING] = max(drive_label_to_id.values()) + 1

    index = get_remapping_index(old_label_to_id=drive_label_to_id, new_label_to_id=label_to_id)
    # Add nodes features for padding node
    emb = np.concatenate([emb, np.zeros(shape=(1, emb.shape[-1]))], axis=0)
    return emb[index]


def create_semi_inductive_split(dataset_name: str) -> None:
    """Create semi-inductive split."""
    # TODO: Automatically call this?
    dataset = dataset_resolver.make(query=dataset_name)
    input_root = download_if_necessary(
        dataset.name,
        "transductive.zip",
        file_id=dataset.transductive_statements,
        unzip=True,
    ).with_suffix(suffix="") / "statements"
    output_root = resolve_cache_root(dataset_name, "semi_inductive")

    # combine all statements
    statements = load_rdf_star_statements(path=[
        input_root / f"{part}.txt"
        for part in ("train", "valid", "test")
    ])
    split, e_train = generate_semi_inductive_split(statements=statements, ratios=(0.9, 0.05))
    violations = get_semi_inductiveness_violations(*split, e_train=e_train)
    if violations:
        raise AssertionError(violations)
    # TODO: re-use Dataset.get_path
    for statements, file_name in zip(split, ['si_train.txt', 'si_valid.txt', 'si_test.txt']):
        with (output_root / file_name).open("w") as f:
            f.writelines(
                ",".join(s) + "\n"
                for s in statements
            )
