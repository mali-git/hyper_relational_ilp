import logging
import pathlib
from typing import List, Optional, Union

import pystow
import torch
from google_drive_downloader import GoogleDriveDownloader
from torch.nn import Parameter
from torch.nn.init import xavier_normal_
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_scatter import scatter_add, scatter_max

DATASET_NAMES = frozenset(["wd50", "wd50_33", "wd50_66", "wd50_100"])
PADDING = '__padding__'
logger = logging.getLogger(__name__)

HERE = pathlib.Path(__file__)


def get_loss(loss):
    if loss == "MarginRankingLoss()" or loss == 'mrl':
        return torch.nn.MarginRankingLoss()
    elif loss == 'BCEWithLogitsLoss()':
        return torch.nn.BCEWithLogitsLoss()
    else:
        raise ValueError(f"Loss {loss} not supprted.")


def extract_dataset_name(name):
    for dataset_name in DATASET_NAMES:
        if dataset_name in name:
            return dataset_name

    raise ValueError(f"Dataset {name} not supported.")


def extract_dataset_version(name):
    for version in ['v1', 'v2']:
        if version in name:
            return version

    raise ValueError(f"Dataset version {name} not supported.")


def resolve_cache_root(
    *sub_directories: str,
    cache_root: Union[None, str, pathlib.Path] = None,
) -> pathlib.Path:
    """Resolve the cache root directory and make sure it exists."""
    if cache_root is None:
        cache_root = pystow.join("ilp")
    if isinstance(cache_root, str):
        cache_root = pathlib.Path(cache_root)
    cache_root = cache_root.expanduser().resolve()
    cache_root = cache_root.joinpath(*sub_directories)
    cache_root.mkdir(exist_ok=True, parents=True)
    return cache_root


def load_rdf_star_statements(
    path: Union[str, pathlib.Path, List[str], List[pathlib.Path]],
    max_len: Optional[int] = None,
) -> List[List[str]]:
    """
    Load statements from a RDF* file.

    :param path:
        The path to the file.
    :param max_len:
        If given, trim statements to maximum length.

    :return:
        A list of statements in the format (h, r, t, *q), where q are qualifiers.
    """
    if isinstance(path, list):
        return sum((load_rdf_star_statements(p, max_len=max_len) for p in path), [])
    path = pathlib.Path(path).expanduser().resolve()
    statements = []
    with path.open('r') as f:
        for line in f.readlines():
            # list[:None] returns the full list
            statements.append(line.strip("\n").split(",")[:max_len])
    logger.info(f"Loaded {len(statements)} statements from {path.as_uri()}.")
    return statements


def _download_if_necessary(
    cache_root: pathlib.Path,
    file_id: str,
    file_name: str,
    force: bool = False,
    unzip: bool = False,
) -> pathlib.Path:
    """Download a Google Drive file to cache root, if necessary."""
    file_path = cache_root / file_name
    if not file_path.is_file() or force:
        GoogleDriveDownloader.download_file_from_google_drive(
            file_id=file_id,
            dest_path=str(file_path),
            overwrite=force,
            unzip=unzip,
            showsize=True,
        )
    return file_path


def download_if_necessary(
    *path: str,
    file_id: str,
    cache_root: Union[None, str, pathlib.Path] = None,
    force: bool = False,
    unzip: bool = False,
) -> pathlib.Path:
    """Download a Google Drive file to cache_root, if necessary."""
    cache_root = resolve_cache_root(*path[:-1], cache_root=cache_root)
    file_name = path[-1]
    force = force
    return _download_if_necessary(
        cache_root=cache_root,
        file_id=file_id,
        file_name=file_name,
        force=force,
        unzip=unzip,
    )


def get_device():
    # Define device
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def get_parameter(shape) -> torch.nn.Parameter:
    """."""
    param = Parameter(torch.empty(*shape))
    xavier_normal_(param.data)
    return param


def softmax(src, index, num_nodes=None):
    """Computes a sparsely evaluated softmax.
    Given a value tensor :attr:`src`, this function first groups the values
    along the first dimension based on the indices specified in :attr:`index`,
    and then proceeds to compute the softmax individually for each group.
    Args:
        src (Tensor): The source tensor.
        index (LongTensor): The indices of elements for applying the softmax.
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`index`. (default: :obj:`None`)
    :rtype: :class:`Tensor`
    """

    num_nodes = maybe_num_nodes(index, num_nodes)

    out = src - scatter_max(src, index, dim=0, dim_size=num_nodes)[0][index]
    out = out.exp()
    out = out / (
        scatter_add(out, index, dim=0, dim_size=num_nodes)[index] + 1e-16)

    return out
