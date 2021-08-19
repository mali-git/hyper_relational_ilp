"""Training instances as PyTorch datasets."""
from dataclasses import dataclass
from typing import Mapping

import numpy as np
import torch
from torch.utils import data

__all__ = [
    "LCWAInstances",
    "SLCWAInstances",
]


@dataclass
class Instances(data.Dataset):
    """Triples and mappings to their indices."""

    #: A PyTorch tensor of triples
    mapped_statements: torch.Tensor

    #: A mapping from relation labels to integer identifiers
    entity_to_id: Mapping[str, int]

    #: A mapping from relation labels to integer identifiers
    relation_to_id: Mapping[str, int]

    @property
    def num_instances(self) -> int:  # noqa: D401
        """The number of instances."""
        return self.mapped_statements.shape[0]

    @property
    def num_entities(self) -> int:  # noqa: D401
        """The number of entities."""
        return len(self.entity_to_id)

    def __len__(self):  # noqa: D105
        return self.num_instances


@dataclass
class SLCWAInstances(Instances):
    """Triples and mappings to their indices for sLCWA."""

    #: The entity IDs from which to draw negative samples
    entities: torch.LongTensor

    def __len__(self):  # noqa: D105
        return self.mapped_statements.shape[0]

    def __getitem__(self, item):  # noqa: D105
        return self.mapped_statements[item]


@dataclass
class LCWAInstances(Instances):
    """Triples and mappings to their indices for LCWA."""

    labels: np.ndarray

    def __getitem__(self, item):  # noqa: D105
        # Create dense target
        batch_labels_full = torch.zeros(self.num_entities)
        batch_labels_full[self.labels[item]] = 1
        return self.mapped_statements[item], batch_labels_full
