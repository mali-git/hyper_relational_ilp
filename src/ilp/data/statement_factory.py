# -*- coding: utf-8 -*-

"""A graph factory that creates the PyTorchGeometric data object, and holds references to the textual descriptions."""
import functools
import logging
import pathlib
import re
from collections import defaultdict
from typing import Collection, Dict, Iterable, List, Mapping, Optional, Tuple, Union

import numpy as np
import torch
from torch_geometric.data import Data
from tqdm import tqdm

from .features import dataset_resolver, get_node_features, get_remapped_node_features
from .instances import LCWAInstances, SLCWAInstances
from .split import filter_qualifiers
from ..utils import PADDING, load_rdf_star_statements

logger = logging.getLogger(__name__)

INVERSE_SUFFIX = '_inverse'
TRIPLES_DF_COLUMNS = ('head_id', 'head_label', 'relation_id', 'relation_label', 'tail_id', 'tail_label')


def _map_statements(
    statements: np.ndarray,
    entity_to_id,
    relation_to_id,
    max_num_qualifier_pairs: int,
) -> Tuple[torch.LongTensor, Collection[np.ndarray]]:
    values = set(list(entity_to_id.keys()) + list(relation_to_id.keys()))
    statements_filtered = []
    mapped_statements = []
    max_length = max_num_qualifier_pairs * 2 + 3
    for s in statements:
        unique_values_in_s = set(s).difference(values)
        if len(unique_values_in_s) != 0:
            continue
        statements_filtered.append(s)
        # Map to id
        mapped_s = []
        for i, v in enumerate(s):
            if i % 2 == 0:
                mapped_s.append(entity_to_id[v])
            else:
                mapped_s.append(relation_to_id[v])
        # Pad statement
        if max_length >= len(mapped_s):
            stop = max_length - len(mapped_s)
            mapped_s = mapped_s + [entity_to_id[PADDING] for _ in range(stop)]
        else:
            mapped_s = mapped_s[:max_length]

        mapped_statements.append(mapped_s)
    mapped_statements = torch.as_tensor(data=mapped_statements, dtype=torch.long)
    return mapped_statements, statements_filtered


class StatementsFactory:
    """Create graph and save it as torch_geometric.data."""

    mapped_qualifiers: torch.LongTensor
    triple_to_idx: Mapping[Tuple[int, int, int], Collection[int]]
    head_relation_to_id: Mapping[Tuple[int, int], Collection[int]]
    tail_relation_to_id: Mapping[Tuple[int, int], Collection[int]]
    padding_idx = 0

    def __init__(
        self,
        path: Union[None, str, pathlib.Path, List[str], List[pathlib.Path]] = None,
        dataset_name: str = None,
        statements: List[List[str]] = None,
        max_num_qualifier_pairs: int = None,
        create_inverse_triples: bool = False,
        entity_to_id: Mapping[str, int] = None,
        relation_to_id: Mapping[str, int] = None,
        use_node_features: bool = True,
        feature_dim: Optional[int] = None,
        enforce_node_features: bool = False,
        is_inductive: bool = False,
    ) -> None:
        """
        Initialize the factory.

        :param path:
            path to the graph file
        :param dataset_name:
            The dataset name.
        :param statements:
            directly input triples (optional)
        :param entity_to_id:
            mapping from labels to a numeric id. If None, infer from statements.
        :param relation_to_id:
            mapping relation type to an numeric id. If None, infer from statements.
        :param use_node_features:
            whether to use node features or generate random ones.
        :param feature_dim:
            dimension of random features for nodes and edges.
        """
        self.path = path
        if path is not None:
            self.statements = load_rdf_star_statements(path=path, max_len=max_num_qualifier_pairs * 2 + 3)
        else:
            if statements is None:
                raise ValueError("You must explictly provide statements if no path is given.")
            self.statements = statements

        self.create_inverse_triples = create_inverse_triples
        self._process_inverse_relations()
        self.max_num_qualifier_pairs = max_num_qualifier_pairs

        if use_node_features:
            drive_label_to_id = None
            emb = None
            if enforce_node_features:
                drive_label_to_id, emb = get_node_features(
                    dataset=dataset_name,
                    cache_root=None,
                    force=False,
                )

                nodes_with_features = set(drive_label_to_id.keys())

                # Filter statements based on node features
                filtered_statements = []

                known_relations = None if relation_to_id is None else set(relation_to_id.keys())

                for s in self.statements:
                    # convert to tuple
                    s = tuple(s)
                    # if head or tail have no features -> skip statement
                    if not nodes_with_features.issuperset(s[:3:2]):
                        continue
                    # if any relation is unknown:
                    if is_inductive:
                        if s[1] not in known_relations:
                            continue
                    # filter qualifiers by entities
                    hrt, qs = s[:3], s[3:]
                    qs = filter_qualifiers(qualifiers=qs, allowed=nodes_with_features, entities=True)
                    # filter qualifiers by relations
                    if known_relations is not None:
                        qs = filter_qualifiers(qualifiers=qs, allowed=known_relations, entities=False)
                    filtered_statements.append(hrt + qs)

                self.statements = filtered_statements

        # Generate entity mapping if necessary
        if not is_inductive:
            if entity_to_id is None or relation_to_id is None:
                self._create_mappings_rdf_star()
            else:
                self.entity_to_id = entity_to_id
                self.relation_to_id = relation_to_id
        else:
            if entity_to_id is None:
                self._create_mappings_rdf_star(create_relation_to_id=False)
            else:
                self.entity_to_id = entity_to_id
            if relation_to_id is None:
                raise ValueError("A inductive setting is defined, but the relation_to_id mapping is not provided.")

            self.relation_to_id = relation_to_id

        # Note: Mapped statements are not padded
        self.mapped_statements, self.statements_filtered = _map_statements(
            statements=self.statements,
            entity_to_id=self.entity_to_id,
            relation_to_id=self.relation_to_id,
            max_num_qualifier_pairs=max_num_qualifier_pairs,
        )

        # load node features
        if use_node_features:
            self.node_feature_tensor = torch.as_tensor(
                data=get_remapped_node_features(
                    dataset=dataset_name,
                    label_to_id=self.entity_to_id,
                    drive_label_to_id=drive_label_to_id,
                    emb=emb,
                    # TODO: Allow to provide this via constructor parameter?
                    cache_root=None,
                    force=False,
                ),
                dtype=torch.get_default_dtype(),
            )
            dim = self.node_feature_tensor.shape[-1]
            if feature_dim is not None and dim != feature_dim:
                logger.warning(
                    f"Using pre-trained features with dim {dim}, although feature_dim={feature_dim} was passed. "
                    f"To disable this warning pass feature_dim=None."
                )
        else:
            if feature_dim is None:
                raise ValueError("Must provide feature_dim if random features shall be used.")
            self.node_feature_tensor = torch.rand(
                self.num_entities,
                feature_dim,
            )
        self.feature_dim = self.node_feature_tensor.shape[-1]
        self.max_num_qualifier_pairs = max_num_qualifier_pairs

        # store non qualifier only entities
        self.non_qualifier_only_entities = self.mapped_statements[:, [0, 2]].unique()

    def _process_inverse_relations(self):
        relations = list(sorted(set([e for s in self.statements for e in s[1::2]])))

        # Check if the triples are inverted already
        relations_already_inverted = self._check_already_inverted_relations(relations)

        if self.create_inverse_triples or relations_already_inverted:
            self.create_inverse_triples = True
            if relations_already_inverted:
                logger.info(
                    f'Some triples already have suffix {INVERSE_SUFFIX}. '
                    f'Creating TriplesFactory based on inverse triples',
                )
                self.relation_to_inverse = {
                    re.sub('_inverse$', '', relation): f"{re.sub('_inverse$', '', relation)}{INVERSE_SUFFIX}"
                    for relation in relations
                }

            else:
                self.relation_to_inverse = {
                    relation: f"{relation}{INVERSE_SUFFIX}"
                    for relation in relations
                }
                inverse_statements = [[s[2], self.relation_to_inverse[s[1]], s[0], *s[3:]] for s in self.statements]
                # extend original triples with inverse ones
                self.statements.extend(inverse_statements)

        else:
            self.create_inverse_triples = False
            self.relation_to_inverse = None

    @staticmethod
    def _check_already_inverted_relations(relations: Iterable[str]) -> bool:
        for relation in relations:
            if relation.endswith(INVERSE_SUFFIX):
                # We can terminate the search after finding the first inverse occurrence
                return True

        return False

    def create_data_object(self) -> Data:
        """Create PyTorch Geometric data object."""
        node_features = self._create_node_feature_tensor()
        edge_index = self._create_edge_index()
        qualifier_index = self._create_qualifier_index()
        edge_type = self._create_edge_type()

        entities = torch.as_tensor(
            data=sorted(set(self.entity_to_id.values())),  # exclude padding entity
            dtype=torch.long,
        )

        return Data(
            x=node_features,
            edge_index=edge_index,
            qualifier_index=qualifier_index,
            edge_type=edge_type,
            entities=entities,
        )

    def _compose_qualifier_batch(self, keys, mapping) -> torch.LongTensor:
        # batch.shape:
        batch_size = len(keys)

        # Lookup IDs of qualifiers
        qualifier_ids = [
            mapping.get(tuple(k.tolist()))
            for k in keys
        ]

        # Determine maximum length (for dynamic padding)
        max_len = max(map(len, qualifier_ids))

        # bound maximum length for guaranteed max memory consumption
        # TODO: num_qualifier_pairs = max_num_qualifier_pairs ?
        max_len = min(max_len, self.max_num_qualifier_pairs)

        # Allocate result
        result = torch.full(size=(batch_size, max_len), fill_value=-1, dtype=torch.long)

        # Retrieve qualifiers
        for i, this_qualifier_ids in enumerate(qualifier_ids):
            # limit number of qualifiers
            # TODO: shuffle?
            this_qualifier_ids = this_qualifier_ids[:max_len]
            result[i, :len(this_qualifier_ids)] = torch.as_tensor(data=this_qualifier_ids, dtype=torch.long)

        return result

    def _create_qualifier_index(self):
        """
        Create a COO matrix of shape (3, num_qualifiers). Only non-zero (non-padded) qualifiers are retained.
        row0: qualifying relations
        row1: qualifying entities
        row2: index row which connects a pair (qual_r, qual_e) to a statement index k
        :return: shape: (3, num_qualifiers)
        """
        if self.max_num_qualifier_pairs is None or self.max_num_qualifier_pairs == 0:
            return None

        qual_relations, qual_entities, qual_k = [], [], []

        # It is assumed that statements are already padded
        for triple_id, statement in enumerate(self.mapped_statements):
            qualifiers = statement[3:]
            entities = qualifiers[1::2]
            relations = qualifiers[::2]
            # Ensure that PADDING has id=0
            non_zero_rels = relations[np.nonzero(relations)]
            non_zero_ents = entities[np.nonzero(entities)]
            assert len(non_zero_rels) == len(non_zero_ents), \
                "Number of non-padded qualifying relations is not equal to the # of qualifying entities"
            num_qualifier_pairs = non_zero_ents.shape[0]

            for j in range(num_qualifier_pairs):
                qual_relations.append(non_zero_rels[j].item())
                qual_entities.append(non_zero_ents[j].item())
                qual_k.append(triple_id)

        qualifier_index = torch.stack([
            torch.tensor(qual_relations, dtype=torch.long),
            torch.tensor(qual_entities, dtype=torch.long),
            torch.tensor(qual_k, dtype=torch.long)
        ], dim=0)

        if self.create_inverse_triples:
            # qualifier index is the same for inverse statements
            qualifier_index[2, len(qual_relations) // 2:] = qualifier_index[2, :len(qual_relations) // 2]

        return qualifier_index

    def _create_node_feature_tensor(self) -> torch.Tensor:
        """Create the node feature tensor."""
        if self.node_feature_tensor is not None:
            return self.node_feature_tensor
        if self.one_hot_encoding:
            return torch.eye(n=len(self.entity_to_id))
        else:
            return torch.empty(len(self.entity_to_id), self.feature_dim)

    def _create_edge_index(self) -> torch.Tensor:
        """Create edge index where first row represents the source nodes and the second row the target nodes."""
        mapped_heads = self.mapped_statements[:, 0].view(1, -1)
        mapped_tails = self.mapped_statements[:, 2].view(1, -1)
        edge_index = torch.cat([mapped_heads, mapped_tails], dim=0)
        return edge_index

    def _create_edge_type(self) -> torch.Tensor:
        """Create edge type tensor where each entry correspond to the relationship type of a triple in the dataset."""
        # Inverse triples are created in the base class

        return self.mapped_statements[:, 1]

    def _create_mappings_rdf_star(
        self,
        create_entity_to_id: bool = True,
        create_relation_to_id: bool = True,
    ):
        """."""

        max_length = self.max_num_qualifier_pairs * 2 + 3
        # Ensure that no entity/relation is part of the mappings that will not be seen
        statements = [s[:max_length] if len(s) > max_length else s for s in self.statements]

        if create_relation_to_id:
            relations = [PADDING] + list(sorted(set([r for s in statements for r in s[1::2]])))
            if self.create_inverse_triples:
                relations = [PADDING] + [elem for pair in
                                         zip(list(self.relation_to_inverse.keys()),
                                             list(self.relation_to_inverse.values()))
                                         for elem in pair]
                # old
                # relations = [PADDING] + list(self.relation_to_inverse.keys()) + (list(self.relation_to_inverse.values()))
            # When creating inverse relations, we will also generate padding_inverse which will never be used
            # unless we use score_r()
            self.relation_to_id = {relation: id for id, relation in enumerate(relations)}

        if create_entity_to_id:
            entities = [PADDING] + list(sorted(set([e for s in statements for e in s[::2]])))
            self.entity_to_id = {entity: id for id, entity in enumerate(entities)}

    def create_lcwa_instances(self, use_tqdm: Optional[bool] = None) -> LCWAInstances:
        """Create LCWA instances for this factory's statements."""
        s_p_q_to_multi_tails = _create_multi_label_tails_instance(
            mapped_statements=self.mapped_statements,
            use_tqdm=use_tqdm,
        )
        spq, multi_o = zip(*s_p_q_to_multi_tails.items())
        mapped_statements: torch.LongTensor = torch.tensor(spq, dtype=torch.long)
        labels = np.array([np.array(item) for item in multi_o], dtype=object)

        # create mask
        entity_mask = torch.zeros(self.num_entities, dtype=torch.bool)
        entity_mask[self.non_qualifier_only_entities] = True

        return LCWAInstances(
            mapped_statements=mapped_statements,
            entity_to_id=self.entity_to_id,
            relation_to_id=self.relation_to_id,
            labels=labels,
            # entity_mask=entity_mask,
        )

    def create_slcwa_instances(self) -> SLCWAInstances:
        """Create sLCWA instances for this factory's statements."""
        return SLCWAInstances(
            mapped_statements=self.mapped_statements,
            entity_to_id=self.entity_to_id,
            relation_to_id=self.relation_to_id,
            entities=self.non_qualifier_only_entities,
        )

    @property
    def num_entities(self) -> int:
        """The number of unique entities."""
        return len(self.entity_to_id)

    @property
    def num_relations(self) -> int:
        """The number of unique relations."""
        return len(self.relation_to_id)

    @property
    def statement_length(self) -> int:
        """The number of unique relations."""
        return self.max_num_qualifier_pairs * 2 + 3

    @property
    def num_statements(self) -> int:
        """The number of statements."""
        return self.mapped_statements.shape[0]

    @property
    def qualifier_ratio(self) -> float:
        """Return the percentage of statements with qualifiers."""
        return (~(self.mapped_statements[:, 3::2] == self.padding_idx).all(dim=1)).sum().item() / self.num_statements

    def extra_repr(self) -> str:
        """Extra representation."""
        return f"num_entities={self.num_entities:,}, " \
               f"num_relations={self.num_relations:,}, " \
               f"num_statements={self.num_statements:,}, " \
               f"max_num_qualifier_pairs={self.max_num_qualifier_pairs}, " \
               f"qualifier_ratio={self.qualifier_ratio:.2%}"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.extra_repr()})"

    # compatability with pykeen
    @property
    def num_triples(self) -> int:
        return self.num_statements


def _create_multi_label_tails_instance(
    mapped_statements: torch.Tensor,
    use_tqdm: Optional[bool] = None,
) -> Dict[Tuple[int, int], List[int]]:
    """Create for each (h,r,q*) pair the multi tail label."""
    logger.debug('Creating multi label tails instance')

    '''
    The mapped triples matrix has to be a numpy array to ensure correct pair hashing, as explained in
    https://github.com/pykeen/pykeen/commit/1bc71fe4eb2f24190425b0a4d0b9d6c7b9c4653a
    '''
    mapped_statements = mapped_statements.cpu().detach().numpy()

    s_p_q_to_multi_tails_new = _create_multi_label_instances(
        mapped_statements,
        element_1_index=0,
        element_2_index=1,
        label_index=2,
        use_tqdm=use_tqdm,
    )

    logger.debug('Created multi label tails instance')

    return s_p_q_to_multi_tails_new


def _create_multi_label_instances(
    mapped_statements: torch.Tensor,
    element_1_index: int,
    element_2_index: int,
    label_index: int,
    use_tqdm: Optional[bool] = None,
) -> Dict[Tuple[int, ...], List[int]]:
    """Create for each (element_1, element_2) pair the multi-label."""
    instance_to_multi_label = defaultdict(set)

    if use_tqdm is None:
        use_tqdm = True

    it = mapped_statements

    if use_tqdm:
        it = tqdm(mapped_statements, unit='statement', unit_scale=True, desc='Grouping statements')

    for row in it:
        instance = tuple([row[element_1_index], row[element_2_index]] + row[3:].tolist())
        instance_to_multi_label[instance].add(
            row[label_index])

    # Create lists out of sets for proper numpy indexing when loading the labels
    # TODO is there a need to have a canonical sort order here?
    instance_to_multi_label_new = {
        key: list(value)
        for key, value in instance_to_multi_label.items()
    }

    return instance_to_multi_label_new


def get_semi_inductive_factories(
    dataset_name: str,
    max_num_qualifier_pairs: Optional[int] = None,
    create_inverse_triples: bool = True,
) -> Tuple[StatementsFactory, StatementsFactory, StatementsFactory]:
    """Get factories for semi-inductive split."""
    dataset = dataset_resolver.make(query=dataset_name)
    paths = {
        key: dataset.get_path(split="semi_inductive", part=key)
        for key in [
            "train",
            "validation",
            "test",
        ]
    }

    # Step 1: Get relation_to_id based on relations in train
    tmp_stmt_factory = StatementsFactory(
        path=paths['train'],
        dataset_name=dataset_name,
        use_node_features=True,
        is_inductive=False,
        enforce_node_features=True,
        create_inverse_triples=create_inverse_triples,
        max_num_qualifier_pairs=max_num_qualifier_pairs,
    )

    relation_to_id = tmp_stmt_factory.relation_to_id

    # Step 2: Create entity_to_id based on all entities. Entities need to be connected to train-relations
    tmp_stmt_factory = StatementsFactory(
        path=list(paths.values()),
        dataset_name=dataset_name,
        create_inverse_triples=create_inverse_triples,
        relation_to_id=relation_to_id,
        use_node_features=True,
        is_inductive=True,
        enforce_node_features=True,
        max_num_qualifier_pairs=max_num_qualifier_pairs,
    )
    logging.info(f"Full: {tmp_stmt_factory}")

    entity_to_id = tmp_stmt_factory.entity_to_id

    del tmp_stmt_factory

    # Create statement factories
    train_sf = StatementsFactory(
        path=paths["train"],
        dataset_name=dataset_name,
        entity_to_id=entity_to_id,
        relation_to_id=relation_to_id,
        create_inverse_triples=create_inverse_triples,
        use_node_features=True,
        is_inductive=False,
        enforce_node_features=True,
        max_num_qualifier_pairs=max_num_qualifier_pairs,
    )
    logging.info(f"Training: {train_sf}")

    validation_sf = StatementsFactory(
        path=paths["validation"],
        dataset_name=dataset_name,
        is_inductive=True,
        create_inverse_triples=False,
        # inductive entities
        entity_to_id=train_sf.entity_to_id,
        # shared relations
        relation_to_id=train_sf.relation_to_id,
        use_node_features=True,
        enforce_node_features=True,
        max_num_qualifier_pairs=max_num_qualifier_pairs,
    )
    logging.info(f"Validation: {validation_sf}")

    # In fully inductive setting, validation graph consists of training entities
    test_sf = StatementsFactory(
        path=paths["test"],
        dataset_name=dataset_name,
        create_inverse_triples=False,
        is_inductive=True,
        # inductive entities
        entity_to_id=train_sf.entity_to_id,
        # inductive relations
        relation_to_id=train_sf.relation_to_id,
        use_node_features=True,
        enforce_node_features=True,
        max_num_qualifier_pairs=max_num_qualifier_pairs,
    )
    logging.info(f"Testing: {test_sf}")

    return train_sf, validation_sf, test_sf


def get_fully_inductive_factories(
    dataset_name: str,
    dataset_version: str,
    max_num_qualifier_pairs: int,
    create_inverse_triples: bool,
) -> Tuple[StatementsFactory, StatementsFactory, StatementsFactory, StatementsFactory]:
    """Create statement factories for the fully inductive setting."""
    dataset = dataset_resolver.make(query=dataset_name)
    paths = {
        key: dataset.get_path(split="inductive", part=key, version=dataset_version)
        for key in [
            "train",
            "inference",
            "validation",
            "test",
        ]
    }
    train_transductive_sf = StatementsFactory(
        path=paths["train"],
        dataset_name=dataset_name,
        create_inverse_triples=create_inverse_triples,
        use_node_features=True,
        is_inductive=False,
        max_num_qualifier_pairs=max_num_qualifier_pairs,
    )
    train_inductive_sf = StatementsFactory(
        path=paths["inference"],
        dataset_name=dataset_name,
        create_inverse_triples=create_inverse_triples,
        relation_to_id=train_transductive_sf.relation_to_id,
        is_inductive=True,
        use_node_features=True,
        max_num_qualifier_pairs=max_num_qualifier_pairs,
    )
    # In fully inductive setting, validation graph consists of training entities
    validation_sf = StatementsFactory(
        path=paths["validation"],
        dataset_name=dataset_name,
        entity_to_id=train_inductive_sf.entity_to_id,
        relation_to_id=train_inductive_sf.relation_to_id,
        use_node_features=True,
        is_inductive=True,
        max_num_qualifier_pairs=max_num_qualifier_pairs,
    )
    # Test graph contains only unseen entities, therefore, create
    test_sf = StatementsFactory(
        path=paths["test"],
        dataset_name=dataset_name,
        entity_to_id=train_inductive_sf.entity_to_id,
        relation_to_id=train_inductive_sf.relation_to_id,
        use_node_features=True,
        is_inductive=True,
        max_num_qualifier_pairs=max_num_qualifier_pairs,
    )
    return train_transductive_sf, train_inductive_sf, test_sf, validation_sf


# buffer loading (for HPO)
@functools.lru_cache(maxsize=1)
def get_factories(
    dataset_name: str,
    dataset_version: Optional[str],
    inductive_setting: str,
    max_num_qualifier_pairs: int,
    create_inverse_triples: bool,
) -> Tuple[StatementsFactory, StatementsFactory, StatementsFactory, StatementsFactory]:
    """Load triples factories."""
    if inductive_setting == "semi":
        train_sf, validation_sf, test_sf = get_semi_inductive_factories(
            dataset_name=dataset_name,
            max_num_qualifier_pairs=max_num_qualifier_pairs,
            create_inverse_triples=create_inverse_triples,
        )
        inference_sf = train_sf

    elif inductive_setting == "full":
        train_sf, inference_sf, test_sf, validation_sf = get_fully_inductive_factories(
            dataset_name=dataset_name,
            dataset_version=dataset_version,
            max_num_qualifier_pairs=max_num_qualifier_pairs,
            create_inverse_triples=create_inverse_triples,
        )

    else:
        raise ValueError(f"Unknown inductive setting: {inductive_setting}")

    return inference_sf, test_sf, train_sf, validation_sf
