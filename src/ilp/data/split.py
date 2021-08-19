"""Dataset split methods."""
import logging
import random
from typing import Collection, List, Sequence, Set, Tuple

logger = logging.getLogger(__name__)


def filter_qualifiers(
    qualifiers: Tuple[str, ...],
    allowed: Collection[str],
    entities: bool = True,
) -> Tuple[str, ...]:
    """
    Filter qualifiers by a list of allowed entities/relations.

    :param qualifiers:
        The qualifier sequence, alternatingly relation and entity.
    :param allowed:
        The allowed.
    :param entities:
        Whether to filter for entities or relations.

    :return:
        A tuple of qualifiers, where the entities are guaranteed to be in the allowed set.
    """
    i = 1 if entities else 0
    # convert to (relation, entity) pairs
    pairs = zip(qualifiers[::2], qualifiers[1::2])
    # filter
    pairs = (pair for pair in pairs if pair[i] in allowed)
    # flatten
    return sum(pairs, tuple())


def get_entities(
    statements: Collection[Sequence[str]],
    include_qualifier: bool = False,
) -> Set[str]:
    """
    Get entities from a collection of statements.

    :param statements:
        The statements, in format: (h, r, t, *q), where q are qualifiers.
    :param include_qualifier:
        Whether to include qualifier-only entities.

    :return:
        All entities occurring as head or tail.
    """
    entities = set()
    end = None if include_qualifier else 3
    for s in statements:
        entities.update(s[:end:2])
    logger.info(f"Extracted {len(entities):,} entities from {len(statements):,} statements.")
    return entities


def get_relations(statements: Collection[Sequence[str]]) -> Set[str]:
    """
    Get the relations from a list of statements.

    .. note ::
        This method also extracts qualifier-only relations.

    :param statements:
        The statements, in format: (h, r, t, *q), where q are qualifiers.

    :return:
        All relations occurring at any position.
    """
    relations = set()
    for s in statements:
        relations.update(s[1::2])
    logger.info(f"Extracted {len(relations)} relations from {len(statements)} statements.")
    return relations


def get_entity_split(
    entities: Collection[str],
    ratios: Tuple[float, float] = (0.8, 0.1),
    random_seed: int = 2,
) -> Tuple[Set[str], Set[str], Set[str]]:
    """
    Randomly split entities in three sets according to ratio.

    :param entities:
        The collection of entities.
    :param ratios:
        The train and validation ratio (rest is test).
    :param random_seed:
        The random seed for reproducible splits.

    :return:
        A 3-partition of the set of entities: (train, valid, test).
    """
    # reproducible split
    entities = sorted(set(entities))
    random.seed(random_seed)
    random.shuffle(entities)

    # get number
    i, j = [int(len(entities) * ratio) for ratio in ratios]

    entity_split = set(entities[:i]), set(entities[i:i + j]), set(entities[i + j:])
    logger.info(f"Split {len(entities):,} entities to {[len(s) for s in entity_split]}")
    return entity_split


def split_statements_by_entity_split_semi_inductive(
    statements: Collection[Sequence[str]],
    entity_split: Sequence[Set[str]],
) -> Sequence[Collection[Tuple[str, ...]]]:
    """
    Split statements based on entity split.

    The validation/test set contain only statements with exactly one training entity and one validation/test entity.

    :param statements:
        All statements.
    :param entity_split:
        The entity split. The first component is the training set.

    :return:
        The statement split.
    """
    entity_split = list(entity_split)
    # convert to tuples
    statements = [tuple(statement) for statement in statements]
    # split off training entities
    train_entities = entity_split[0]
    n_train_original = len(train_entities)
    # get training statements
    training_statements = [
        statement
        for statement in statements
        if {statement[0], statement[2]}.issubset(train_entities)
    ]
    # re-adjust training entities to those occurring in training statements
    train_entities = set(sum((list(statement[::2]) for statement in training_statements), []))
    entity_split[0] = train_entities
    logger.info(f"Adjusted original {n_train_original} training entities to {len(train_entities)} which actually occur.")

    result = []
    for entities in entity_split:
        statements_split = statements
        # get (train_entities <-> entities) statements
        statements_split = [
            statement
            for statement in statements_split
            if any((statement[i] in train_entities and statement[j] in entities) for (i, j) in ((0, 2), (2, 0)))
        ]
        # filter out qualifiers outside of train_entities.union(entities)
        allowed_entities = train_entities.union(entities)
        statements_split = [
            statement[:3] + filter_qualifiers(qualifiers=statement[3:], allowed=allowed_entities)
            for statement in statements_split
        ]
        result.append(statements_split)
        # re-adjust entities
        n_entities = len(entities)
        entities = set(sum((statement[::2] for statement in statements_split), tuple()))
        logger.info(f"Adjusted {n_entities} to {len(entities)} actually occurring.")
    return result


def remove_unseen_relations(
    statements_split: Sequence[Collection[Tuple[str, ...]]],
    seen_relations: Collection[str],
) -> Sequence[Collection[Tuple[str, ...]]]:
    """
    Remove unseen relations from statements.

    :param statements_split:
        The collection of statements.
    :param seen_relations:
        The seen relations.

    :return:
        The filtered statements, where all triples with unseen relations are dropped, and all qualifiers are filtered
        to only contain those with seen relations.
    """
    result = []
    for statements in statements_split:
        # drop statements with (h, unseen_r, t, *)
        statements = [
            statement
            for statement in statements
            if statement[1] in seen_relations
        ]
        # filter qualifiers
        statements = [
            statement[:3] + filter_qualifiers(qualifiers=statement[3:], allowed=seen_relations, entities=False)
            for statement in statements
        ]
        result.append(statements)
    return result


def generate_semi_inductive_split(
    statements: Collection[Sequence[str]],
    ratios: Tuple[float, float] = (0.8, 0.1),
    random_seed: int = 2,
) -> Tuple[Sequence[Collection[Tuple[str, ...]]], Set[str]]:
    """
    Create a semi-inductive statement split.

    :param statements:
        All statements.
    :param ratios:
        The train and validation ratio (rest is test).
    :param random_seed:
        The random seed for reproducible splits.

    :return:
        A 4-tuple of training, validation, test statements, and training entities
    """
    # convert to tuples
    statements = [tuple(statement) for statement in statements]

    # extract head & tail entities
    entities = get_entities(statements=statements, include_qualifier=False)

    # split entities
    entity_split = list(get_entity_split(
        entities=entities,
        ratios=ratios,
        random_seed=random_seed,
    ))

    # get training statements
    training_statements = [
        s
        for s in statements
        if {s[0], s[2]}.issubset(entity_split[0])
    ]

    # re-adjust training entities
    training_entities = get_entities(training_statements, include_qualifier=False)
    logger.info(f"Adjusted training entities to {len(training_entities):,} actually occurring.")
    entity_split[0] = training_entities

    # filter qualifiers
    training_statements = [
        s[:3] + filter_qualifiers(qualifiers=s[3:], allowed=training_entities, entities=True)
        for s in training_statements
    ]
    assert all(set(s[::2]).issubset(training_entities) for s in training_statements)

    # get training relations
    relations = get_relations(statements=training_statements)
    logger.info(f"Extracted {len(relations):,} relations occurring in training statements.")

    result = [training_statements]

    for unseen_entities in entity_split[1:]:
        assert training_entities.isdisjoint(unseen_entities)
        eval_statements = []
        for (h, r, t, *qs) in statements:
            # unknown relation
            if r not in relations:
                continue
            if len({h, t}.intersection(unseen_entities)) != 1:
                continue
            if len({h, t}.intersection(training_entities)) != 1:
                continue
            # filter qualifier entities
            qs = filter_qualifiers(qualifiers=qs, allowed=training_entities, entities=True)
            # filter qualifier relations
            qs = filter_qualifiers(qualifiers=qs, allowed=relations, entities=False)
            # append statement
            statement = (h, r, t) + tuple(qs)
            eval_statements.append(statement)

        logger.info(f"Found {len(eval_statements):,} statements between {len(unseen_entities):,} unseen entities and {len(training_entities):,} training entities.")
        result.append(eval_statements)

    return result, training_entities


def get_semi_inductiveness_violations(
    train: Collection[Tuple[str, ...]],
    *statements: Collection[Tuple[str, ...]],
    e_train: Collection[str] = None,
) -> List[Tuple[int, Tuple[str, ...], str]]:
    """Get a list of violations of semi-inductive split property."""
    if e_train is None:
        logger.warning("No training entity set was provided. Try to infer one.")
        e_train = get_entities(statements=train, include_qualifier=True)
    errors = []
    e_train = set(e_train)
    for i, this_statements in enumerate((train, *statements)):
        exp_num_train_entities = (2 if i == 0 else 1)
        # get allowed qualifier entities
        allowed = e_train.union(get_entities(this_statements, include_qualifier=True))
        for statement in this_statements:
            # check length
            if not len(statement) >= 3:
                errors.append((i, statement, "length < 3"))
                continue
            if not len(statement) % 2 == 1:
                errors.append((i, statement, "length not odd"))
                continue
            num_train_entities = sum((1 if x in e_train else 0) for x in statement[:3:2])
            if not num_train_entities == exp_num_train_entities:
                errors.append((i, statement, f"{num_train_entities} train entities ({exp_num_train_entities} expected)"))
                continue
            # check qualifier entities
            invalid_qualifier_entities = set(statement[::2]).difference(allowed)
            if len(invalid_qualifier_entities) > 0:
                errors.append((i, statement, f"invalid entities in qualifiers: {invalid_qualifier_entities}"))
                continue
    return errors
