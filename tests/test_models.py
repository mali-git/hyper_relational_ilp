"""Tests for models."""
import random
from typing import Any, List, MutableMapping

import unittest_templates
from pykeen.losses import BCEWithLogitsLoss

from ilp.data.statement_factory import StatementsFactory
from ilp.models import BLP, QBLP, QualifierModel, StarE
from ilp.models.base import BaseQualifierModel


def _generate_statements(
    num_entities: int,
    num_relations: int,
    max_num_qualifier_pairs: int,
    num_statements: int,
) -> List[List[str]]:
    """Generate random statements."""
    entities = [f"e_{i}" for i in range(num_entities)]
    relations = [f"r_{i}" for i in range(num_relations)]
    return [
        [
            random.choice(entities),
            random.choice(relations),
            random.choice(entities),
        ] + sum(
            (
                [
                    random.choice(relations),
                    random.choice(entities),
                ]
                for _ in range(random.randrange(max_num_qualifier_pairs + 1))
            ),
            []
        )
        for _ in range(num_statements)
    ]


class QualifierModelTests(unittest_templates.GenericTestCase[QualifierModel]):
    """Tests for qualifier models."""

    num_entities: int = 33
    num_relations: int = 5
    num_statements: int = 101
    max_num_qualifier_pairs: int = 2
    input_dim = 8
    batch_size: int = 2
    create_inverse_triples: bool = False

    factory: StatementsFactory

    def _pre_instantiation_hook(self, kwargs: MutableMapping[str, Any]) -> MutableMapping[str, Any]:  # noqa: D102
        kwargs["loss"] = BCEWithLogitsLoss
        self.factory = kwargs["statement_factory"] = StatementsFactory(
            statements=_generate_statements(
                num_entities=self.num_entities,
                num_relations=self.num_relations,
                num_statements=self.num_statements,
                max_num_qualifier_pairs=self.max_num_qualifier_pairs,
            ),
            max_num_qualifier_pairs=self.max_num_qualifier_pairs,
            use_node_features=False,
            feature_dim=self.input_dim,
            create_inverse_triples=self.create_inverse_triples,
        )
        self.data_geometric = self.factory.create_data_object()
        return kwargs

    def test_score_h(self):
        """Test score_h."""
        batch = self.factory.mapped_statements[:self.batch_size, :]
        rt_batch = batch[:, 1:3]
        qualifiers = batch[:, 3:]
        scores = self.instance.score_h(
            rt_batch=rt_batch,
            qualifiers=qualifiers,
            data_geometric=self.data_geometric,
        )

    def test_score_t(self):
        """Test score_t."""
        batch = self.factory.mapped_statements[:self.batch_size, :]
        hr_batch = batch[:, :2]
        qualifiers = batch[:, 3:]
        scores = self.instance.score_t(
            hr_batch=hr_batch,
            qualifiers=qualifiers,
            data_geometric=self.data_geometric,
        )


class BLPTests(QualifierModelTests):
    """Tests for BLP."""

    cls = BLP


class QBLPTests(QualifierModelTests):
    """Tests for QBLP."""

    cls = QBLP
    kwargs = dict(
        embedding_dim=QualifierModelTests.input_dim,
    )
    create_inverse_triples = True


class StarETests(QualifierModelTests):
    """Tests for StarE."""

    cls = StarE
    kwargs = dict(
        embedding_dim=QualifierModelTests.input_dim,
    )
    create_inverse_triples = True


class QualifierModelTestsTest(unittest_templates.MetaTestCase[QualifierModel]):
    """Test for tests for all qualifier models."""

    base_cls = QualifierModel
    base_test = QualifierModelTests
    skip_cls = {
        BaseQualifierModel,
    }
