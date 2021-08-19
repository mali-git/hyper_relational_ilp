"""Resolver for models."""
import class_resolver

from .base import QualifierModel
from .blp import BLP
from .qblp import QBLP
from .stare import StarE

model_resolver = class_resolver.Resolver(
    classes={
        BLP,
        QBLP,
        StarE,
    },
    base=QualifierModel,  # type: ignore
)
