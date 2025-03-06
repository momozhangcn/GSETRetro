"""rxnmapper initialization."""
__name__ = "rxnmapper"
__version__ = "0.3.0"  # managed by bump2version

import os
import sys
if os.path.dirname(__file__) not in sys.path:
    sys.path.append(os.path.dirname(__file__))
from batched_mapper import BatchedMapper
from core import RXNMapper

__all__ = [
    "BatchedMapper",
    "RXNMapper",
]
