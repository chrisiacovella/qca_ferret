"""
Unit and regression test for the qca_ferret package.
"""

# Import package, test suite, and other packages as needed
import sys

import pytest

import qca_ferret


def test_qca_ferret_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "qca_ferret" in sys.modules
