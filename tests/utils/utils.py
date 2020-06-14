# -*- coding: utf-8 -*-

"""Utilities for tests."""

import collections
from pathlib import Path


def list_compare(a: list, b: list) -> bool:
    """Check if two lists contain the same elements."""
    return collections.Counter(a) == collections.Counter(b)


def get_text(p: Path) -> str:
    """Return the contents from a single text file with newlines stripped."""
    with open(p, 'r') as file:
        data = file.read().replace('\n', '')
    return data
