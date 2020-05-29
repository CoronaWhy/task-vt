# -*- coding: utf-8 -*-

"""Example tests for CoronaWhy."""

import unittest

from coronawhy_vt import get_version


class TestVersion(unittest.TestCase):
    """Tests for getting the version."""

    def test_version_type(self):
        """Test the version has the right type."""
        self.assertIsInstance(get_version(), str)
