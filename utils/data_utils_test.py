import unittest
from utils import data_utils


class TestLoadCord(unittest.TestCase):

    def test_load_preprocessed_cord_from_dir(self):
        cord_dir = "../data"
        out = data_utils.load_preprocessed_cord_from_dir(cord_dir)
        self.assertEqual(out, True)


if __name__ == '__main__':
    unittest.main()
