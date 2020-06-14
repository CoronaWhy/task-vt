import unittest

from spacify import char_idx_to_token_idx

class TestSpacify(unittest.TestCase):
    def test_char_idx_to_token_idx(self):
        sent = "The quick brown fox jumps over the lazy dog."
        char_start = 4
        char_end = 9
        result = char_idx_to_token_idx(sent, char_start, char_end)
        answer = (1, 2, 'quick')
        self.assertEqual(answer, result)

if __name__ == '__main__':
    unittest.main()