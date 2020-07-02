# -*- coding: utf-8 -*-

"""Tests for coronawhy_vt utilities."""

import unittest
from collections import namedtuple
from math import inf

# Project specific packages
from coronawhy_vt.utils.local_processing import (
    aggregate_text_mentions,
    make_ngram_map_and_list,
    pull_mentions,
    run_replace,
    same_sentence_check,
    title_blocks,
)
from tests.constants import PDF_JSON_DOCUMENT_PARSES, SAMPLE_SENTENCE, STUDY_WORDS
from tests.utils.utils import get_text, list_compare


class TestUtils(unittest.TestCase):
    """Main class for testing utilities."""

    # TODO Update Travis CI file to include this once tests are stable.
    # TODO add logging debug args

    @classmethod
    def setUp(cls):
        """Set up test class."""
        # TODO initialize class when we refactor the functions into classes
        cls.expected_virus_words = [
            'COVID-19', 'COVID 19', '2019-nCoV', '2019 nCoV', 'SARS-CoV-2', 'SARS-CoV 2', 'SARS CoV-2',
            'SARS CoV 2', 'SARS-CoV2', 'SARS-CoV19', 'SARS-CoV2019', 'SARSCoV2', 'SARSCoV 2',
            'SARSCoV19', 'SARSCoV2019', 'SARS CoV2', 'SARS Cov19', 'SARS Cov 19', 'SARS CoV 2019',
            'SARS CoV2019', '2019 Coronavirus', 'Coronavirus Disease 2019', 'coronavirus 2019',
            'novel coronaviruses', 'covid19',
        ]

    def test_make_ngram_map_and_list_study_words(self):
        """Test make_ngram_map_and_list on study words."""
        study_names, study_ngram_map = make_ngram_map_and_list(STUDY_WORDS, min_word_len=0, max_num_tokens=inf)

        expected_study_names = [
            'vitro', 'in-vitro', 'cell_culture', 'fbs', 'fetal_bovine_serum',
            'co2', 'air-liquid_interface', 'mouse', 'affected_health_professionals',
            'iqr', 'chest_computed_tomographic_scans', 'non-invasive_ventilation',
            'patient-to-patient',
        ]

        expected_study_ngram_map = {
            'cell culture': 'cell_culture',
            'fetal bovine serum': 'fetal_bovine_serum',
            'air-liquid interface': 'air-liquid_interface',
            'affected health professionals': 'affected_health_professionals',
            'chest computed tomographic scans': 'chest_computed_tomographic_scans',
            'non-invasive ventilation': 'non-invasive_ventilation',
        }

        self.assertTrue(list_compare(study_names, expected_study_names))
        self.assertEqual(study_ngram_map, expected_study_ngram_map)

    def test_make_ngram_map_and_list_treatment_words(self):
        """Test make_ngram_map_and_list on treatment words."""
        # TODO
        # treatment_names_path = Path('test_files/sample_treatment_words.txt')

    def test_make_ngram_map_and_list_virus_words(self):
        """Test make_ngram_map_and_list on virus words."""
        # TODO
        # virus_names_path = Path('test_files/sample_virus_words.txt')

    def test_pull_mentions_lemmatized(self):
        """Test pull_mentions_lemmatized."""
        # TODO
        path = [PDF_JSON_DOCUMENT_PARSES]
        lemmatized_paper = pull_mentions(path, title_blocks, 'title', self.expected_virus_words, lemmatized=True)
        self.assertTrue(lemmatized_paper is not None)

    def test_pull_mentions_direct(self):
        """Test pull_mentions_direct."""
        # TODO
        path = [PDF_JSON_DOCUMENT_PARSES]
        lemmatized_paper = pull_mentions(path, title_blocks, 'title', self.expected_virus_words, lemmatized=False)
        self.assertTrue(lemmatized_paper is not None)

    def test_run_replace(self):
        """Test run_replace."""
        # TODO super basic test because anticipating this function to be changed
        s = 'the quick brown fox jumped over the lazy dog.'
        replace_dict = {'the quick': 'the_quick', 'fox jumped': 'fox_jumped'}
        expected_replaced_s = 'the_quick brown fox_jumped over the lazy dog.'
        self.assertEqual(run_replace(s, replace_dict), expected_replaced_s)

    def test_aggregate_text_mentions(self):
        """Check if list of namedtuples is successfully aggregated into a dict."""
        coords = namedtuple('coords', ['x', 'y', 'z'])
        a = coords(1, 2, 3)
        b = coords(4, 5, 6)
        c = coords(7, 8, 9)
        coord_list = [a, b, c]
        summed_coords = {'x': 12, 'y': 15, 'z': 18}
        self.assertEqual(summed_coords, aggregate_text_mentions(coord_list))

    def test_extract_to_csv(self):
        """Test extract_to_csv."""
        # TODO

    def test_same_sentence_check(self):
        """Test sentence enders are properly detected."""
        block = get_text(SAMPLE_SENTENCE)
        self.assertTrue(same_sentence_check(block, 0, 98))
        self.assertFalse(same_sentence_check(block, 166, 168))

    def test_convert_date_to_year(self):
        """Test date is properly converted."""
        # TODO there might be a more robust way to write this with try except statements.
