import unittest
import logging
import argparse
import json
from pathlib import Path

# TODO proper packaging
import sys
sys.path.append(str(Path('../')))
from data_utils.local_processing import *

# NOTE right now these tests check how I believe the function should behave, need to check in with either 
# Dan Sosa or Ben Jones to confirm

# TODO Update Travis CI file to include this once tests are stable.
# TODO add logging debug args

class TestUtils(unittest.TestCase):

    @classmethod
    def setUp(self):
        # TODO initialize class when we refactor the functions into classes
        log = logging.getLogger('TestUtils')
        self.expected_virus_words = ['COVID-19', 'COVID 19', '2019-nCoV', '2019 nCoV', 'SARS-CoV-2', 'SARS-CoV 2', 'SARS CoV-2'\
            , 'SARS CoV 2', 'SARS-CoV2', 'SARS-CoV19', 'SARS-CoV2019', 'SARSCoV2', 'SARSCoV 2', 'SARSCoV19', 'SARSCoV2019'\
            , 'SARS CoV2', 'SARS Cov19', 'SARS Cov 19', 'SARS CoV 2019', 'SARS CoV2019', '2019 Coronavirus'\
            , 'Coronavirus Disease 2019', 'coronavirus 2019', 'novel coronaviruses', 'covid19']

    def test_MakeNGramMapAndList_study_words(self):
        study_words_path = Path('test_files/sample_study_words.txt')
        study_names, study_ngram_map = MakeNGramMapAndList(study_words_path)

        expected_study_names = ['Vitro', 'In-vitro', 'Cell culture', 'FBS', 'fetal bovine serum', 'CO2','Air-liquid interface'\
            , 'Mouse', 'Affected health professionals', 'IQR', 'chest computed tomographic scans', 'non-invasive ventilation' \
            , 'patient-to-patient']
        
        expected_study_ngram_map = {'Vitro': 'Vitro', 'In-vitro': 'In-vitro', 'Cell culture': 'Cell_culture', 'FBS': 'FBS'\
            ,'fetal bovine serum': 'fetal_bovine_serum', 'CO2': 'CO2', 'Air-liquid interface': 'Air-liquid_interface'\
            , 'Mouse': 'Mouse', 'Affected health professionals': 'Affected_health_professionals', 'IQR': 'IQR'\
            , 'chest computed tomographic scans': 'chest_computed_tomographic_scans'\
            , 'non-invasive ventilation': 'non-invasive_ventilation', 'patient-to-patient': 'patient-to-patient'}

        self.assertEqual(study_names, expected_study_names)
        self.assertEqual(study_ngram_map, expected_study_ngram_map)

    def test_MakeNGramMapAndList_treatment_words(self):
        # TODO
        # treatment_names_path = Path('test_files/sample_treatment_words.txt')
        pass

    def test_MakeNGramMapAndList_virus_words(self):
        # TODO
        # virus_names_path = Path('test_files/sample_virus_words.txt')
        pass

    def test_TitleBlocks(self):
        # TODO do we really need to test these functions...
        pass
    def test_AbstractBlocks(self):
        # TODO
        pass
    def test_BodyBlocks(self):
        # TODO
        pass

    def test_PullMentionsLemmatized(self):
        # TODO
        path = [Path('test_files/document_parses/pdf_json')]
        lemmatized_paper = PullMentionsLemmatized(path, TitleBlocks, 'title', self.expected_virus_words)


    def test_PullMentionsDirect(self):
        # TODO
        path = [Path('test_files/document_parses/pdf_json')]
        lemmatized_paper = PullMentionsDirect(path, TitleBlocks, 'title', self.expected_virus_words)

    def test_RunReplace(self):
        # TODO super basic test because anticipating this function to be changed
        s = 'the quick brown fox jumped over the lazy dog.'
        replace_dict = {'the quick': 'the_quick', 'fox jumped': 'fox_jumped'}
        expected_replaced_s = 'the_quick brown fox_jumped over the lazy dog.'
        self.assertEqual(RunReplace(s, replace_dict), expected_replaced_s)

    def test_ExtractToCSV(self):
        # TODO 
        pass

    def test_SameSentenceCheck(self):
        block = 'We demonstrated_that PfSWIB was involved in the process of clonal variation in var gene expression, ' + \
            'and crucial for the survival and development of Plasmodium parasite. These findings could provide better ' + \
            'understanding of the mechanism and function of PfSWIB contributing to the pathogenesis in malaria parasites.'
        self.assertTrue(SameSentenceCheck(block, 0, 98))
        self.assertFalse(SameSentenceCheck(block, 166, 168))

    def test_Make2DPlot(self):
        # TODO not sure how to test this
        pass

    def test_ConvertDateToYear(self):
        # TODO there might be a more robust way to write this with try except statements.
        pass

if __name__ == '__main__':
    test_utils_log = 'test_utils_log.txt'
    with open(test_utils_log, 'w') as f:
        runner = unittest.TextTestRunner(f)
        unittest.main(testRunner=runner)
