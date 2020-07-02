# -*- coding: utf-8 -*-

"""Constants for tests."""

from pathlib import Path

HERE = Path(__file__).parent.absolute()
RESOURCES = HERE / 'resources'
TEST_FILES = RESOURCES / 'test_files'
DOCUMENT_PARSES = TEST_FILES / 'document_parses'
PDF_JSON_DOCUMENT_PARSES = DOCUMENT_PARSES / 'pdf_json'

STUDY_WORDS = TEST_FILES / 'sample_study_words.txt'
SAMPLE_PAPER = PDF_JSON_DOCUMENT_PARSES / 'sample_paper.json'
SAMPLE_SENTENCE = TEST_FILES / 'sample_sentence.txt'
