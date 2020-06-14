# -*- coding: utf-8 -*-

"""Temporary storage for utils used in the processing of local data structures."""
# TODO function names are in camelcase suggest refactoring to conform to PEP-8
import json
import os
from typing import Callable

import dateutil.parser as parser
import numpy as np
import pandas as pd
import pylab
import spacy

# TODO All of these functions open a file and do not close it, putting them in the same class would solve some issues
# TODO lowercase?


def make_ngram_map_and_list(filename: str, min_word_len: int = 6, min_num_tokens: int = 0,
                            max_num_tokens: int = 4) -> (list, dict):
    """Given the path of a file of tokens, replace every " " with "_".

    :param filename: The path of the file
    :param min_word_length: Minimum length of vocab entry to be accepted
    :param min_num_tokens: Minimum number of tokens in the vocab entry
    :param max_num_tokens: Maximum number of tokens in the vocab entry

    :return (names, ngram_map): (Original tokens with spaces instead of underscores
        {names: names_with_space_repaced_by_underscores})
    """
    ngram_map = {}
    names = []
    all_names = []

    file = open(filename).readlines()
    for line in file:
        # TODO this is meant to remove the last newline character but can remove the last character of the last line if
        # file does not end with newline recommend rstrip()
        all_names.append(line.rstrip())

    for name in all_names:
        if len(name) >= min_word_len:
            if " " in name:
                if min_num_tokens <= len(name.split(" ")) <= max_num_tokens:
                    newname = name.replace(" ", "_").lower()
                    ngram_map[name.lower()] = newname
                    names.append(newname)
            else:
                names.append(name.lower())
    names = np.unique(names)
    return names, ngram_map


# TODO Pull the next 6 functions into a class?
# These functions determine what blocks are pulled from the paper for matching
def title_blocks(paper: dict) -> list:
    """Retrieve title block of a paper in json form.

    :param paper (json): The paper to be parsed

    :return: Lines of text in the paper title
    """
    return [{'text': paper['metadata']['title']}]


def abstract_blocks(paper: dict) -> list:
    """Retrieve abstract block of a paper in json form.

    :param paper: The paper to be parsed

    :return: Lines of text in the paper abstract
    """
    return paper['abstract']


def body_blocks(paper: dict) -> list:
    """Retrieve body block of a paper in json form.

    :param paper: The paper to be parsed

    :return: Lines of text in the paper body
    """
    return paper['body_text']


# TODO the reference nlp is an external dependence, either add into function or make it a class
nlp = spacy.load('en_core_web_lg')


def pull_mentions_lemmatized(
        paths: list,
        block_selector: Callable[[dict], list],
        sec_name: str,
        words: list,
        replace_dict: dict = None) -> dict:
    """Aggregate the positional features of the lemmatized text corpus.

    :param paths: pathlib.Path objects corresponding to directories containing
        corpus to aggregate features from.
    :param block_selector: The function to retrieve the relevant block from text
    :param sec_name: Corresponds with block_selector to select the relevant
        block of text. Options are 'title', 'abstract', 'body'
        TODO seems like this should be pulled into the function as it depends
        entirely on the block_selector argument
    :param words: List of lemmatized words
    :param replace_dict: Takes a dict of {token: lemmatized underscore token}
        and replaces occurences of token in text with its value

    :return: {'identifier feature': list of occurrences}
    """
    positions = []
    found_words = []
    section = []
    block_id = []
    block_text = []
    paper_id = []

    tokenized_words = []
    for w in words:
        tokenized_words.append(nlp(w.lower())[0].lemma_)
    for path in paths:
        files = os.listdir(path)
        for p in files:

            readfile = open(path / p, 'r')
            paper = json.load(readfile)
            blocks = block_selector(paper)

            for b in range(0, len(blocks)):
                text = blocks[b]['text'].lower()
                if(replace_dict is not None):
                    text = run_replace(text, replace_dict)
                text = nlp(text)
                for t in text:
                    for w in tokenized_words:
                        if(w == t.lemma_):
                            section.append(sec_name)
                            found_words.append(w)
                            positions.append(t.idx)
                            block_text.append(blocks[b]['text'])
                            block_id.append(b)
                            paper_id.append(p[:-5])
    # TODO discuss pros/cons of this format vs a list of tuples
    return {'sha': paper_id,
            'blockid': block_id,
            'word': found_words,
            'sec': section,
            'pos': positions,
            'block': block_text}


def pull_mentions_direct(
        paths: list,
        block_selector: Callable[[dict], list],
        sec_name: str,
        words: list,
        replace_dict: dict = None) -> dict:
    """Aggregate the positional features of the text corpus.

    :param paths: List of pathlib.Path objects containing paths to the directories containing the corpus of text to
        aggregate features from
    :param block_selector: The function to retrieve the relevant block from text
    :param sec_name: Corresponds with block_selector to select the relevant block of text. Options are 'title',
        'abstract', 'body' TODO seems like this should be pulled into the function as it depends entirely on the
        block_selector argument
    :param words: List of lemmatized words
    :param replace_dict: Takes a dict of {token: lemmatized underscore token} and replaces occurences of token in text
        with its value

    :return: {'identifier feature': list of occurrences}
    """
    # TODO reduce nesting?
    positions = []
    found_words = []
    section = []
    block_id = []
    block_text = []
    paper_id = []
    words = [word.lower() for word in words]
    for path in paths:
        files = os.listdir(path)
        for p in files:

            readfile = open(path / p, 'r')
            paper = json.load(readfile)
            blocks = block_selector(paper)

            for b in range(0, len(blocks)):
                text = blocks[b]['text'].lower()
                if(replace_dict is not None):
                    text = run_replace(text, replace_dict)
                for w in words:
                    if(w in text):
                        pos = text.find(w)

                        # check we're not in the middle of another word
                        if(text[pos - 1] == " " and ((pos + len(w)) >= len(text) or not text[pos + len(w)].isalpha())):
                            section.append(sec_name)
                            found_words.append(w)
                            positions.append(text.find(w))
                            block_text.append(blocks[b]['text'])
                            block_id.append(b)
                            paper_id.append(p[:-5])
    return {'sha': paper_id,
            'blockid':
            block_id,
            'word': found_words,
            'sec': section,
            'pos': positions,
            'block': block_text}

# Recommend changing expected block argument with the entire of the processed
# paper as it seems the only functionality for this is updating the text of
# the section of paper to the underscore tokens


def run_replace(block: str, replace_dict: dict) -> str:
    """Replace routine for dealing with n-grams.

    :param block: Contains the text block replacement is done on
    :param replace_dict: {token: replacement_token}

    :return: Occurrences of token replaced with replacement_token
    """
    for k in replace_dict.keys():
        if k in block:
            block = block.replace(k, replace_dict[k])
    return block


# TODO These paths are kaggle paths, replace first with github paths then DataVerse paths once setup
paths = ["/kaggle/input/CORD-19-research-challenge/noncomm_use_subset/noncomm_use_subset/pdf_json/",
         "/kaggle/input/CORD-19-research-challenge/comm_use_subset/comm_use_subset/pdf_json/",
         "/kaggle/input/CORD-19-research-challenge/biorxiv_medrxiv/biorxiv_medrxiv/pdf_json/",
         "/kaggle/input/CORD-19-research-challenge/custom_license/custom_license/pdf_json/"]
# TODO Right now this function references an externally declared variable, suggest either moving
# into a class or make paths an argument
# Also function name might need to change, a little ambiguous


def extract_to_csv(
        words: list,
        file_name: str,
        lemmatized: bool = True,
        run_title: bool = True,
        run_abstract: bool = True,
        run_body: bool = False,
        replace_dict: dict = None) -> None:
    """Store aggregate features of vocab words in the main CORD-19 dataset.

    :param words: List of strings representing words of interest
    :param file_name: Path of the location to store final csv
    :param lemmatized: If true, perform lemmatized feature extraction
    :param run_title: If true, features extracted from the title will be included in the aggregate
    :param run_abstract: If true, features extracted from the abstract will be included in the aggregate
    :param run_body: If true, features extracted from the body will be included in the aggregate
    :param replace_dict (dict): {token: replacement_token}
    """
    if lemmatized:
        pull_mentions = pull_mentions_lemmatized
    else:
        pull_mentions = pull_mentions_direct

    data_dicts = []
    if run_title:
        data_dicts.append(pull_mentions(paths, title_blocks, "title", words, replace_dict))
    if run_abstract:
        data_dicts.append(pull_mentions(paths, abstract_blocks, "abstract", words, replace_dict))
    if run_body:
        data_dicts.append(pull_mentions(paths, body_blocks, "body", words, replace_dict))

    summed_dictionary = data_dicts[0]
    for k in data_dicts[0].keys():
        for d in data_dicts:
            summed_dictionary[k] = summed_dictionary[k] + d[k]

    dat = pd.DataFrame(summed_dictionary)
    dat.to_csv(file_name)


# TODO change index to be of type int rather than string
def same_sentence_check(block: str, pos1: str, pos2: str) -> bool:
    """Check if two words are in the same sentence by looking for sentence delimiters between their starting positions.

    :param block: Block of string text the two words are found in
    :param pos1: The index of the beginning of word1
    :param pos2: The index of the beginning of word2

    :return: true if they word1 and word2 are not separated by
        one of the follwing sentence delimiters ., ;, ?, !, false otherwise
    """
    if pos1 < pos2:
        interstring = block[int(pos1):int(pos2)]
    else:
        interstring = block[int(pos2):int(pos1)]
    sentence_enders = [".", ";", "?", "!"]
    for s in sentence_enders:
        if s in interstring:
            return 0
    return 1

# This function makes the 2D quilt plot for showing co-occurences at block
#   or sentence level of various classes of search terms
#


def make_2d_plot(dat_joined: pd.DataFrame, factor_1: str, factor_2: str, single_sentence_plots: bool = False) -> None:
    """Create 2D quilt plot from dataframe row columns ('word_' + factor_1) and ('word_' + factor_2).

    :param dat_joined: Dataframe of the format returned by pull_mentions_lemmatized and pull_mentions_direct
    :param factor_1: x-axis of the graph, possible entries 'virus', 'therapy', 'drug', 'exp'
    :param factor_2: y-axis of the graph, possible entries 'virus', 'therapy', 'drug', 'exp'
    :param single_sentece_plots: If true, plot only coocurrences in the same sentence
    """
    if single_sentence_plots:
        grouped = dat_joined[dat_joined.same_sentence is True].groupby(['word_' + factor_1, 'word_' + factor_2])
    else:
        grouped = dat_joined.groupby(['word_' + factor_1, 'word_' + factor_2])

    values = grouped.count().values[:, 0]

    index = grouped.count().index

    # Holds aggregated, in order appearences of factor_1 and factor_2 respectively
    index_1 = []
    index_2 = []
    for f_1, f_2 in index:
        index_1.append(f_1)
        index_2.append(f_2)

    uniq_1 = np.unique(index_1)
    uniq_2 = np.unique(index_2)

    # Populates index_1 and index_2 with the index they appear in uniq_1 and uniq_2 respectively
    for i in range(0, len(index_1)):
        index_1[i] = np.where(index_1[i] == uniq_1)[0][0]
        index_2[i] = np.where(index_2[i] == uniq_2)[0][0]

    pylab.figure(figsize=(5, 5), dpi=200)
    hist = pylab.hist2d(index_1, index_2, (range(
        0, len(uniq_1) + 1), range(0, len(uniq_2) + 1)), weights=values, cmap='Blues')
    pylab.xticks(np.arange(0, len(uniq_1)) + 0.5, uniq_1, rotation=90)
    pylab.yticks(np.arange(0, len(uniq_2)) + 0.5, uniq_2)
    pylab.clim(0, np.max(hist[0]) * 1.5)
    for i in range(0, len(uniq_1)):
        for j in range(0, len(uniq_2)):
            pylab.text(
                i + 0.5, j + 0.5, int(hist[0][i][j]), ha='center', va='center')

    pylab.colorbar()
    if single_sentence_plots:
        pylab.title(factor_1 + " and " + factor_2 + " in One Sentence")
        pylab.tight_layout()
        pylab.savefig("Overlap" + factor_1 + "_Vs_" + factor_2 + "_2D_sentence.png", bbox_inches='tight', dpi=200)
    else:
        pylab.title(factor_1 + " and " + factor_2 + " in One Block")
        pylab.tight_layout()
        pylab.savefig("Overlap" + factor_1 + "_Vs_" + factor_2 + "_2D_block.png", bbox_inches='tight', dpi=200)


def convert_date_to_year(datestring: str):
    """Extract the year from the inconsistently formatted metadata.

    :param datestring: string in datetime format

    :return: datetime obj if success, 0 if failure
    """
    if pd.notna(datestring):
        try:
            date = parser.parse(str(datestring), fuzzy=True)
            return date.year
        except ValueError:
            return 0
    else:
        return 0
