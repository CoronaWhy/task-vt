#-*- coding: utf-8 -*-
"""
Use SpaCy NLP scientific models
"""
import re
import json
from typing import Optional

import pandas as pd
import spacy
import scispacy
from spacy_langdetect import LanguageDetector
from scispacy.abbreviation import AbbreviationDetector
from scispacy.umls_linking import UmlsEntityLinker


SCIMODELS = [
    "en_core_sci_lg", 
    "en_ner_craft_md", 
    "en_ner_jnlpba_md", 
    "en_ner_bc5cdr_md", 
    "en_ner_bionlp13cg_md"
]

stopwords_path = "stopwords-all.json"
with open(stopwords_path, 'r', encoding='utf-8') as infile:
    STOPWORDS = json.load(infile)

class Document(object):
    """A document with a list of tokenized sentences and other metadata."""
    def __init__(
            self,
            tokenized_sentences: Optional[list]=None,
            entities: Optional[list]=None
        ):
        self.tokenized_sentences = tokenized_sentences
        if tokenized_sentences is None:
            self.tokenized_sentences = []
        self.entities = entities
        if entities is None:
            self.entities = []

    def __repr__(self):
        return "{}({})".format(
            type(self).__name__,
            ', '.join([
                f'tokenized_sentences="{self.tokenized_sentences}"',
                f'entities="{self.entities}"',
            ])
        )

    def __str__(self):
        return f"<{repr(self)}>" 

class Entity(object):
    """A named entity, extracted from some text."""
    def __init__(
            self, 
            canonical_name: Optional[str]=None,
            token: Optional[str]=None, 
            string: Optional[str]=None, 
            umls_id: Optional[str]=None, 
            start_token: Optional[int]=None,
            end_token: Optional[int]=None,
            start_char: Optional[int]=None,
            end_char: Optional[int]=None,
        ):
        self.token = token
        self.umls_id = umls_id
        self.canonical_name = canonical_name
        self.start_token = start_token
        self.end_token = end_token
        self.start_char = start_char
        self.end_char = end_char
        self.string = string

    def to_dict(self):
        return dict(
            token=f"{self.token}",
            string=f"{self.string}",
            start_token=self.start_token,
            end_token=self.end_token,
            umls_id=f"{self.umls_id}",
            canonical_name=f"{self.canonical_name}",
            start_char=self.start_char,
            end_char=self.end_char,
        )

    def __repr__(self):
        d = self.to_dict()
        return "{}({})".format(
            type(self).__name__,
            ', '.join([f"{k}={d[k]}" for k in d])
        )

    def __str__(self):
        return f"<{repr(self)}>" 




def init_nlp(
        model: Optional[str]="en_core_sci_lg", 
        seg_sents: Optional[bool]=False
    ) -> tuple:
    """
    Initialize an nlp pipeline.

    Args:
        model (str): the name of an installed model from SpaCy
        seg_sents (bool): segment texts fed into this model into
            sentences first (default=False, i.e. the texts fed to the
            model will be a list of sentences)

    Returns:
        nlp: SpaCy NLP pipeline
        linker: entity linker (also used in the pipeline)
    """
    nlp = spacy.load(model)
    nlp.max_length=2000000

    #don't use sentence segmentation if it's not needed
    if not seg_sents:
        nlp.add_pipe(_prevent_sbd, before='tagger')

    #detect language to avoid parsing non-english text as if it were English
    nlp.add_pipe(LanguageDetector(), name='language_detector', last=True)

    #add the abbreviation pipe to the spacy pipeline
    abbreviation_pipe = AbbreviationDetector(nlp)
    nlp.add_pipe(abbreviation_pipe, before='tagger')

    #linker looks ups named entities/concepts in UMLS graph, normalizes data
    linker = UmlsEntityLinker(resolve_abbreviations=True)
    nlp.add_pipe(linker)
    
    return nlp, linker

def extract_abbrevs(doc: spacy.tokens.Doc) -> dict:
    """
    Extract abbreviations from SpaCy doc. Return a dict of abbrev to long form.
    """
    abbrevs = {}
    if len(doc._.abbreviations) > 0:
        for abbrev in doc._.abbreviations:
            #Increase length so "a" and "an" don't get un-abbreviated
            if len(abbrev._.long_form) > 4: 
                abbrev_str = str(doc.text[abbrev.start_char:abbrev.end_char])
                abbrevs[abbrev_str] = abbrev._.long_form
    return abbrevs

def expand_abbrevs(sentence: str, abbrevs: dict) -> list:
    """
    Return a sentence with expanded abbreviations.

    Args:
        sentence (str): the sentence as a string 
        abbrevs (dict): a dictionary of abbreviation to long form to expand

    Returns:
        sent_expanded (str): the sentence with abbreviations expanded
    """
    sent_str_expanded = sentence
    for k in abbrevs:
        sent_str_expanded = sent_str_expanded.replace(k, abbrevs[k].text)
    sent_expanded = sent_str_expanded.split()
    return sent_expanded


def is_stop(token, lang='en'):
    """
    Check if token is one of the most common words in the language.
    Stopwords list from: https://github.com/6/stopwords-json
    """
    stop_words = STOPWORDS[lang]
    if token in stop_words or token.lower() in stop_words:
        return True
    return False

def run_nlp(texts: list, model: Optional[str]="en_core_sci_lg", unabbrev=True) -> list:
    """
    Extract the list of text documents into documents of tokenized sentences 
    entities for each sentence.

    Args:
        texts (list): a list of strings
        model (str): the name of the installed SpaCy model to use
    
    Returns:
        documents (list): a list of Document objects (incl sents and entities)
    """
    #load nlp in here in case we parallelize this func (e.g. w/ joblib) later?
    nlp, linker = init_nlp(model=model, seg_sents=False)

    documents = []

    #use nlp.pipe parallization from spacy, because it's faster
    docs = nlp.pipe(texts)
    for i, doc in enumerate(docs):
        abbrevs = extract_abbrevs(doc)

        document = Document()
        
        for sent in doc.sents:
            tokens = [token.text for token in sent]
            tokens = expand_abbrevs(' '.join(tokens), abbrevs)

            document.tokenized_sentences.append(tokens)

            sent_ents = []
            for ent in sent.ents:
                result = char_idx_to_token_idx(
                    ' '.join(tokens), 
                    ent.start_char, 
                    ent.end_char  
                )
                if not result:
                    #the whole entity doesn't exist in the sent (NER mistake)
                    continue

                entity = Entity()
                entity.start_token, entity.end_token, entity.string = result
                entity.start_char = ent.start_char
                entity.end_char = ent.end_char
                entity.token = ' '.join(
                    tokens[entity.start_token:entity.end_token])

                if (
                        is_stop(entity.token) or 
                        not re.search('[a-zA-Z]', str(entity.token))
                   ):
                    continue
            
                if len(ent._.umls_ents) > 0:
                    entity.umls_id = ent._.umls_ents[0][0]
                    name = linker.umls.cui_to_entity[entity.umls_id].canonical_name
                    entity.canonical_name = name

                sent_ents.append(entity)

            document.entities.append(sent_ents)

        documents.append(document)

    return documents 

def char_idx_to_token_idx(
        sentence: str, 
        char_start: int, 
        char_end: int
    ) -> tuple:
    """
    Convert string character indicies into token indicies, where tokens
    are space-separated words of the string.

    Args:
        sentence (str): the sentence as a string
        char_start (int): the index in the string where the first character
            of the desired word token begins
        char_start (int): the index in the string where the last character
            of the desired word token ends

    Returns:
        token_start (int): the index in the space-tokenized list of words in
            the sentence where the desired tokens begin (inclusive on the left)
        token_end (int): the index in the space-tokenized list of words in
            the sentence where the desired tokens end (exclusive on the right)
        term (int): the matched word tokens themselves
    """
    token_end = 0
    token_start = 0
    ending = False
    term = ''
    for i, char in enumerate(sentence):
        if char == ' ':
            token_end += 1
        if i == char_start:
            token_start = token_end 
        if i >= char_start:
            term += char
        if i == char_end:
            return token_start, token_end, term.strip()

def _prevent_sbd(doc):
    """
    If you already have one sentence per line in your file
    you may wish to disable sentence segmentation with this function,
    which is added to the nlp pipe before the tagger
    """
    for token in doc:
        token.is_sent_start = False
    return doc
