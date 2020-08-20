#-*- coding: utf-8 -*-
"""
Download the Frauenhofer Knowledge Graph. Extract data from the graph
covering sentences, entity relations, and paper citation metadata.
"""

import re
import json
import requests

import numpy as np
import pandas as pd
import pybel
import gilda
from fuzzywuzzy import fuzz

from spacify import run_nlp, SCIMODELS 
from utils import setup_logger

logger = setup_logger(name="get-frauenhofer")

FRAUENHOFER_CSV = "covid19_frauenhofer_kg.csv"
FRAUENHOFER_JSON = "covid19_frauenhofer_kg.json"
#FRAUENHOFER_URL = 'https://github.com/covid19kg/covid19kg/raw/master/covid19kg/_cache.bel.nodelink.json'
FRAUENHOFER_URL = 'https://github.com/CoronaWhy/bel4corona/raw/master/data/covid19kg/covid19-fraunhofer-grounded.bel.nodelink.json'

def download_frauenhofer():
    res = requests.get(FRAUENHOFER_URL)
    res_json = res.json()
    with open(FRAUENHOFER_JSON, 'w', encoding='utf-8') as outfile:
        json.dump(res_json, outfile)
    graph = pybel.from_nodelink(res_json)
    #col0: subject (type e.g. chemical, genes, proteins, etc. and entity)
    #col1: predicate (relationship) 
    #col2: object (type e.g. chemical, genes, proteins, etc. and entity)
    #col4: raw annotation data
    pybel.to_csv(graph, FRAUENHOFER_CSV) #pybel's csv sep='\t'

def load_frauenhofer_json():
    with open(FRAUENHOFER_JSON, 'r', encoding='utf-8') as infile:
        data = json.load(infile)
    graph = pybel.from_nodelink(data)
    return data, graph


def get_entity_names(node):
    """
    Recursively walk nodes in the json pybel graph to get all 
    entity name strings in the graph and their concept.
    """
    if "concept" in node:
        names = {}
        names[node["concept"]["name"]] = node["concept"]
        return names
    elif "members" in node:
        names = {}
        for item in node["members"]:
            names.update(get_entity_names(item))
        return names
    elif "reactants" in node:
        names = {}
        for item in node["reactants"]:
            names.update(get_entity_names(item))
        return names
    else:
        return {}

def get_entity_names_from_graph(data):
    """
    Get all the entity name strings and their concept from the graph-as-json.
    """
    names = {}
    skipped = 0
    for i in range(len(data["nodes"])):
        result = get_entity_names(data["nodes"][i])
        if not result:
            skipped += 1
        names.update(result)
    logger.info("Skipped nodes without names: {} / {} ({:.1f}%) entries".format(
        skipped, i, (skipped / i)*100))
    return names

def collect_texts(data):
    """
    Read the PyBel graph-as-json-dict structure of the Frauenhofer dataset
    and extract all the necessary data to construct a list of lists
    consisting of the [sentence, source, relation, target, metadata].

    NOTE: this is distinct from using pybel.to_csv in that the 
    source and target entities are simple names here, without all
    all the extra graph structure annotations. We still make sure to
    keep all of that graph metadata in the 'link' though.
    """
    entries = []
    fail = 0
    pmc_citations = 0
    doi_citations = 0
    for i in range(len(data["links"])):
        link = data['links'][i]
        src_idx = link['source']
        tgt_idx = link['target']
        source = data['nodes'][src_idx]
        target = data['nodes'][tgt_idx]
        relation = link['relation']
        src_term = get_entity_names(source)
        tgt_term = get_entity_names(target)
        pmc_id = None
        doi_id = None

        if 'citation' in link:
            citation = link['citation']
            if 'db' in citation and citation['db'] == 'PubMed' and 'db_id' in citation:
                pmc_id = citation['db_id']
                pmc_citations += 1
            if 'db' in citation and citation['db'] == 'DOI' and 'db_id' in citation:
                doi_id = citation['db_id']
                doi_citations += 1

        if 'evidence' in link:
            text = link['evidence']
        else:
            #use an empty string instead of None so that when we feed the whole
            #list of sentences to Spacy later we don't run into type problems
            text = ''
            fail += 1

        entries.append([
            text, 
            src_term, 
            relation, 
            tgt_term, 
            link, 
            pmc_id, 
            doi_id
        ])
        
    missing_citations = i - (pmc_citations + doi_citations)
    logger.info("Text evidence not available for {} / {} ({:.2f}%) entries".format(
        fail, i, (fail / i)*100))
    logger.info("Citations not available for {} / {} ({:.2f}%) entries".format(
        missing_citations, 
        i, 
        (missing_citations / i) * 100
    ))
    return entries

def get_cited_sentences(data):
    """
    Extract Frauenhofer sentences and relations with paper
    citations into a pandas dataframe.
    """
    entries = collect_texts(data)
    df = pd.DataFrame(
        entries,
        columns=['text', 'source', 'relation', 'target', 'link', 'pmc_id', 'doi_id']
    )
    cite_df = df.dropna(subset=['pmc_id', 'doi_id'], how="all")
    return cite_df

def clean_entity_name(term):
    """
    Manual rules to strip notations that show up in the graph entity 
    that usually won't show up in natural language, to have a better 
    chance at matching.
    """
    to_strip = [
        "^\(\+\)-", 
        "^\(\w\)-", 
        "^[\d',]+[-\+]", 
        "^\w+\(\w\)-", 
        "[\d',]+[-\+]$", 
        "\(\w+\)$",
        "-$",
        "^-",
        "\d\.\d\.\d."
    ] 
    for s in to_strip:
        term = re.sub(s, '', term)
    term = term.strip()
    term = term.lower()
    return term

def same_term(extracted, annotated):
    """
    Check if a term extracted by SpaCy looks like the annotated term.
    
    Args:
        extracted (str): the term extracted from the text
        annotated (str): the term from the reference annotations

    Returns:
        same (bool): a boolean if the term is considered the same term
        metadata (tuple): a 3-tuple of the (namespace, id, name) with 
                          None's when grounding was unsuccessful
    """
    annot_matches = gilda.ground(annotated)
    extrt_matches = gilda.ground(extracted)

    if annot_matches and extrt_matches:
        annot_ns = annot_matches[0].term.db
        annot_ident = annot_matches[0].term.id
        annot_term = annot_matches[0].term.entry_name
        annot_grounded = (annot_ns, annot_ident, annot_term) 

        extrt_ns = extrt_matches[0].term.db
        extrt_ident = extrt_matches[0].term.id
        extrt_term = extrt_matches[0].term.entry_name
        extrt_grounded = (extrt_ns, extrt_ident, extrt_term) 

        if (
            annot_ident is not None and 
            (annot_ns == extrt_ns) and 
            (annot_ident == extrt_ident)
        ):
            return True, annot_grounded

    else: 
        annot = clean_entity_name(annotated)
        extrt = clean_entity_name(extracted)

        #first try the simple exact-match way
        if annot == extrt:
            return True, (None, None, annot)

        #then try fuzzy matching
        score = fuzz.partial_ratio(extrt, annot)
        if score is not None and score > 70:
            return True, (None, None, annot)
    
    return False, (None, None, None)

def add_spacy_nlp_data(df):
    """
    Use SpaCy to get additional metadata about sentences in the dataframe,
    namely, the set of entities found by using each of the scispacy models. 

    Each document we get as output from run_nlp is one text entry from the 
    Frauenhofer dataset, which will have the same number of sents and ents. 
    Example document, with two sentences and two entity lists (one per sent):
    
    tokenized_sentences: [
        ['sent1_tok1', 'sent1_tok2', 'sent1_tok3', ..., 'sent1_tokn], 
        ['sent2_tok1', 'sent2_ent2', ... 'sent2_tokn']
    ]
    entities: [
        ['sent1_ent1', 'sent1_ent2'], 
        ['sent2_ent1', 'sent2_ent2', ... 'sent2_entn']
    ]
    
    Since we are using all the different SCIMODELS, and they may give the
    same entities back, we would like to add the entities from all the
    models into a set. 
    
    For this, we are going to loop over each entity in each sentence in 
    each of the documents. We are going to add that entity into the set 
    of entities for this document to all_entities.

    We are further going to try to match that entity's token against the 
    Frauenhofer human annotated one, and only add the matching token into 
    the set of matched_entities.
    """
        
    matched_entities = [None] * len(df)
    all_entities = [None] * len(df)
    sources_matched = [0] * len(df)
    targets_matched = [0] * len(df)
    tokenized_texts = []

    for j, model in enumerate(SCIMODELS):

        documents = run_nlp(df['text'].to_list(), model=model)

        for i, doc in enumerate(documents):
            source = df.iloc[i]['source']
            target = df.iloc[i]['target']

            if j == 0: 
                #only need to add tokens output from the 1st model
                tokenized_texts.append(
                    [' '.join(sent) for sent in doc.tokenized_sentences]) 

                #init matched_entities w/ entries for each sent in this text 
                matched_entities[i] = [None] * len(doc.entities)
                all_entities[i] = [None] * len(doc.entities)

            for k, ents in enumerate(doc.entities):

                if matched_entities[i][k] is None:
                    matched_entities[i][k] = set()
                if all_entities[i][k] is None:
                    all_entities[i][k] = set()

                for ent in ents:
                    #we will add a string repr to a set to keep from having
                    #duplicates coming from each of the scispacy models

                    for term in source:
                        matched, grounded = same_term(ent.text, term)
                        if matched: 
                            ent.namespace, ent.identifier, ent.canonical_name = grounded
                            matched_entities[i][k].add(json.dumps(ent.to_dict()))
                            sources_matched[i] = 1

                    for term in target:
                        matched, grounded = same_term(ent.text, term)
                        if matched: 
                            ent.namespace, ent.identifier, ent.canonical_name = grounded
                            matched_entities[i][k].add(json.dumps(ent.to_dict()))
                            targets_matched[i] = 1

                    all_entities[i][k].add(json.dumps(ent.to_dict()))

    sources_matched = np.array(sources_matched)
    targets_matched = np.array(targets_matched)
    both_matched_entries = sum(sources_matched & targets_matched)
    sources_matched_entries = sum(sources_matched)
    targets_matched_entries = sum(targets_matched)

    logger.info("Matched sources: {} / {} ({:.2f}%)".format(
        sources_matched_entries, len(df), sources_matched_entries / len(df) * 100 ))
    logger.info("Matched targets: {} / {} ({:.2f}%)".format(
        targets_matched_entries, len(df), targets_matched_entries / len(df) * 100 ))
    logger.info("Matched sources AND targets: {} / {} ({:.2f}%)".format(
        both_matched_entries, len(df), both_matched_entries / len(df) * 100 ))

    return tokenized_texts, matched_entities, all_entities

def to_doccano(df, outfp):
    with open(outfp, 'w', encoding='utf-8') as outfile:
        for i in range(len(df)):
            texts = df.iloc[i]['tokenized_texts']
            entities = df.iloc[i]['all_entities']
            for j, sent in enumerate(texts):
                entry = {}
                sent_ents = entities[j]
                sent_ents = [json.loads(ent) for ent in sent_ents]
                entry['text'] = texts[j]
                entry['labels'] = [[ent['start_char'], ent['end_char'], ent['label']] 
                                   for ent in sent_ents]
                outfile.write(json.dumps(entry) + '\n')
                 

def main():
    download_frauenhofer()
    data, graph = load_frauenhofer_json()

    cite_df = get_cited_sentences(data)

    csv_output = 'covid19_frauenhofer_annotations.csv'
    cite_df.to_csv(csv_output)
    logger.info("Text annotations with citations: {} / {} ({:.2f}%)".format(
        len(cite_df), len(cite_df), (len(cite_df) / len(cite_df))*100))

    tokenized_texts, matched_entities, all_entities = add_spacy_nlp_data(cite_df)

    csv_output = 'covid19_frauenhofer_annotations_entities.csv'
    cite_df['matched_entities'] = matched_entities
    cite_df['all_entities'] = all_entities
    cite_df['tokenized_texts'] = tokenized_texts
    cite_df.to_csv(csv_output)

    doccano_outfile = "covid19_frauenhofer_spacy_ner.txt"
    to_doccano(cite_df, doccano_outfile)

    logger.info(f"Sample of {csv_output}:")
    logger.info(cite_df.head())
    logger.info(cite_df.tail())

    return cite_df


if __name__ == '__main__':
    main()





