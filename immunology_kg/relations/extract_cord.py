#-*- coding: utf-8 -*-
"""
Extract text content from articles in the CORD-19 dataset.
You must download the CORD-19 dataset first. This was last
tested with CORD-19 v25 from 2020-05-30.
"""
import os
import json

import pandas as pd

from utils import setup_logger

logger = setup_logger(name="extract-cord")

class ArticleContent():
    """Text content extracted from a CORD-19 article json file."""
    def __init__(self, filepath=None, pmc_id=None):
        self.filepath = filepath
        self.pmc_id = pmc_id
        self.data = self._get_json_data()

    @property
    def paper_id(self) -> str:
        return self.data['paper_id']

    @property
    def title(self) -> str:
        return self.data['metadata']['title']

    @property
    def abstracts(self) -> list:
        abstracts = []
        if "abstract" in self.data['metadata']:
            for j in range(len(self.data['metadata']["abstract"])):
                abstracts.append(self.data['metadata']['abstract'][i]['text'])
        return abstracts

    @property
    def table_captions(self) -> list:
        texts = []
        for item in self.data['ref_entries']:
            texts.append(self.data['ref_entries'][item]['text'])
        return texts

    @property
    def texts(self) -> dict:
        """Return a dict of section to list of texts."""
        texts = {}
        for item in self.data['body_text']:
            if 'section' in item:
                section = item['section']
            else:
                section = ''
            text = item['text']
            if section in texts:
                texts[section].append(text)
            else:
                texts[section] = [text]
        return texts

    def _get_json_data(self):
        with open(self.filepath, 'r', encoding='utf-8') as infile:
            data = json.load(infile)
        return data
    
    def __repr__(self):
        return "{}({})".format(
            type(self).__name__,
            ' '.join([
                f"filepath={self.filepath}",
                f"pmc_id={self.pmc_id}"
            ])
        ) 
        
    def __str__(self):
        return f"<{repr(self)}>"


def find_pmc_jsons(df, direc):
    files = {}
    filtered_df = df[df[f'pmc_json_files'].notnull()]
    for i in filtered_df.index:
        cord_uid = df.iloc[i].cord_uid
        pmc_id = df.iloc[i].pmcid
        pmc_file = df.iloc[i].pmc_json_files
        pmc_path = os.path.join(direc, pmc_file)
        if os.path.exists(pmc_path):
            files[pmc_id] = pmc_path
    return files

def find_pdf_jsons(df, direc):
    files = {}
    filtered_df = df[df[f'pdf_json_files'].notnull()]
    for i in filtered_df.index:
        cord_uid = df.iloc[i].cord_uid
        pdf_id = df.iloc[i].sha
        pdf_file = df.iloc[i].pdf_json_files
        pdf_path = os.path.join(direc, pdf_file)
        if os.path.exists(pdf_path):
            files[pdf_id] = pdf_path
    return files


if __name__ == '__main__':
    direc = '/home/tchistiak/coronawhy/2020-05-30'
    logger.info(f"Finding CORD-19 json articles in {direc}")
    df = pd.read_csv(os.path.join(direc, 'metadata.csv'), sep=',')
    cord_jsons = find_pmc_jsons(df, direc)
    articles = []
    for k in cord_jsons:
        content = ArticleContent(filepath=cord_jsons[k], pmc_id=k)
        articles.append(content)
    logger.info(articles[0])
        