import snorkel

from snorkel.preprocess import preprocessor
from snorkel.preprocess.nlp import SpacyPreprocessor
from snorkel.types import DataPoint

spacy = SpacyPreprocessor(text_field="text", doc_field="doc", memoize=True)

def make_source_target_preprocessor(spacy, sources, targets):
    @preprocessor(pre=[spacy])
    def get_source_target(cand: DataPoint) -> DataPoint:
        """Returnsthe source and target mentioned in the sentence."""
        person_names = []

        source = [token.text for token in cand.doc if token.text in sources]
        target = [token.text for token in cand.doc if token.text in targets]

        try:
            cand.source_target = (source[0], target[0])
        except:
            cand.source_target = (np.nan, np.nan)
        return cand
    return get_source_target

def make_text_between_preprocessor(spacy, sources, targets):
    @preprocessor(pre=[spacy])
    def get_text_between(cand: DataPoint) -> DataPoint:
        """
        Returns the text between a source-target pair and the text to the left of the source
        """

        source_idx = [token.i for token in cand.doc if token.text in sources]
        target_idx = [token.i for token in cand.doc if token.text in targets]

        try:

            if (len(target_idx)==1) & (len(source_idx)==1) & (source_idx[0]<target_idx[0]):
                cand.text_between = cand.doc[source_idx[0]:target_idx[0]]
                cand.text_to_source_left = cand.doc[:source_idx[0]]

            elif (len(target_idx)>1) & (len(source_idx)==1):
                for target_index in target_idx:
                    if source_idx[0]<target_index:
                        cand.text_between = cand.doc[source_idx[0]:target_index]
                        cand.text_to_source_left = cand.doc[:source_idx[0]]

            elif (len(source_idx)>1) & (len(target_idx)==1):
                for source_index in source_idx:
                    if source_index<target_idx[0]:
                        cand.text_between = cand.doc[source_index:target_idx[0]]
                        cand.text_to_source_left = cand.doc[:source_index]

            elif (len(source_idx)>1) & (len(target_idx)>1):
                for source_index in source_idx:
                    for target_index in target_idx:
                        if source_index<target_index:
                            cand.text_between = cand.doc[source_index:target_index]
                            cand.text_to_source_left = cand.doc[:source_index]

            else:
                cand.text_between = 'NaN'
                cand.text_to_source_left = 'NaN'
        except:

            cand.text_between = 'NaN'
            cand.text_to_source_left = 'NaN'

        return cand
    return get_text_between
