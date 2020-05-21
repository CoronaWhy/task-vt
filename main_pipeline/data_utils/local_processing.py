# Temporary storage for utils used in the processing of local data structures, TODO are in place for code that probably shouldn't be modified until proper testing can be implemented
# This inital file just serves as a place to put the newly commented functions
import os
import json
import pandas as pd
import numpy as np
import pylab
import spacy
import dateutil.parser as parser

def MakeNGramMapAndList(filename):

    """ Given the path of a file of tokens, replace every " " with "_"
    Args:
        filename (str): The path of the file

    Returns:
        names (list): Original tokens with spaces instead of underscores
        ngram_map (dict): {names: names_with_space_repaced_by_underscores}
    """

    ngram_map={}
    names=[]
    allnames=[]

    file=open(filename).readlines()
    for line in file:
        allnames.append(line[:-1])

    for name in allnames:
        if(len(name)>5):
            if " " in name:
                if((len(name.split(" "))>1) and len(name.split(" "))<4):
                    newname=name.replace(" ","_")
                    ngram_map[name]=newname
                    names.append(newname)
            else:
                names.append(name)
    names=np.unique(names)
    return names, ngram_map


# TODO Pull the next 6 functions into a class?
# These functions determine what blocks are pulled from the paper for matching
def TitleBlocks(paper):
    """ Retrieve title block of a paper in json form
    Args:
        paper (json): The json form of the paper to be parsed
    Returns:
        (list): Lines of text in the paper title
    """
    return([{'text':paper['metadata']['title']}])

def AbstractBlocks(paper):
    """ Retrieve abstract block of a paper in json form
    Args:
        paper (json): The json form of the paper to be parsed
    Returns:
        (list): Lines of text in the paper abstract
    """
    return(paper['abstract'])

def BodyBlocks(paper):
    """ Retrieve body block of a paper in json form
    Args:
        paper (json): The json form of the paper to be parsed
    Returns:
        (list): Lines of text in the paper body
    """
    return(paper['body_text'])

# TODO the reference nlp is an external dependence, either add into function or make it a class
def PullMentionsLemmatized(Paths, BlockSelector,SecName, Words,replace_dict=None):
    """ Aggregates the positional features of the lemmatized text corpus
    Args:
        Paths (list): List of strings containing paths to the corpus of text to aggregate features from
        BlockSelector (function): The function to retrieve the relevant block from text
        SecName (string): Corresponds with BlockSelector to select the relevant block of text. Options are 'title', 'abstract', 'body' 
            TODO seems like this should be pulled into the function as it depends entirely on the BlockSelector argument
        Words (list): List of lemmatized words
        replace_dict (dict): Takes a dict of {token: lemmatized underscore token} and replaces occurences of token in text with its value
    Returns:
        (dict): {'identifier feature': list of occurrences}
    """
    Positions=[]
    FoundWords=[]
    Section=[]
    BlockID=[]
    BlockText=[]
    PaperID=[]

    tokenized_words=[]
    for w in Words:
        tokenized_words.append(nlp(w.lower())[0].lemma_)
    for Path in Paths:
        print(Path)

        Files=os.listdir(Path)
        for p in Files:

            readfile=open(Path+p,'r')
            paper=json.load(readfile)
            Blocks=BlockSelector(paper)

            for b in range(0,len(Blocks)):
                text=Blocks[b]['text'].lower()
                if(not replace_dict==None):
                    text=RunReplace(text,replace_dict)
                text=nlp(text)
                for t in text:
                    for w in tokenized_words:
                        if(w == t.lemma_):
                            Section.append(SecName)
                            FoundWords.append(w)
                            Positions.append(t.idx)
                            BlockText.append(Blocks[b]['text'])
                            BlockID.append(b)
                            PaperID.append(p[:-5])
    return {'sha':PaperID,'blockid':BlockID,'word':FoundWords,'sec':Section,'pos':Positions,'block':BlockText}


def PullMentionsDirect(Paths, BlockSelector,SecName, Words, replace_dict=None):
    """ Aggregates the positional features of the text corpus
    Args:
        Paths (list): List of strings containing paths to the corpus of text to aggregate features from
        BlockSelector (function): The function to retrieve the relevant block from text
        SecName (string): Corresponds with BlockSelector to select the relevant block of text. Options are 'title', 'abstract', 'body' 
            TODO seems like this should be pulled into the function as it depends entirely on the BlockSelector argument
        Words (list): List of words
        replace_dict (dict): Takes a dict of {token: underscore token} and replaces occurences of token in text with its value
    Returns:
        (dict): {'identifier feature': list of occurrences}
    """
    Positions=[]
    FoundWords=[]
    Section=[]
    BlockID=[]
    BlockText=[]
    PaperID=[]
    for wi in range(0,len(Words)):
        Words[wi]=Words[wi].lower()
    for Path in Paths:
        print(Path)

        Files=os.listdir(Path)
        for p in Files:

            readfile=open(Path+p,'r')
            paper=json.load(readfile)
            Blocks=BlockSelector(paper)

            for b in range(0,len(Blocks)):
                text=Blocks[b]['text'].lower()
                if(not replace_dict==None):
                    text=RunReplace(text,replace_dict)
                for w in Words:
                    if(w in text):
                        pos=text.find(w)
                   
                        #check we're not in the middle of another word
                        if(text[pos-1]==" " and ( (pos+len(w))>=len(text) or not text[pos+len(w)].isalpha())):
                            Section.append(SecName)
                            FoundWords.append(w)
                            Positions.append(text.find(w))
                            BlockText.append(Blocks[b]['text'])
                            BlockID.append(b)
                            PaperID.append(p[:-5])
    return {'sha':PaperID,'blockid':BlockID,'word':FoundWords,'sec':Section,'pos':Positions,'block':BlockText}


def RunReplace(block, replace_dict):
    """ Replace routine for dealing with n-grams
    Args:
        block (json): contains the text block replacement is done on
        replace_dict (dict): {token: replacement_token}
    Returns:
        block: with occurrences of token replaced with replacement_token
    """
    for k in replace_dict.keys():
        if(k in block):
            block=block.replace(k,replace_dict[k])
    return block


# TODO These paths are kaggle paths, replace first with github paths then DataVerse paths once setup
Paths=["/kaggle/input/CORD-19-research-challenge/noncomm_use_subset/noncomm_use_subset/pdf_json/","/kaggle/input/CORD-19-research-challenge/comm_use_subset/comm_use_subset/pdf_json/","/kaggle/input/CORD-19-research-challenge/biorxiv_medrxiv/biorxiv_medrxiv/pdf_json/","/kaggle/input/CORD-19-research-challenge/custom_license/custom_license/pdf_json/"]
# TODO Right now this function references an externally declared variable, suggest either moving into a class or make paths an argument
# Also function name might need to change, a little ambiguous
def ExtractToCSV(Words,Filename,Lemmatized=True, RunTitle=True, RunAbstract=True, RunBody=False,replace_dict=None):
    """ Stores aggregate features of vocab words in the main CORD-19 dataset
    Args: 
        Words (list): List of strings representing words of interest
        Filename (string): Path of the location to store final csv
        Lemmatized: If true, perform lemmatized feature extraction
        RunTitle: If true, features extracted from the title will be included in the aggregate 
        RunAbstract: If true, features extracted from the abstract will be included in the aggregate
        RunBody: If true, features extracted from the body will be included in the aggregate
        replace_dict (dict): {token: replacement_token}
    Returns:
        None
    """

    if(Lemmatized):
        PullMentions = PullMentionsLemmatized
    else:
        PullMentions = PullMentionsDirect
    
    DataDicts=[]
    if(RunTitle): 
        DataDicts.append(PullMentions(Paths, TitleBlocks,    "title",    Words, replace_dict))
    if(RunAbstract):
        DataDicts.append(PullMentions(Paths, AbstractBlocks, "abstract", Words, replace_dict))
    if(RunBody):
        DataDicts.append(PullMentions(Paths, BodyBlocks,     "body",     Words, replace_dict))

    SummedDictionary=DataDicts[0]
    for k in DataDicts[0].keys():
        for d in DataDicts:
            SummedDictionary[k]=SummedDictionary[k]+d[k]

    dat=pd.DataFrame(SummedDictionary)
    dat.to_csv(Filename)


# TODO change index to be of type int rather than string
def SameSentenceCheck(block,pos1,pos2):
    """ Checks if two words are in the same sentence by looking for sentence delimiters between their starting positions
    Args:
        block (string): Block of string text the two words are found in
        pos1 (string): The index of the beginning of word1
        pos2 (string): The index of the beginning of word2
    Returns:
        1: if they word1 and word2 are in the same sentence
        0: If word1 and word2 are separated by one of the follwing sentence delimiters ., ;, ?, !
    """
    if(pos1<pos2):
        Interstring=block[int(pos1):int(pos2)]
    else:
        Interstring=block[int(pos2):int(pos1)]
    SentenceEnders=[".",";","?","!"]
    for s in SentenceEnders:
        if s in Interstring:
            return 0
    return 1

# This function makes the 2D quilt plot for showing co-occurences at block
#   or sentence level of various classes of search terms
#
def Make2DPlot(dat_joined, factor1, factor2, single_sentence_plots=False):
    """ Creates 2D quilt plot from dataframe row columns ('word_' + factor1) and ('word_' + factor2)
    Args:
        dat_joined (pandas.df): Dataframe of the format returned by PullMentionsLemmatized and PullMentionsDirect
        factor1: x-axis of the graph, possible entries 'virus', 'therapy', 'drug', 'exp'
        factor2: y-axis of the graph, possible entries 'virus', 'therapy', 'drug', 'exp'
        single_sentece_plots: If true, plot only coocurrences in the same sentence
    Returns:
        None
    """
    if(single_sentence_plots):
        grouped = dat_joined[dat_joined.same_sentence==True].groupby(['word_'+factor1,'word_'+factor2])
    else:
        grouped = dat_joined.groupby(['word_'+factor1,'word_'+factor2])

    Values    = grouped.count().values[:,0]

    Index=grouped.count().index
    Index1=[]
    Index2=[]
    for i in Index:
        Index1.append(i[0])
        Index2.append(i[1])

    Uniq1=np.unique(Index1)
    Uniq2=np.unique(Index2)

    for i in range(0,len(Index1)):
        Index1[i]=np.where(Index1[i]==Uniq1)[0][0]
        Index2[i]=np.where(Index2[i]==Uniq2)[0][0]

    pylab.figure(figsize=(5,5),dpi=200)
    hist=pylab.hist2d(Index1,Index2, (range(0,len(Uniq1)+1),range(0,len(Uniq2)+1)), weights=Values,cmap='Blues')
    pylab.xticks(np.arange(0,len(Uniq1))+0.5, Uniq1,rotation=90)
    pylab.yticks(np.arange(0,len(Uniq2))+0.5, Uniq2)
    pylab.clim(0,np.max(hist[0])*1.5)
    for i in range(0,len(Uniq1)):
        for j in range(0,len(Uniq2)):
            pylab.text(i+0.5,j+0.5,int(hist[0][i][j]),ha='center',va='center')

    pylab.colorbar()
    if(single_sentence_plots):
        pylab.title(factor1+" and " +factor2+" in One Sentence")
        pylab.tight_layout()
        pylab.savefig("Overlap"+factor1+"_Vs_"+factor2+"_2D_sentence.png",bbox_inches='tight',dpi=200)
    else:
        pylab.title(factor1+" and " +factor2+" in One Block")
        pylab.tight_layout()
        pylab.savefig("Overlap"+factor1+"_Vs_"+factor2+"_2D_block.png",bbox_inches='tight',dpi=200)


def ConvertDateToYear(datestring):
    """ extract the year from the inconsistently formatted metadata
    Args (string): 
        string in datetime format
    Returns:
        datetime object: if success
        0: if failure
    """
    if(pd.notna(datestring)):
        try:
            date=parser.parse(str(datestring),fuzzy=True)
            return date.year
        except ValueError:
            return 0
    else:
        return 0
    

