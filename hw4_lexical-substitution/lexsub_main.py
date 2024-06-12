#!/usr/bin/env python
import sys

from lexsub_xml import read_lexsub_xml
from lexsub_xml import Context 

# suggested imports 
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords

import numpy as np
import tensorflow

import gensim
import transformers 

from typing import List
import string
from collections import Counter


stop_words = stopwords.words('english')

def tokenize(s): 
    """
    a naive tokenizer that splits on punctuation and whitespaces.  
    """
    s = "".join(" " if x in string.punctuation else x for x in s.lower()) 
    s = s.split()
    for word in s:
        if word in stop_words: # remove function words
            s.remove(word)
    return s

def get_candidates(lemma, pos) -> List[str]:

    canidates = set()
    lemmas = wn.lemmas(lemma, pos = pos)

    for lem in lemmas:
        synset = lem.synset()
        synlemmas = synset.lemmas()
        for synlem in synlemmas:
            if synlem.name().replace("_", " ") != lemma: # don't insert original lemma
                canidates.add(synlem.name().replace("_", " "))

    return list(canidates)

def smurf_predictor(context : Context) -> str:
    """
    suggest 'smurf' as a substitute for all words.
    """
    return 'smurf'

def wn_frequency_predictor(context : Context) -> str:

    canidates = Counter()
    lemmas = wn.lemmas(context.lemma, pos = context.pos)

    for lem in lemmas:
        synset = lem.synset()
        synlemmas = synset.lemmas()
        for synlem in synlemmas:
            if synlem.name().replace("_", " ") != context.lemma: # don't insert original lemma
                canidates[synlem.name().replace("_", " ")] += synlem.count() # dictionary of name : count

    return canidates.most_common(1)[0][0]

def wn_simple_lesk_predictor(context : Context) -> str:
    
    lemmas = wn.lemmas(context.lemma, pos = context.pos)
    best_substitute = ("smurf", 0)

    for lem in lemmas:
        synset = lem.synset()
        synlemmas = synset.lemmas()

        target_in_syn = 0
        syncontext = set()
        syncontext.update(tokenize(synset.definition()))
        for ex in synset.examples():
            syncontext.update(tokenize(ex))
        
        for word in synset.lemmas():
            if word.name() == context.lemma:
                target_in_syn = word.count()

        hypers = synset.hypernyms() # add hypernym context
        for hyper in hypers:
            syncontext.update(tokenize(hyper.definition()))
            for ex in hyper.examples():
                syncontext.update(tokenize(ex))

        full_context = tokenize(" ".join(context.right_context + context.left_context))

        overlap = len([value for value in full_context if value in list(syncontext)]) # list comprehension idea from GeeksforGeeks

        for synlem in synlemmas:
            if synlem.name() != context.lemma: # don't include original word     
                score = (1000 * overlap) + (100 * target_in_syn) + synlem.count() # weighted score idea from Prof Bauer on EdStem

                if score > best_substitute[1]:
                    best_substitute = (synlem.name().replace("_", " "), score)

    return best_substitute[0]
   

class Word2VecSubst(object):
        
    def __init__(self, filename):
        self.model = gensim.models.KeyedVectors.load_word2vec_format(filename, binary=True)    

    def predict_nearest(self,context : Context) -> str:

        canidates = set()
        lemmas = wn.lemmas(context.lemma, pos = context.pos)
        best_substitute = ("smurf", 0)

        for lem in lemmas:
            synset = lem.synset()
            synlemmas = synset.lemmas()
            for synlem in synlemmas:
                if synlem.name().replace("_", " ") != context.lemma: # don't insert original lemma
                    canidates.add(synlem.name())
        
        
        for can in canidates:
            try:
                if self.model.similarity(context.lemma, can) > best_substitute[1]:
                    best_substitute = (can.replace("_", " "), self.model.similarity(context.lemma, can))

            except KeyError: # continue looping through if word is not in Word2Vec
                pass
            

        return best_substitute[0]


class BertPredictor(object):

    def __init__(self): 
        self.tokenizer = transformers.DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.model = transformers.TFDistilBertForMaskedLM.from_pretrained('distilbert-base-uncased')

    def predict(self, context : Context) -> str:

        canidates = get_candidates(context.lemma, context.pos)

        full_context = context.left_context + ["[MASK]"] + context.right_context
        full_context = " ".join(full_context)

        input_toks = self.tokenizer.tokenize(full_context)
        mask_idx = input_toks.index("[MASK]")
        input_toks = self.tokenizer.encode(input_toks)
        input_mat = np.array(input_toks).reshape((1,len(input_toks)))
        outputs = self.model.predict(input_mat, verbose=0)
        predictions = outputs[0]
        best_words = np.argsort(predictions[0][mask_idx])[::-1] # Sort in increasing order
        best_words = self.tokenizer.convert_ids_to_tokens(best_words)
        best_words.remove(context.lemma)

        for word in best_words:
            if word in canidates:
                return word
        
        return best_words[0]


class IsobelPredictor(object): # part 6

    def __init__(self, filename): 
        self.tokenizer = transformers.DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.model = transformers.TFDistilBertForMaskedLM.from_pretrained('distilbert-base-uncased')
        self.W2Vmodel = gensim.models.KeyedVectors.load_word2vec_format(filename, binary=True)    

    def predict(self,context : Context) -> str:

        canidates = set()
        lemmas = wn.lemmas(context.lemma, pos = context.pos)
        best_substitute = ("smurf", np.NINF)

        for lem in lemmas:
                    synset = lem.synset()
                    synlemmas = synset.lemmas()
                    for synlem in synlemmas:
                        if synlem.name().replace("_", " ") != context.lemma: # don't insert original lemma
                            canidates.add(synlem.name())

        
        full_context = context.left_context + ["[MASK]"] + context.right_context
        mask_idx = full_context.index("[MASK]")
        full_context = " ".join(full_context)

        input_toks = self.tokenizer.encode(full_context)
        input_mat = np.array(input_toks).reshape((1,len(input_toks)))
        outputs = self.model.predict(input_mat, verbose=0)
        predictions = outputs[0]
        best_words = np.argsort(predictions[0][mask_idx])[::-1] # Sort in increasing order
        best_words = self.tokenizer.convert_ids_to_tokens(best_words)
        best_words.remove(context.lemma) # remove original word

        for can in canidates: # evaluate canidates using weighted score of relatedness and index in sorted Bert predictions
            try:
                score = (self.W2Vmodel.similarity(can, context.lemma)*1000) - best_words.index(can.replace("_", " "))
                if score > best_substitute[1]:
                    best_substitute = (can.replace("_", " "), score)

            except (KeyError, ValueError) as e: # continue looping through if word is not in Word2Vec or Bert
                pass
        
        return best_substitute[0]



if __name__=="__main__":

    # At submission time, this program should run your best predictor (part 6).

    W2VMODEL_FILENAME = 'GoogleNews-vectors-negative300.bin.gz'
    #predictor = Word2VecSubst(W2VMODEL_FILENAME)
    #predictor = BertPredictor()
    predictor = IsobelPredictor(W2VMODEL_FILENAME)

    for context in read_lexsub_xml("lexsub_trial.xml"):

        prediction = predictor.predict(context)
        print("{}.{} {} :: {}".format(context.lemma, context.pos, context.cid, prediction))
