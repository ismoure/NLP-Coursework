import sys
from collections import defaultdict
from collections import Counter
import math
import random
import os
import os.path
"""
COMS W4705 - Natural Language Processing - Fall 2022
Prorgramming Homework 1 - Trigram Language Models
Daniel Bauer
"""

def corpus_reader(corpusfile, lexicon=None): 
    with open(corpusfile,'r') as corpus: 
        for line in corpus: 
            if line.strip():
                sequence = line.lower().strip().split()
                if lexicon: 
                    yield [word if word in lexicon else "UNK" for word in sequence]
                else: 
                    yield sequence

def get_lexicon(corpus):
    word_counts = defaultdict(int)
    for sentence in corpus:
        for word in sentence: 
            word_counts[word] += 1
    return set(word for word in word_counts if word_counts[word] > 1)

def get_ngrams(sequence, n):
    """
    Given a sequence, this function should return a list of n-grams, where each n-gram is a Python tuple.
    This should work for any value n (pad with STARTS and STOPS)
    """
    ngrams = []

    if n==1:
        sequence.insert(0, "START")
    else:
        for i in range(n-1): # pad with STARTs
            sequence.insert(0, "START")

    sequence.append("STOP") # put STOP

    for i in range(len(sequence)-n+1):
        ngram = tuple(sequence[i+j] for j in range(n))
        ngrams.append(ngram)

    return ngrams

class TrigramModel(object):
    
    def __init__(self, corpusfile):
    
        # Iterate through the corpus once to build a lexicon 
        generator = corpus_reader(corpusfile)
        self.lexicon = get_lexicon(generator)
        self.lexicon.add("UNK")
        self.lexicon.add("START")
        self.lexicon.add("STOP")
    
        # Now iterate through the corpus again and count ngrams
        generator = corpus_reader(corpusfile, self.lexicon)
        self.count_ngrams(generator)


    def count_ngrams(self, corpus):
        """
        Given a corpus iterator, populate Counter dictionaries of unigram, bigram,
        and trigram counts. 
        """

        self.unigramcounts = Counter()
        self.bigramcounts = Counter()
        self.trigramcounts = Counter()

        for sentence in corpus:
            unigrams = get_ngrams(sentence, 1)
            for unigram in unigrams:
                self.unigramcounts[unigram] += 1

            bigrams = get_ngrams(sentence, 2)
            for bigram in bigrams:
                self.bigramcounts[bigram] += 1
          
            trigrams = get_ngrams(sentence, 3)
            for trigram in trigrams:
                self.trigramcounts[trigram] += 1
            

        self.wordtotal = self.unigramcounts.total() - self.unigramcounts["START"]
        # save total number of words

        self.bigramtotal = self.bigramcounts.total()
        self.trigramtotal = self.trigramcounts.total()

        return

    def raw_trigram_probability(self,trigram):
        """
        Returns the raw (unsmoothed) trigram probability
        """

        if tuple(trigram[0:2]) == ('START', 'START',):
            return self.raw_bigram_probability(tuple(trigram[1:2])) # P(START, START, word) = P(START, word)
        
        if self.bigramcounts[tuple(trigram[0:2])] == 0: # assume uniform dist if never seen bigram
            return 1/len(self.lexicon)

        return self.trigramcounts[trigram]/self.bigramcounts[tuple(trigram[0:2])]

    def raw_bigram_probability(self, bigram):
        """
        Returns the raw (unsmoothed) bigram probability
        """
        return self.bigramcounts[bigram]/self.unigramcounts[(bigram[0],)]
    
    def raw_unigram_probability(self, unigram):
        """
        Returns the raw (unsmoothed) unigram probability.
        """
        return self.unigramcounts[unigram]/self.wordtotal

    def smoothed_trigram_probability(self, trigram):
        """
        Returns the smoothed trigram probability (using linear interpolation). 
        """

        lambda1 = 1/3.0
        lambda2 = 1/3.0
        lambda3 = 1/3.0

        unigram = (trigram[0],)
        bigram  = tuple(trigram[0:2])

        uniprob = self.raw_unigram_probability(unigram)
        biprob = self.raw_bigram_probability(bigram)
        triprob = self.raw_trigram_probability(trigram)

        return (lambda1 * triprob) + (lambda2 * biprob) + (lambda3 * uniprob)
        
    def sentence_logprob(self, sentence):
        """
        Returns the log probability of an entire sequence.
        """
        probability = 0
        trigrams = get_ngrams(sentence, 3)

        for trigram in trigrams:
            #print(self.smoothed_trigram_probability(trigram))
            probability += math.log2(self.smoothed_trigram_probability(trigram))

        return probability

    def perplexity(self, corpus):
        """
        Returns the log probability of an entire sequence.
        """

        total_prob = 0
        l = 0
        word_total = 0

        for sentence in corpus:
            l += self.sentence_logprob(sentence)
            word_total += len(sentence)

        l = l/word_total

        return 2 ** (-1 * l)


def essay_scoring_experiment(training_file1, training_file2, testdir1, testdir2):

    model1 = TrigramModel(training_file1) # high trained
    model2 = TrigramModel(training_file2) # low trained

    total = 0
    correct = 0       
 
    for f in os.listdir(testdir1): # testing high
        pp_high = model1.perplexity(corpus_reader(os.path.join(testdir1, f), model1.lexicon))  # high with high trained
        pp_low = model2.perplexity(corpus_reader(os.path.join(testdir1, f), model2.lexicon)) # high with low trained

        if pp_low > pp_high:
            correct += 1
        total += 1
    
    
    for f in os.listdir(testdir2): # testing low
        pp_high = model1.perplexity(corpus_reader(os.path.join(testdir2, f), model1.lexicon))  # low with high trained
        pp_low = model2.perplexity(corpus_reader(os.path.join(testdir2, f), model2.lexicon)) # low with low trained
            
        if pp_low < pp_high:
            correct += 1
        total += 1
        
    return correct/total

if __name__ == "__main__":

    model = TrigramModel("brown_train.txt")
  
    # Testing perplexity: 
    # dev_corpus = corpus_reader("brown_train.txt", model.lexicon)
    # pp = model.perplexity(dev_corpus)
    # print(pp)

    # Essay scoring experiment: 
    acc = essay_scoring_experiment("train_high.txt", "train_low.txt", "test_high", "test_low")
    print(acc)
