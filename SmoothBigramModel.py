import math, collections

class SmoothBigramModel:

  def __init__(self, corpus):
    """Initialize your data structures in the constructor."""
    self.unigramCounts = collections.defaultdict(lambda: 0)
    self.bigramCounts = collections.defaultdict(lambda: 0)
    self.v = 0
    self.total = 0
    self.train(corpus)

  def train(self, corpus):
    """ Takes a corpus and trains your language model. 
        Compute any counts or other corpus statistics in this function.
    """  
    # TODO your code here
    for sentence in corpus.corpus:
      for i in range(len(sentence.data) - 1):
        token1 = sentence.data[i].word
        token2 = sentence.data[i + 1].word
        self.unigramCounts[token1] = self.unigramCounts[token1] + 1
        self.bigramCounts[(token1, token2)] = self.bigramCounts[(token1, token2)] + 1
        self.total += 1

      if len(sentence.data) > 1:
        self.unigramCounts[sentence.data[-1].word] += 1

  def score(self, sentence):
    """ Takes a list of strings as argument and returns the log-probability of the 
        sentence using your language model. Use whatever data you computed in train() here.
    """
    # TODO your code here
    score = 0.0
    for i in range(len(sentence)-1):
      if (sentence[i], sentence[i+1]) not in self.bigramCounts:
        self.v += 1

    for i in range(1, len(sentence)):
      unicount = self.unigramCounts[sentence[i]]
      bicount = self.bigramCounts[(sentence[i-1], sentence[i])]

      score += math.log(bicount + 1)
      score -= math.log(unicount + self.v)
      # Ignore unseen wordsn
    return score
