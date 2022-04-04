import math, collections


class CustomModel:

    def __init__(self, corpus):
        """Initialize your data structures in the constructor."""
        self.unigramCounts = collections.defaultdict(lambda: 0)
        self.bigramCounts = collections.defaultdict(lambda: 0)
        self.trigramCounts = collections.defaultdict(lambda: 0)
        self.preBi = collections.defaultdict(lambda: 0)
        self.preTri = collections.defaultdict(lambda: 0)
        self.unicount = 0
        self.v = 0
        self.train(corpus)

    def train(self, corpus):
        """ Takes a corpus and trains your language model.
        Compute any counts or other corpus statistics in this function.
    """
        # TODO your code here
        # Tip: To get words from the corpus, try
        #    for sentence in corpus.corpus:
        #       for datum in sentence.data:
        #         word = datum.word
        for sentence in corpus.corpus:
            for i in range(len(sentence.data)):
                token1 = sentence.data[i].word
                self.unigramCounts[token1] += 1
                self.unicount += 1
                if i > 0:
                    token2 = sentence.data[i - 1].word
                    self.bigramCounts[(token2, token1)] += 1
                    self.preBi[token2] += 1
                if i > 1:
                    token2 = sentence.data[i - 1].word
                    token3 = sentence.data[i - 2].word
                    self.trigramCounts[(token3, token2, token1)] += 1
                    self.preTri[(token3, token2)] += 1

    def score(self, sentence):
        """ Takes a list of strings as argument and returns the log-probability of the
        sentence using your language model. Use whatever data you computed in train() here.
    """
        # TODO your code here
        for word in sentence:
            if word not in self.unigramCounts:
                self.v += 1

        score = 0.0
        if sentence[0] in self.unigramCounts:
            score += math.log(self.unigramCounts[sentence[0]] + 1)
            score -= math.log(self.unicount + self.v)

        if sentence[1] in self.bigramCounts:
            score += math.log(self.bigramCounts[(sentence[0], sentence[1])])
            score -= math.log(self.preBi[sentence[1]])

        elif sentence[1] in self.unigramCounts:
            score += math.log(self.unigramCounts[sentence[1]] + 1)
            score -= math.log(self.unicount + self.v)
            score += math.log(0.4)

        for i in range(2, len(sentence)):
            if (sentence[i - 2], sentence[i - 1], sentence[i]) in self.trigramCounts:
                score += math.log(self.trigramCounts[(sentence[i - 2], sentence[i - 1], sentence[i])])
                score -= math.log(self.preTri[(sentence[i - 2], sentence[i - 1])])

            elif (sentence[i - 1], sentence[i]) in self.bigramCounts:
                score += math.log(self.bigramCounts[(sentence[i - 1], sentence[i])])
                score -= math.log(self.preBi[sentence[i - 1]])
                score += math.log(0.4)

            else:
                score += math.log(self.unigramCounts[sentence[i]] + 1)
                score -= math.log(self.unicount + self.v)
                score += math.log(0.8)
        return score
