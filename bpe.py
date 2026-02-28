class BPE():
    def __init__(self, vocab_size: int) -> None:
        self.vocab_size = vocab_size
        self.id2token = {}
        self.token2id = {}
        self.uniquietokens = []

    def fit(self, text: str):
        tokens = list(text)
        self.uniquietokens = sorted(set(tokens))
        while len(self.uniquietokens) != self.vocab_size:
            pairs_freqs = {}
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                pairs_freqs[pair] = pairs_freqs.get(pair, 0) + 1
            if not pairs_freqs:
                break
            
            most_freq = max(pairs_freqs, key=pairs_freqs.get)

            new_token = most_freq[0] + most_freq[1]
            merged = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == most_freq[0] and tokens[i + 1] == most_freq[1]:
                    merged.append(new_token)
                    i+=2
                else:
                    merged.append(tokens[i])
                    i += 1
            tokens = merged
            self.uniquietokens.append(new_token)

        for idx in range(self.vocab_size):
            self.id2token[idx] = self.uniquietokens[idx]
            self.token2id.update({self.uniquietokens[idx]: idx})
        
text = 'Из кузова в кузов шла перегрузка арбузов. В грозу в грязи от груза арбузов развалился кузов.'
BPE(30).fit(text)
# print(sorted(set(text)))