class BPE():
    def __init__(self, vocab_size: int) -> None:
        self.vocab_size = vocab_size
        self.id2token = {}
        self.token2id = {}
        self.uniquietokens = []

    def fit(self, text: str) -> None:
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
    
    def encode(self, text:str) -> list:
        split_text = list(text)
        tokenized_text = []
        idx = 0
        while idx < len(split_text):
            candidates = [t for t in self.uniquietokens if t[0] == split_text[idx]]
            candidates = sorted(candidates, key=lambda x: len(x), reverse=True)
            matched = False
            for elem in candidates:
                if text[idx: idx + len(elem)] == elem:
                    tokenized_text.append(elem)
                    idx += len(elem)
                    matched = True
                    break
            if not matched:
                tokenized_text.append(split_text[idx])
                idx += 1
        encoded_text = [self.token2id.get(elem) for elem in tokenized_text]
        return encoded_text
    
    def decode(self, token_ids: list[int]) -> str:
        tokens = [self.id2token.get(elem) for elem in token_ids]
        return ''.join(tokens)



text = 'Из кузова в кузов шла перегрузка арбузов. В грозу в грязи от груза арбузов развалился кузов.'
model = BPE(30)
model.fit(text)
encoded = model.encode(text)
print(model.decode(encoded))
