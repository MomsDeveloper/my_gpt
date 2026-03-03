import glob
from bpe import BPE

all_text = []
for file_path in glob.glob('./texts/*.*'):
    file = open(file_path, 'r', encoding='utf8')
    all_text.append(file.read())
    
all_text = '\n\n\n'.join(all_text)
print(sorted(set(all_text)))

bpe = BPE(2000)
bpe.fit(all_text)
bpe.save('./data/bpe.dill')

print(bpe.id2token.items())
