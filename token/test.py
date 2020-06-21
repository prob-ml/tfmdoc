from tokenizers import Tokenizer, trainers
from tokenizers.models import BPE
from tokenizers.normalizers import Lowercase
from tokenizers.pre_tokenizers import CharDelimiterSplit

# We build our custom tokenizer:
tokenizer = Tokenizer(BPE())
tokenizer.normalizer = Lowercase()
tokenizer.pre_tokenizer = CharDelimiterSplit(' ')

# We can train this tokenizer by giving it a list of path to text files:
trainer = trainers.BpeTrainer(special_tokens=[ "[UNK]" ])

files = [str(x) for x in Path(".").glob("**/test_X.txt")]
print(files)
for file in files:
    tokenizer.train(trainer, file)

# And now it is ready, we can save the vocabulary with
tokenizer.model.save('./save_directory', 'tokenizer-name')

# And simply use it
tokenizer.encode('Hello_there!')