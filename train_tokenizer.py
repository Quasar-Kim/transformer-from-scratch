from tokenizers import Tokenizer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.models import WordPiece
from tokenizers.trainers import WordPieceTrainer
from data import ChatbotDataset

train_dataset = ChatbotDataset(root='./chatbot_dataset', split='train', download=True)
test_dataset = ChatbotDataset(root='./chatbot_dataset', split='test', download=True)

def batch_iterator():
    for sample in train_dataset:
        yield sample['question']
        yield sample['answer']
    for sample in test_dataset:
        yield sample['question']
        yield sample['answer']

tokenizer = Tokenizer(WordPiece(unk_token='[UNK]'))
tokenizer.pre_tokenizer = Whitespace()
trainer = WordPieceTrainer(vocab_size=16000, special_tokens=['[BOS]', '[EOS]', '[PAD]', '[UNK]'])
tokenizer.train_from_iterator(batch_iterator(), trainer=trainer, length=(len(train_dataset) + len(test_dataset)))
tokenizer.enable_padding(pad_id=tokenizer.token_to_id('[PAD]'))
tokenizer.save('tokenizer.json')