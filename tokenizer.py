import os
from transformers import PreTrainedTokenizerFast
from tokenizers import decoders

class WordPieceTokenizer(PreTrainedTokenizerFast):
    def __init__(self, parallelism=False):
        if not parallelism:
            os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        super().__init__(tokenizer_file='tokenizer.json')
        self.bos_token = '[BOS]'
        self.eos_token = '[EOS]'
        self.pad_token = '[PAD]'
        self.unk_token = '[UNK]'
        self._decoder = decoders.WordPiece()

    def decode(self, token_ids, **kwargs):
        tokens = self.convert_ids_to_tokens(token_ids)
        decoded = self._decoder.decode(tokens)
        return decoded

    
    