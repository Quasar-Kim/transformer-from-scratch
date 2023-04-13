import torch
from data import ChatbotDataModule
from pathlib import Path
from tokenizer import WordPieceTokenizer

# this test takes too much time
# def test_download():
#     data_file_path = Path('./chatbot_dataset/chatBotData.csv')
#     data_file_path.unlink(missing_ok=True)
#     dm = ChatbotDataModule(data_dir='./chatbot_dataset', batch_size=32)
#     dm.prepare_data()
#     assert data_file_path.exists()

def test_tokenizer():
    tokenizer = WordPieceTokenizer()
    dm = ChatbotDataModule(batch_size=32, tokenizer=tokenizer)
    dm.prepare_data()
    dm.setup(stage='fit')
    loader = dm.train_dataloader()
    batch = next(iter(loader))
    assert batch['enc_x'].dtype == torch.int64 or batch['enc_x'].dtype == torch.int32
    assert batch['dec_x'].dtype == torch.int64 or batch['dec_x'].dtype == torch.int32
    assert batch['y'].dtype == torch.int64 or batch['y'].dtype == torch.int32

def test_data_shape():
    tokenizer = WordPieceTokenizer()
    dm = ChatbotDataModule(batch_size=32, tokenizer=tokenizer)
    dm.prepare_data()
    dm.setup(stage='fit')
    loader = dm.train_dataloader()
    batch = next(iter(loader))
    assert batch['enc_x'].shape == (32, 128)
    assert batch['enc_x_padding_mask'].shape == (32, 128)
    assert batch['dec_x'].shape == (32, 128)
    assert batch['dec_x_padding_mask'].shape == (32, 128)
    assert batch['y'].shape == (32, 128)