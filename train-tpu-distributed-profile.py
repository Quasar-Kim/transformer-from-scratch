import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import RichModelSummary, RichProgressBar, ModelCheckpoint
from tokenizer import WordPieceTokenizer
from data import ChatbotDataModule
from model import LitTransformer

# tested on TPU VM v3-8 (kaggle)
# 1 epoch = 13s

if __name__ == '__main__':
    pl.seed_everything(42)
    tokenizer = WordPieceTokenizer()
    dm = ChatbotDataModule(batch_size=128, max_length=128, tokenizer=tokenizer)
    model = LitTransformer(
        tokenizer=tokenizer,
        vocab_size=2**14, # 16384
        d_embed=256, # 256
        d_model=256, # 256
        num_layers=2,
        num_heads=8, # 8
        d_ff=512, # 512
        dropout_rate=0.1,
        lr=0.0014,
        num_warmup_steps=4000
    )
    trainer = pl.Trainer(
        accelerator='tpu',
        devices=8,
        max_epochs=50,
        check_val_every_n_epoch=5,
        precision='bf16-mixed',
        callbacks=[
            RichModelSummary(),
            RichProgressBar(),
            ModelCheckpoint(save_top_k=-1, every_n_epochs=5, filename='{epoch}')
        ],
        # BUG: wandb offline 모드를 사용할 경우 timeout 오류 발생
        # 패치가 릴리스될때까지는 offline 모드 사용 불가
        # https://github.com/wandb/wandb/issues/5071
        logger=WandbLogger(project='transformer_from_scratch', log_model=True)
    )
    trainer.fit(model, dm)