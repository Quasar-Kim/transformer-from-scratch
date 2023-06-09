import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import RichModelSummary, ModelCheckpoint, LearningRateMonitor
from tokenizer import WordPieceTokenizer
from data import ChatbotDataModule
from model import LitTransformer

if __name__ == '__main__':
    pl.seed_everything(42)
    tokenizer = WordPieceTokenizer()
    dm = ChatbotDataModule(batch_size=128, max_length=128, tokenizer=tokenizer)
    model = LitTransformer(
        tokenizer=tokenizer,
        vocab_size=2**14, # 16384
        d_embed=256,
        d_model=256,
        num_layers=2,
        num_heads=8,
        d_ff=512,
        dropout_rate=0.1,
        lr=1e-5,
        num_warmup_steps=100
    )
    trainer = pl.Trainer(
        accelerator='tpu',
        devices=8,
        max_epochs=150,
        check_val_every_n_epoch=5,
        precision='bf16-mixed',
        callbacks=[
            # BUG: RichProgressbar breaks in kaggle
            RichModelSummary(),
            ModelCheckpoint(save_top_k=3, every_n_epochs=5, filename='{epoch}-{val_loss:.2f}', monitor='val_loss'),
            LearningRateMonitor(logging_interval='epoch')
        ],
        # BUG: wandb offline 모드를 사용할 경우 timeout 오류 발생
        # 패치가 릴리스될때까지는 offline 모드 사용 불가
        # https://github.com/wandb/wandb/issues/5071
        logger=WandbLogger(project='transformer_from_scratch', log_model=True)
    )
    trainer.fit(model, dm)