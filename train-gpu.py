import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import RichModelSummary, ModelCheckpoint
from tokenizer import WordPieceTokenizer
from data import ChatbotDataModule
from model import LitTransformer
import wandb

if __name__ == '__main__':
    pl.seed_everything(42)
    tokenizer = WordPieceTokenizer()
    dm = ChatbotDataModule(batch_size=32, max_length=64, is_gpu=True, tokenizer=tokenizer)
    model = LitTransformer(
        tokenizer=tokenizer,
        vocab_size=2**14, # 16384
        d_embed=256,
        d_model=256,
        num_layers=2,
        num_heads=8,
        d_ff=512,
        dropout_rate=0.1,
        lr=0.0014,
        num_warmup_steps=4000
    )
    trainer = pl.Trainer(
        accelerator='gpu',
        max_epochs=50,
        check_val_every_n_epoch=5,
        precision='16-mixed',
        callbacks=[
            RichModelSummary(), 
            ModelCheckpoint(save_top_k=-1, every_n_epochs=5, filename='{epoch}')
        ],
        logger=WandbLogger(project='transformer_from_scratch', offline=True)
    )
    trainer.fit(model, dm)

