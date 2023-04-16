from contextlib import contextmanager
import torch
import torch.nn.functional as F
from torch import optim
import lightning.pytorch as pl
from transformers import get_inverse_sqrt_schedule
from modules import Transformer
from torchmetrics.classification import MulticlassAccuracy

class LitTransformer(pl.LightningModule):
    def __init__(self, *, tokenizer, lr, num_warmup_steps, **kwargs):
        super().__init__()
        self.save_hyperparameters(ignore=['tokenizer'])
        self.tokenizer = tokenizer
        self.model = Transformer(**kwargs)
        self.train_losses = []
        self.train_acc = MulticlassAccuracy(num_classes=self.hparams.vocab_size)
        self.validation_losses = []
        self.val_acc = MulticlassAccuracy(num_classes=self.hparams.vocab_size)
    
    def forward(self, enc_x, dec_x, enc_x_padding_mask=None, dec_x_padding_mask=None):
        return self.model(enc_x, dec_x, enc_x_padding_mask, dec_x_padding_mask)
    
    def training_step(self, batch, batch_idx):
        y_true = batch.pop('y')
        logits = self(**batch)
        loss = self.loss(logits, y_true)
        self.train_losses.append(loss)
        y_preds = logits.argmax(dim=2)
        self.train_acc(y_preds, y_true)
        return {'loss': loss}
    
    def validation_step(self, batch, batch_idx):
        y_true = batch.pop('y')
        logits = self(**batch)
        loss = self.loss(logits, y_true)
        self.validation_losses.append(loss)
        y_preds = logits.argmax(dim=2)
        self.val_acc(y_preds, y_true)
        return {'loss': loss}
    
    @contextmanager
    def predict_mode(self):
        training = self.training
        grad_enabled = torch.is_grad_enabled()
        try:
            self.eval()
            torch.set_grad_enabled(False)
            yield
        finally:
            self.train(training)
            torch.set_grad_enabled(grad_enabled)
    
    def ask_question(self, inp):
        # inp['dec_x']: (1, N)
        with self.predict_mode():
            while True:
                logits = self(**inp) # (1, N, vocab_size)
                pred_id = logits[0, -1].argmax()
                inp['dec_x'] = torch.cat((inp['dec_x'], pred_id.unsqueeze(0).unsqueeze(0)), dim=1)
                answer = self.tokenizer.decode(inp['dec_x'].squeeze()[1:])
                if pred_id.item() == self.tokenizer.eos_token_id:
                    print('got eos token')
                    break
                if inp['dec_x'].size(1) > 30:
                    print('max length exceeded')
                    break
        return answer

    def loss(self, logits, y_true):
        # y_true: (B, N)
        # y_pred: (B, N, vocab_size)
        # B: batch size, N: target sequence size
        # F.cross_entropy expects second dimension to be classes
        logits = logits.permute(0, 2, 1) # (B, vocab_size, N)
        loss = F.cross_entropy(logits, y_true, ignore_index=self.tokenizer.pad_token_id)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr, betas=(0.9, 0.98), eps=1e-9)
        lr_scheduler = get_inverse_sqrt_schedule(optimizer, num_warmup_steps=self.hparams.num_warmup_steps)
        return { 'optimizer': optimizer, 'lr_scheduler': lr_scheduler }
        # return optimizer
    
    def on_train_batch_end(self, outputs, batch, batch_idx):
        pass

    def on_train_epoch_end(self, *args):
        loss = torch.stack(self.train_losses).mean()
        self.train_losses.clear()
        # loss -> mean of local losses
        # sync_dist=True -> will reduce metrics across processes (as specified by reduce_fx, which is by default torch.mean())
        self.log('train_loss', loss, on_epoch=True, sync_dist=True, prog_bar=True)
        self.log('train_acc', self.train_acc, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self, *args):
        loss = torch.stack(self.validation_losses).mean()
        self.validation_losses.clear()
        self.log('val_loss', loss, on_epoch=True, sync_dist=True)
        self.log('val_acc', self.val_acc, on_epoch=True)

    