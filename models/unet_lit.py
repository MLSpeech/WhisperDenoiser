import lightning as pl
from torchmetrics import MeanMetric
import torch
import wandb
from collections import OrderedDict
from models.ingan_unet import Unet

class LitPretrainUnetModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
    
        self.config = config
        self.unet = Unet(config["n_mels"], config["n_blocks"], config["n_downsampling"], config["use_bias"], config["skip_flag"])
        self.train_l1_loss_tracker = MeanMetric()
        self.val_l1_loss_tracker = MeanMetric()
        self.l1loss = torch.nn.L1Loss()
        self.save_hyperparameters(config)

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        logits = self.unet(x)
        return logits

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        mel, gpt2_tokens, multilingual_tokens = batch
        logits = self.unet(mel)
        loss = self.l1loss(logits, mel)
        self.train_l1_loss_tracker(loss)
        return loss.mean()

    def on_train_epoch_end(self):
        # log epoch metric
        self.log("train_l1_loss", self.train_l1_loss_tracker.compute(), on_step=False, on_epoch=True, prog_bar=True)
        self.train_l1_loss_tracker.reset()
        return
        
    def validation_step(self, batch, batch_idx):
        return self.eval_step(batch, "val")
    
    def test_step(self, batch, batch_idx):
        return self.eval_step(batch, "test")

    def eval_step(self, batch, step_type):
        mel, gpt2_tokens, multilingual_tokens = batch
        logits = self.unet(mel)
        loss = self.l1loss(logits, mel)
        self.val_l1_loss_tracker(loss)
        return OrderedDict({f'{step_type}_loss': loss.mean()})
   
    def on_validation_epoch_end(self):
        self.log('val_l1_loss', self.val_l1_loss_tracker.compute(), on_step=False, on_epoch=True, prog_bar=True)
        self.val_l1_loss_tracker.reset()
        # globals_.Epoch = self.current_epoch
        return

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config["lr"])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,factor=self.config["factor"], patience=self.config["lr_patience"])
        return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": self.config["early_stop_metric"],
                },
                }
