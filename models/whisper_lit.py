import lightning as pl
from torchmetrics import MeanMetric
import torch
import wandb
import string 
import torchmetrics
from collections import OrderedDict
from models.ingan_unet import Unet
from whisper import whisper
import evaluate
from transformers import (
    AdamW,
    get_linear_schedule_with_warmup
)
from torch.optim.lr_scheduler import _LRScheduler
from dataloader.dataset import AudioDataset, WhisperDataCollatorWithPadding
import torch.nn.functional as F

def remove_punctuation(text):
    text = text.replace('<|endoftext|>', '')
    return ''.join([char for char in text if char not in string.punctuation])

class GradientNormTracker(pl.Callback):
    def on_after_backward(self, trainer, pl_module):
        for name, parameter in pl_module.named_parameters():
            if parameter.requires_grad and parameter.grad is not None:
                grad_norm = parameter.grad.norm()
                trainer.logger.log_metrics({f'grad_norm/{name}': grad_norm}, step=trainer.global_step)

class GradualWarmupScheduler(_LRScheduler):
    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        self.multiplier = multiplier
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super().__init__(optimizer)

    def step(self, epoch=None, metrics=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        if epoch <= self.total_epoch:
            lr = self.base_lrs[0] * self.multiplier * epoch / self.total_epoch
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        else:
            if not self.finished:
                self.finished = True
                if self.after_scheduler is not None:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.optimizer.param_groups]
            if self.after_scheduler is not None:
                if epoch == self.total_epoch + 1:
                    self.after_scheduler.step(epoch - 1)
                self.after_scheduler.step(epoch)
                
        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]

class LitWhisperModel(pl.LightningModule):
    def __init__(self, config, train_dataset=[], eval_dataset=[]):
        super().__init__()
    
        self.config = config
        # self.whisper = whisper.load_model(f'pretrained_models/whisper/{config["whisper_name"]}.pt')
        self.loss_update_epoch = config["loss_update_epoch"]
        self.save_hyperparameters(config)

        self.options = whisper.DecodingOptions(language=config['lang'], without_timestamps=True)
        self.whisper = whisper.load_model(f'pretrained_models/whisper/{config["whisper_name"]}.pt')
        self.tokenizer = whisper.tokenizer.get_tokenizer(True, language="en", task=self.options.task)
        if config['decoder_only']:
            print("decoder only")
            # only decoder training
            for p in self.whisper.encoder.parameters():
                p.requires_grad = False
        if config['encoder_only']:
            print("encoder only")
            # only encoder training
            for p in self.whisper.decoder.parameters():
                p.requires_grad = False

        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)
        self.metrics_wer = evaluate.load("wer")
        self.metrics_cer = evaluate.load("cer")
        self.metrics_torch_wer = torchmetrics.WordErrorRate()
        self.val_wer = MeanMetric()
        self.val_wer_no_teacher_forcing = MeanMetric()
        self.val_orig_wer = MeanMetric()
        self.val_generate_wer = MeanMetric()
        self.__train_dataset = train_dataset
        self.__eval_dataset = eval_dataset


    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        # logits = self.unet(x)
        whisper_pred = self.whisper(x)
        return x, whisper_pred

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        mel = batch['mels']
        clean_mels = batch['clean_mels']
        labels = batch["labels"].long()
        multilingual_tokens = batch["multilingual_tokens"].long()
        # with torch.no_grad():
        audio_features = self.whisper.encoder(mel.squeeze())
        out = self.whisper.decoder(multilingual_tokens, audio_features)
        ce_loss = self.loss_fn(out.view(-1, out.size(-1)), labels.view(-1))
        self.log("train/ce_loss", ce_loss, on_step=True, prog_bar=True, logger=True)
        loss = ce_loss
        return {"loss":loss, "ce_loss": ce_loss}

    def on_train_epoch_end(self):
        # log epoch metric
        return
        
    def validation_step(self, batch, batch_idx):
        return self.eval_step(batch, "val")
    
    def test_step(self, batch, batch_idx):
        return self.eval_step(batch, "test")

    def eval_step(self, batch, step_type):
        mel = batch['mels']
        clean_mels = batch['clean_mels']
        labels = batch["labels"].long()
        multilingual_tokens = batch["multilingual_tokens"].long()
        audio_features = self.whisper.encoder(mel.squeeze())
        out = self.whisper.decoder(multilingual_tokens, audio_features)

        ce_loss = self.loss_fn(out.view(-1, out.size(-1)), labels.view(-1))
        loss = ce_loss
        out[out == -100] = self.tokenizer.eot
        labels[labels == -100] = self.tokenizer.eot

        o_list, l_list = [], []
        for o, l in zip(out, labels):
            o = torch.argmax(o, dim=1)
            o_list.append(self.tokenizer.decode(o).lower())
            l_list.append(self.tokenizer.decode(l).lower().replace('<|endoftext|>', '').replace("<|en|>", "").replace("<|transcribe|>", "").replace("<|notimestamps|>", ""))
        cer = self.metrics_cer.compute(references=l_list, predictions=o_list)
        wer = self.metrics_wer.compute(references=l_list, predictions=o_list)
        self.val_wer(wer)

        self.log("val/ce_loss", ce_loss, on_step=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("val/cer", cer, on_step=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("val/wer", wer, on_step=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("val/loss", loss, on_step=True, prog_bar=True, logger=True, sync_dist=True)
        
        ### Calculate generate
        options = whisper.DecodingOptions(language='en', without_timestamps=True)
        results = whisper.decode(self.whisper, audio_features, options)    
        r_list = []
        if type(results) == list:
            for res in results:
                r_list.append(remove_punctuation(res.text.lower()))
        else:            
            r_list.append(remove_punctuation(results.text.lower()))
        ntf_wer = self.metrics_wer.compute(references=l_list, predictions=r_list)
        self.val_generate_wer(ntf_wer)
        self.log("val/wer_orig", ntf_wer, on_step=True, prog_bar=True, logger=True, sync_dist=True)
        return {
            "loss": loss,
            "cer": cer,
            "wer": wer,
            "ce_loss": ce_loss
        }
        # return OrderedDict({f'{step_type}_loss': loss.mean()})
   
    def on_validation_epoch_end(self):
        self.log("val/wer", self.val_wer.compute(), on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        #elf.log("val/orig_wer", self.val_orig_wer.compute(), on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("val/generate_wer", self.val_generate_wer.compute(), on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.val_wer.reset()
        self.val_generate_wer.reset()
        return

    def configure_optimizers(self):
        parameters = filter(lambda p: p.requires_grad, self.parameters())
        optimizer = torch.optim.Adam(parameters, lr=self.config["lr"])
        #optimizer = torch.optim.Adam(self.parameters(), lr=self.config["lr"])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,factor=self.config["factor"], patience=self.config["lr_patience"])
        return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": self.config["early_stop_metric"],
                },
                }
    
    def setup(self, stage=None):

        if stage == 'fit' or stage is None:
            self.t_total = (
                (len(self.__train_dataset) // (self.config["batch_size"]))
                // self.config["gradient_accumulation_steps"]
                * float(self.config["epochs"])
            )
    
    def train_dataloader(self):
        dataset = AudioDataset(self.__train_dataset, self.tokenizer, self.config["sample_rate"], 
                                self.config["supported_snrs"], self.config["enhancement_loss"], train=True)
        return torch.utils.data.DataLoader(dataset, 
                          batch_size=self.config["batch_size"], 
                          drop_last=True, shuffle=True, num_workers=self.config["dl_num_workers"],
                          collate_fn=WhisperDataCollatorWithPadding()
                          )

    def val_dataloader(self):
        dataset = AudioDataset(self.__eval_dataset, self.tokenizer, self.config["sample_rate"], 
                                self.config["supported_snrs"], self.config["enhancement_loss"])
        return torch.utils.data.DataLoader(dataset, 
                          batch_size=self.config["batch_size"], 
                          num_workers=self.config["dl_num_workers"],
                          collate_fn=WhisperDataCollatorWithPadding()
                          )
    
    def test_dataloader(self):
        dataset = AudioDataset(self.__eval_dataset, self.tokenizer, self.config["sample_rate"], 
                                self.config["supported_snrs"], self.config["enhancement_loss"])
        return torch.utils.data.DataLoader(dataset, 
                          batch_size=self.config["batch_size"], 
                          num_workers=self.config["dl_num_workers"],
                          collate_fn=WhisperDataCollatorWithPadding()
                          )