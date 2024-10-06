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
from utils.metrics import WERCalculator

torch.set_float32_matmul_precision('high') 
def remove_punctuation(text):
    text = text.replace('<|endoftext|>', '')
    return ''.join([char for char in text if char not in string.punctuation])

class GradientNormTracker(pl.Callback):
    def on_after_backward(self, trainer, pl_module):
        for name, parameter in pl_module.named_parameters():
            if parameter.requires_grad and parameter.grad is not None:
                if "final_block" in name:
                    gradients = parameter.grad
                    # Compute L2 Norm, Mean (average), and Standard Deviation
                    l2_norm = torch.norm(gradients, p=2)
                    mean = torch.mean(gradients)
                    std = torch.std(gradients)
                    trainer.logger.log_metrics({f'grad_norm/{name}': l2_norm}, step=trainer.global_step)
                    trainer.logger.log_metrics({f'grad_mean/{name}': mean}, step=trainer.global_step)
                    trainer.logger.log_metrics({f'grad_std/{name}': std}, step=trainer.global_step)

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

class LitUnetWhisperModel(pl.LightningModule):
    def __init__(self, config, train_dataset=[], eval_dataset=[]):
        super().__init__()
    
        self.config = config
        self.unet = Unet(config["n_mels"], config["n_blocks"], config["n_downsampling"], config["use_bias"], config["skip_flag"])
        # self.whisper = whisper.load_model(f'pretrained_models/whisper/{config["whisper_name"]}.pt')
        self.loss_update_epoch = config["loss_update_epoch"]
        self.train_l1_loss_tracker = MeanMetric()
        self.val_l1_loss_tracker = MeanMetric()
        self.l1loss = torch.nn.L1Loss()
        self.save_hyperparameters(config)

        self.options = whisper.DecodingOptions(language=config['lang'], without_timestamps=True)
        self.whisper = whisper.load_model(f'pretrained_models/whisper/{config["whisper_name"]}.pt')
        self.tokenizer = whisper.tokenizer.get_tokenizer(True, language="en", task=self.options.task)

        # # only decoder training
        # for p in self.whisper.encoder.parameters():
        #     p.requires_grad = False

        # # only encoder training
        # for p in self.whisper.decoder.parameters():
        #     p.requires_grad = False
        
        # Only Unet training
        for param in self.whisper.parameters():
            param.requires_grad = False

        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)
        self.metrics_wer = evaluate.load("wer")
        self.metrics_cer = evaluate.load("cer")
        self.metrics_torch_wer = torchmetrics.WordErrorRate()
        
        self.val_wer = MeanMetric()
        self.val_wer_no_teacher_forcing = MeanMetric()
        self.val_orig_wer = MeanMetric()
        self.val_generate_wer = MeanMetric()
        self.pub_val_orig_wer = MeanMetric()
        self.pub_val_generate_wer = MeanMetric()
        # self.white_val_orig_wer = MeanMetric()
        # self.white_val_generate_wer = MeanMetric()
        self.__train_dataset = train_dataset
        self.__eval_dataset = eval_dataset
        self.pub_val = config["pub_val"]
        # self.white_val = config["white_val"]
        self.l1_weight = config["enhancement_weight"]
        self.ce_weight = config["ce_weight"]
        self.current_step = 0 
        self.weight_decay = config["weight_decay"]
        self.min_lr = 0.0000001 #config["min_lr"]
        self.pretrain_epoch = config["warmup_epoch"]
        self.best_wer = float('inf')
        # self.dbg = False
        self.val_wer_j = WERCalculator()
        self.val_orig_wer_j = WERCalculator()


    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        logits = self.unet(x)
        whisper_pred = self.whisper(logits)
        return logits, whisper_pred

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        mel = batch['mels']
        clean_mels = batch['clean_mels']
        labels = batch["labels"].long()
        multilingual_tokens = batch["multilingual_tokens"].long()
        logits = self.unet(mel)
        
        l1loss = self.l1loss(logits, clean_mels)
        self.train_l1_loss_tracker(l1loss)
        # with torch.no_grad():
        audio_features = self.whisper.encoder(logits.squeeze())
        # out = self.whisper.decoder(multilingual_tokens, audio_features)
        # audio_features = self.whisper.encoder(mel.squeeze())
        out = self.whisper.decoder(multilingual_tokens, audio_features)
        ce_loss = self.loss_fn(out.view(-1, out.size(-1)), labels.view(-1))
        self.log("train/ce_loss", ce_loss, on_step=True, prog_bar=True, logger=True)
        self.log("train/l1_loss", l1loss, on_step=True, prog_bar=True, logger=True)
        
        if self.trainer.current_epoch < self.pretrain_epoch:
            loss = l1loss
        elif self.trainer.current_epoch <= self.loss_update_epoch:
            loss = self.ce_weight * ce_loss + self.l1_weight * l1loss #+ 
        else:
            loss = self.ce_weight * ce_loss + (self.l1_weight * l1loss / self.trainer.current_epoch)
        # loss = 0.5 + self.l1_weight * l1loss
        return {"loss":loss, "l1loss": loss.mean(), "ce_loss": ce_loss}

    def on_train_epoch_end(self):
        # log epoch metric
        self.log("train_l1_loss", self.train_l1_loss_tracker.compute(), on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.train_l1_loss_tracker.reset()
        return
        
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        return self.eval_step(batch, "val")
    
    def test_step(self, batch, batch_idx, dataloader_idx=0):
        return self.eval_step(batch, "test")

    def eval_step(self, batch, step_type, dataloader_idx=0):
        mel = batch['mels']
        clean_mels = batch['clean_mels']
        labels = batch["labels"].long()
        multilingual_tokens = batch["multilingual_tokens"].long()
        logits = self.unet(mel)
        l1loss = self.l1loss(logits, clean_mels)
        self.val_l1_loss_tracker(l1loss)
        
        # print(mel.shape, logits.shape, logits.squeeze().shape)
        audio_features = self.whisper.encoder(logits.squeeze(dim=1))

        if dataloader_idx == 0:
            # audio_features = self.whisper.encoder(mel.squeeze())
            out = self.whisper.decoder(multilingual_tokens, audio_features)

            ce_loss = self.loss_fn(out.view(-1, out.size(-1)), labels.view(-1))
            
            if self.trainer.current_epoch <= self.loss_update_epoch:
                loss = self.ce_weight * ce_loss + self.l1_weight * l1loss #+ 
            else:
                loss =  ce_loss + (l1loss * 0.0)
            
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
        options = whisper.DecodingOptions(language='en', without_timestamps=True)#, beam_size=5)  
        results = whisper.decode(self.whisper, logits.squeeze(dim=1), options)
        r_list = []
        # print(type(results))
        if type(results) == list:
            for res in results:
                r_list.append(remove_punctuation(res.text.lower()))
        else:            
            r_list.append(remove_punctuation(results.text.lower()))
        generate_wer = self.metrics_wer.compute(references=l_list, predictions=r_list)
        self.val_wer_j.update(references=l_list, predictions=r_list)

        if dataloader_idx == 0:
            # self.log("val/wer_generate", generate_wer, on_step=True, prog_bar=True, logger=True, sync_dist=True)
            self.val_generate_wer(generate_wer)
        elif dataloader_idx == 1:
            # self.log("pub_val/wer_generate", generate_wer, on_step=True, prog_bar=True, logger=True, sync_dist=True)
            self.pub_val_generate_wer(generate_wer)
        # elif dataloader_idx == 2:
        #     # self.log("white_val/wer_generate", generate_wer, on_step=True, prog_bar=True, logger=True, sync_dist=True)
        #     self.white_val_generate_wer(generate_wer)
        ### calculate orig wer
        orig_audio_features = self.whisper.encoder(mel.squeeze(dim=1))
        if self.trainer.current_epoch < 2:
            ### Calculate generate
            results = whisper.decode(self.whisper, orig_audio_features, options)    
            r_list = []
            # print(type(results))
            if type(results) == list:
                for res in results:
                    r_list.append(remove_punctuation(res.text.lower()))
            else:            
                r_list.append(remove_punctuation(results.text.lower()))
            ntf_wer = self.metrics_wer.compute(references=l_list, predictions=r_list)
            self.val_orig_wer_j.update(references=l_list, predictions=r_list)

            if dataloader_idx == 0:
                # self.log("val/wer_orig", ntf_wer, on_step=True, prog_bar=True, logger=True, sync_dist=True)
                self.val_orig_wer(ntf_wer)
            elif dataloader_idx == 1:
                self.log("pub_val/wer_orig", ntf_wer, on_step=True, prog_bar=True, logger=True, sync_dist=True)
                self.pub_val_orig_wer(ntf_wer)
            # elif dataloader_idx == 2:
            #     # self.log("white_val/wer_orig", ntf_wer, on_step=True, prog_bar=True, logger=True, sync_dist=True)
            #     self.white_val_orig_wer(ntf_wer)
        return {
            "loss": loss,
            "cer": cer,
            "wer": wer,
            "ce_loss": ce_loss,
            "l1loss": l1loss.mean()
        }
        # return OrderedDict({f'{step_type}_loss': loss.mean()})
   
    def on_validation_epoch_end(self):
        self.log('val_l1_loss', self.val_l1_loss_tracker.compute(), on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val/wer", self.val_wer.compute(), on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        # self.log("val_wer_no_teacher_forcing", self.val_wer_no_teacher_forcing.compute(), on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        
        metrics = self.val_wer_j.compute()
        self.log("val/wer", metrics['WER'], on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("val/insertions", metrics['Insertions'], on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("val/deletions", metrics['Deletions'], on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("val/substitutions", metrics['Substitutions'], on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.val_wer_j.reset()  # Reset the metrics after logging them
        
        if self.trainer.current_epoch < 2:
            self.log("val/orig_wer", self.val_orig_wer.compute(), on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
            self.val_orig_wer.reset()
             # Log metrics for original features
            orig_metrics = self.val_orig_wer_j.compute()
            self.log("val/orig_wer", orig_metrics['WER'], on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
            self.log("val/orig_insertions", orig_metrics['Insertions'], on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
            self.log("val/orig_deletions", orig_metrics['Deletions'], on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
            self.log("val/orig_substitutions", orig_metrics['Substitutions'], on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
            self.val_orig_wer_j.reset() 
            # self.pub_val_orig_wer.reset()
            # self.white_val_orig_wer.reset()

        self.log("val/generate_wer", self.val_generate_wer.compute(), on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        # globals_.Epoch = self.current_epoch
        self.update_learning_rate(self.val_generate_wer.compute(), self.weight_decay)
        return
    
    def load_unet_state_dict_from_checkpoint(self, checkpoint_path):
        # Load the entire checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
        
        # Assuming state dict keys are prefixed with 'unet.' if they belong to the unet model
        # Adjust this logic if your checkpoint has a different naming convention
        unet_state_dict = {k[len("unet."):]: v for k, v in checkpoint['state_dict'].items() if k.startswith("unet.")}
        
        # Load the filtered state dict into the unet model
        self.unet.load_state_dict(unet_state_dict)

    def update_learning_rate(self, current_wer, decay_factor=0.85):
        # Check if current WER is worse than the best WER observed so far
        if current_wer > self.best_wer:# or self.dbg:
            # Load the best model checkpoint if WER has increased
            checkpoint_path = self.trainer.checkpoint_callback.best_model_path
            if checkpoint_path:
                self.load_unet_state_dict_from_checkpoint(checkpoint_path)

                self.log(f'rollback to {checkpoint_path}', 1.0, on_step=False, on_epoch=True, logger=True, sync_dist=True)
                # self.trainer.model = self.trainer.model.load_from_checkpoint(checkpoint_path)
                # # Optionally, reset any state that is not captured by the checkpoint but affects training
        else:
            # Update the best WER to the current WER if it's an improvement or equal
            self.best_wer = current_wer
            self.log('rollback', 0.0, on_step=False, on_epoch=True, logger=True, sync_dist=True)
        # self.dbg = True
        # self.log("dbg = ", self.dbg)
        # Proceed with learning rate decay
        optimizer = self.trainer.optimizers[0]
        for param_group in optimizer.param_groups:
            if param_group['lr'] > self.min_lr:
                param_group['lr'] *= decay_factor
        new_lr = optimizer.param_groups[0]['lr']
        self.log('learning_rate', new_lr, on_step=False, logger=True, on_epoch=True)

    def configure_optimizers(self):
        parameters = filter(lambda p: p.requires_grad, self.parameters())
        optimizer = torch.optim.Adam(parameters, lr=self.config["lr"])
        #optimizer = torch.optim.Adam(self.parameters(), lr=self.config["lr"])
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,factor=self.config["factor"], patience=self.config["lr_patience"])
        return {
                "optimizer": optimizer#,
                # "lr_scheduler": {
                #     "scheduler": scheduler,
                #     "monitor": self.config["early_stop_metric"],
                # },
                }
    
    def _configure_optimizers(self):
        parameters = filter(lambda p: p.requires_grad, self.parameters())
        optimizer = torch.optim.Adam(parameters, lr=self.config["lr"])
        def custom_lr_lambda(epoch):
            # # Reduce LR by a factor of 10 after self.loss_update_epoch
            # if epoch > self.loss_update_epoch:
            #     return 0.1
            # else:
            #     return 1.0
            lr = 1.0/(epoch+1.0) / 10.0
            return lr

        custom_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=custom_lr_lambda)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": custom_scheduler,
                "monitor": self.config["early_stop_metric"],
                "interval": "epoch",
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
        dataset = AudioDataset(self.__train_dataset, self.tokenizer, self.config["sample_rate"], self.config["supported_snrs"], 
                                self.config["enhancement_loss"], train=True, 
                                packet_loss=self.config["packet_loss"], clipping=self.config["clipping"], reverb=self.config["reverb"])
        return torch.utils.data.DataLoader(dataset, 
                          batch_size=self.config["batch_size"], 
                          drop_last=True, shuffle=True, num_workers=self.config["dl_num_workers"],
                          collate_fn=WhisperDataCollatorWithPadding()
                          )

    def val_dataloader(self):
        dataset = AudioDataset(self.__eval_dataset, self.tokenizer, self.config["sample_rate"], 
                                self.config["supported_snrs"], self.config["enhancement_loss"], packet_loss=self.config["packet_loss"], clipping=self.config["clipping"], reverb=self.config["reverb"], white=self.config["white"], pub=self.config["pub"])
        dataloader = torch.utils.data.DataLoader(dataset, 
                          batch_size=self.config["batch_size"], 
                          num_workers=self.config["dl_num_workers"],
                          collate_fn=WhisperDataCollatorWithPadding()
                          )
        dataset1 = AudioDataset(self.pub_val, self.tokenizer, self.config["sample_rate"], 
                                self.config["supported_snrs"], self.config["enhancement_loss"], packet_loss=self.config["packet_loss"], clipping=self.config["clipping"], reverb=self.config["reverb"], white=self.config["white"], pub=self.config["pub"])
        dataloader1 = torch.utils.data.DataLoader(dataset1, 
                          batch_size=self.config["batch_size"], 
                          num_workers=self.config["dl_num_workers"],
                          collate_fn=WhisperDataCollatorWithPadding()
                          )
        # dataset2 = AudioDataset(self.white_val, self.tokenizer, self.config["sample_rate"], 
        #                         self.config["supported_snrs"], self.config["enhancement_loss"], packet_loss=self.config["packet_loss"], clipping=self.config["clipping"], reverb=self.config["reverb"])
        # dataloader2 = torch.utils.data.DataLoader(dataset2, 
        #                   batch_size=self.config["batch_size"], 
        #                   num_workers=self.config["dl_num_workers"],
        #                   collate_fn=WhisperDataCollatorWithPadding()
        #                   )
        return [dataloader, dataloader1]#, dataloader2]

    def test_dataloader(self):
        dataset = AudioDataset(self.__eval_dataset, self.tokenizer, self.config["sample_rate"], self.config["supported_snrs"], 
        self.config["enhancement_loss"], packet_loss=self.config["packet_loss"], clipping=self.config["clipping"], reverb=self.config["reverb"])
        return torch.utils.data.DataLoader(dataset, 
                          batch_size=self.config["batch_size"], 
                          num_workers=self.config["dl_num_workers"],
                          collate_fn=WhisperDataCollatorWithPadding()
                          )