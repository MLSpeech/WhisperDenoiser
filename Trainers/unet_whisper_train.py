import argparse
import random
import time
import os
import inspect
import sys
from os import mkdir
from os.path import exists, join
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
import os
from lightning.pytorch.loggers import WandbLogger, TensorBoardLogger, CSVLogger
from torch.backends import cudnn
from lightning.pytorch.strategies import DDPStrategy

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

from dataloader.dataset import AudioDataset
from models.unet_whisper_lit import LitUnetWhisperModel, GradientNormTracker
from models.whisper_lit import LitWhisperModel

def main(hparams):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    cudnn.benchmark = True
    device ='cpu'
    if args.cuda:
        device = 'cuda'
    loggers_list = []  
    all_tags = ["white", "white_dbg"] + hparams.tags 
    if hparams.wandb:
        if hparams.wandb_id is None:
            resume = "allow"
        else:
            resume = "must"
        wandb_logger = WandbLogger(project="unet_pretrain", group="all_noises", id=hparams.wandb_id, tags=all_tags, resume=resume)
        
        loggers_list.append(wandb_logger)
    elif hparams.tensorboard:
        logger_dir = os.path.join(os.path.join("tensorboard"), hparams.exp_name)
        if not os.path.exists(logger_dir):
            os.makedirs(logger_dir)
        tensorboard_logger = TensorBoardLogger(name=hparams.exp_name, save_dir=os.path.join("tensorboard"))
        loggers_list.append(tensorboard_logger)
    else:
        csv_logger = CSVLogger("logs", name="unet_pretrain")
        loggers_list.append(csv_logger)

    hparams.run_dir = join(hparams.run_dir, hparams.exp_name)
    model_save_path = join(hparams.run_dir, "ckpt")


    hparams_dict = vars(hparams)
    if hparams.finetune:
        print("Fine tuning")
        solver = LitWhisperModel(hparams_dict, hparams.train_data, hparams.val_data).to(device)
    else:
        print("Train Unet")
        solver = LitUnetWhisperModel(hparams_dict, hparams.train_data, hparams.val_data).to(device)
    if hparams.load_ckpt:
        checkpoint = torch.load(hparams.load_ckpt, map_location=device)
        solver.load_state_dict(checkpoint['state_dict'])

    for name, param in solver.named_parameters():
        if param.is_sparse:
            print(f"Sparse parameter found: {name}")

    for name, buffer in solver.named_buffers():
        if buffer.is_sparse:
            print(f"Sparse buffer found: {name}")

    if hasattr(solver.whisper, 'alignment_heads') and solver.whisper.alignment_heads.is_sparse:
        # Convert the sparse buffer to a dense buffer
        solver.whisper.alignment_heads = solver.whisper.alignment_heads.to_dense()
        
    early_stop = EarlyStopping(
        monitor=hparams.early_stop_metric,
        patience=hparams.patience,
        verbose=True,
        mode=hparams.early_stop_mode
    )
    checkpoint = ModelCheckpoint(
        dirpath=model_save_path,
        save_top_k=20,
        save_last=True,
        verbose=True,
        monitor=hparams.early_stop_metric,
        mode=hparams.early_stop_mode,
        every_n_epochs=1,
        every_n_train_steps=0,
    )
    checkpoint.CHECKPOINT_NAME_LAST = "{epoch}-last"
    lr_monitor = LearningRateMonitor(logging_interval='step')

    if hparams.load_ckpt: #and hparams.wandb_id:
        min_epochs = 10 #int(hparams.load_ckpt.split("/")[-1].split("=")[1].split("-")[0])
        ckpt_path = hparams.load_ckpt
        print("restore")
    else:
        min_epochs = 10
        ckpt_path = None
        print("new training")

    trainer = Trainer(
            logger=loggers_list,
            precision=16,
            check_val_every_n_epoch=1,
            val_check_interval=0.5,
            min_epochs=min_epochs,
            max_epochs=hparams.epochs,
            callbacks = [checkpoint, early_stop, lr_monitor], #, GradientNormTracker()
            accumulate_grad_batches=hparams.gradient_accumulation_steps,
            accelerator='gpu',
            devices=hparams.gpus,
            # strategy="ddp"
            )
    try:
        loggers_list[0]._experiment.log_code(".", include_fn=lambda path: path.endswith(".py"))
    except:
        print("")
    if not hparams.test:
        # import ipdb; ipdb.set_trace()
        trainer.fit(model=solver, ckpt_path=ckpt_path)
        trainer.test(model=solver, ckpt_path="last")
        solver.unet.save_model(model_path=checkpoint.best_model_path.replace(".ckpt", "best_model_unet.ckpt"))
    else:
        trainer.test(model=solver)

def parse_args():
    parser = argparse.ArgumentParser(description='train unet')
    parser.add_argument('--train-data', default="/home/mlspeech/shua/home/Shua/recipies/Whisper_denoiser/Data/white_pub_noise_librispeech_2468Q_train.csv", help='train dataset json')
    parser.add_argument('--val-data', default="/home/mlspeech/shua/home/Shua/recipies/Whisper_denoiser/Data/L1_by_session_speaker.csv", help='val dataset json')
    parser.add_argument('--pub_val', default="/home/mlspeech/shua/home/Shua/recipies/Whisper_denoiser/Data/pub_noise_librispeech_2468Q_test.csv", help='val dataset json')
    # parser.add_argument('--white_val', default="/home/mlspeech/shua/home/Shua/recipies/Whisper_denoiser/Data/random_noise_librispeech_2468Q_test.csv", help='val dataset json')
    parser.add_argument('--test-data', default="/home/mlspeech/shua/home/Shua/recipies/Whisper_denoiser/Data/L1_by_session_speaker.csv", help='test dataset json')
    parser.add_argument('--db_name', type=str, default='Libri', help='db name')
    parser.add_argument('--run_dir', type=str, default='.exp/', help='directory for saving run outputs (logs, ckpt, etc.)')
    parser.add_argument('--exp_name', type=str, default='multi_noise', help='experiment name')
    parser.add_argument('--whisper_name', type=str, default='base', help='experiment name')
    parser.add_argument('--lang', type=str, default='en', help='lang')
    parser.add_argument('--load_ckpt', type=str, default=None, help='path to a pre-trained model, if provided, training will resume from that point')
    # parser.add_argument('--load_ckpt', type=str, default=".exp/libri_enhancement_loss/ckpt/epoch=49-step=45875[0.ckpt", help='path to a pre-trained model, if provided, training will resume from that point')
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--test', default=False, action='store_true', help='flag to indicate to run a test epoch only (not training will take place)')
    parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate')
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--weight_decay', type=float, default=0.85 , help='weight_decay')
    parser.add_argument('--min_lr', type=float, default=0.0000001 , help='minimum lr')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--dl_num_workers', type=int, default=10, help='dl_num_workers')
    parser.add_argument('--epochs', type=int, default=200, help='upper epoch limit')
    parser.add_argument('--warmup_epoch', type=int, default=1, help='warmup_epoch')
    parser.add_argument('--batch_size', type=int, default=16, metavar='N', help='batch size')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, metavar='N', help='batch size')
    parser.add_argument('--val_batch_size', type=int, default=8, metavar='N', help='batch size')
    parser.add_argument('--loss_update_epoch', type=int, default=10, metavar='N', help='Epoch to  switch  from L1 to CE')
    parser.add_argument('--seed', type=int, default=1245, help='random seed')
    parser.add_argument('--patience', type=int, default=100, help='patience for early stopping')
    parser.add_argument('--factor', type=float, default=0.99, help='lr scheduler factor')
    parser.add_argument('--lr_patience', type=int, default=4, help=' lr scheduler patience ')
    parser.add_argument('--early_stop_metric', type=str, default="val/generate_wer", help=' metric to track ')
    parser.add_argument('--early_stop_mode', type=str, default="min", help=' metric mode to track ')
    parser.add_argument('--input_size', type=int, default=6400, help='number of inputs')
    parser.add_argument('--sample_rate', type=int, default=16000, help='number of inputs')
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout')
    parser.add_argument('--wandb', action='store_true', help='log to wandb')
    parser.add_argument('--tensorboard', action='store_true', help='log to wandb')
    parser.add_argument('--wandb_id', type=str, default=None, help='id to continue runnig')
    parser.add_argument('--n_mels', type=int, default=80, help='number of mel filters')
    parser.add_argument('--n_blocks', type=int, default=6, help='number of blocks')
    parser.add_argument('--n_downsampling', type=int, default=3, help='number of downsampling operations')
    parser.add_argument('--use_bias', type=bool, default=True, help='whether to use bias in layers')
    parser.add_argument('--skip_flag', type=bool, default=True, help='flag for using skip connections')
    parser.add_argument('--supported_snrs', nargs='*', type=str, default=['Q', '8', '6', '4', '2'], help='List of supported SNRs')
    parser.add_argument('--enhancement_loss',  action='store_true', help='use enhancement loss')
    parser.add_argument('--ce_weight', type=float, default=0.9, help='Weight of cross entropy loss')
    parser.add_argument('--enhancement_weight', type=float, default=0.1, help='weight of l1 enhancement loss')
    parser.add_argument('--decoder_only',  action='store_true', help='use enhancement loss')
    parser.add_argument('--encoder_only',  action='store_true', help='use enhancement loss')
    parser.add_argument('--finetune',  action='store_true', help='use enhancement loss')
    parser.add_argument('--packet-loss',  action='store_true', help='simulate packet loss')
    parser.add_argument('--clipping',  action='store_true', help='simulate saturation')
    parser.add_argument('--reverb',  action='store_true', help='simulate reverberation')
    parser.add_argument('--white',  action='store_true', help='simulate white noise')
    parser.add_argument('--pub',  action='store_true', help='simulate pub noise')
    parser.add_argument('--tags', nargs='+', default=[], help='Additional tags for the Wandb logging')



    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    args.cuda = torch.cuda.is_available()
    main(args)