import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from modules.cwgangp import CWGANGP
from config import Config, GeneratorConfig, DiscriminatorConfig
from utils.callbacks import ConditionalWandbPCLogCallback
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('-gpu', '--gpu', dest='gpu', default=0, type=int)
parser.add_argument('-batch_size', '--batch_size', dest='batch_size', default=16, type=int)
parser.add_argument('-d_lr', '--d_lr', dest='d_lr', default=1e-5, type=float)
parser.add_argument('-g_lr', '--g_lr', dest='g_lr', default=1e-5, type=float)
parser.add_argument('-ckpt_save_path', '--ckpt_save_path', dest='ckpt_save_path', type=str, default='.')
parser.add_argument('-n_points', '--n_points', dest='n_points', type=int, default=1024)
args = parser.parse_args()


if __name__ == '__main__':
    config = Config(generator=GeneratorConfig(lr=args.g_lr),
                    discriminator=DiscriminatorConfig(lr=args.d_lr),
                    batch_size=args.batch_size,
                    n_points=args.n_points)
    gan = CWGANGP(config)
    wand_cb = ConditionalWandbPCLogCallback(3)
    logger = WandbLogger()
    ckpt = pl.callbacks.ModelCheckpoint(args.ckpt_save_path, save_top_k=-1, every_n_train_steps=20000)
    trainer = pl.Trainer(
        gpus=[1],
        callbacks=[wand_cb, ckpt],
        logger=logger,
        max_epochs=2000,
    )
    trainer.fit(gan)
