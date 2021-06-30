import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from modules.cwgangp import CWGANGP
from config import Config, GeneratorConfig, DiscriminatorConfig
from utils.callbacks import ConditionalWandbPCLogCallback
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('-gpu', '--gpu', dest='gpu', default=0, type=int)
parser.add_argument('-data_save_path', '--data_save_path', dest='data_save_path', type=str, default='.')
parser.add_argument('-batch_size', '--batch_size', dest='batch_size', default=8, type=int)
parser.add_argument('-d_lr', '--d_lr', dest='d_lr', default=1e-4, type=float)
parser.add_argument('-g_lr', '--g_lr', dest='g_lr', default=1e-4, type=float)
parser.add_argument('-d_multistep_milestones', '--d_multistep_milestones', dest='d_multistep_milestones',
                    nargs='+', default=[int(1e9)], type=int)
parser.add_argument('-g_multistep_milestones', '--g_multistep_milestones', dest='g_multistep_milestones',
                    nargs='+', default=[int(1e9)], type=int)
parser.add_argument('-d_freq', '--d_freq', dest='d_freq', default=5, type=int)
parser.add_argument('-lr_decay', '--lr_decay', dest='lr_decay', default=0.5, type=float)
parser.add_argument('-ckpt_save_path', '--ckpt_save_path', dest='ckpt_save_path', type=str, default='.')
parser.add_argument('-n_points', '--n_points', dest='n_points', type=int, default=1024)
parser.add_argument('-from_ckpt', '--from_ckpt', dest='from_ckpt', type=str, default=None)
args = parser.parse_args()


if __name__ == '__main__':
    config = Config(
        data_path=args.data_save_path,
        generator=GeneratorConfig(
            lr=args.g_lr, lr_decay=args.lr_decay, multistep_milestones=args.g_multistep_milestones
        ),
        discriminator=DiscriminatorConfig(
            lr=args.d_lr, lr_decay=args.lr_decay, multistep_milestones=args.d_multistep_milestones, freq=args.d_freq
        ),
        batch_size=args.batch_size,
        n_points=args.n_points,
    )
    print(config)
    gan = CWGANGP(config)
    if args.from_ckpt is not None:
        gan = gan.load_from_checkpoint(args.from_ckpt)
    wand_cb = ConditionalWandbPCLogCallback(3)
    logger = WandbLogger(name='PCGAN', project='cwgangp')
    ckpt = pl.callbacks.ModelCheckpoint(args.ckpt_save_path, save_top_k=-1, every_n_val_epochs=1)
    trainer = pl.Trainer(
        gpus=[1],
        callbacks=[wand_cb, ckpt],
        logger=logger,
        max_epochs=2000,
    )
    trainer.fit(gan)
