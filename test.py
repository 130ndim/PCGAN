import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from modules.gan import GAN
from utils.callbacks import WandbPCLogCallback


if __name__ == '__main__':
    gan = GAN()
    wand_cb = WandbPCLogCallback(8)
    logger = WandbLogger()
    ckpt = pl.callbacks.ModelCheckpoint('/mnt/tank/scratch/dleonov/models/pcgan_toilet')
    trainer = pl.Trainer(
        gpus=[1],
        callbacks=[wand_cb, ckpt],
        logger=logger,
        max_epochs=1500,
        reload_dataloaders_every_epoch=True,
    )
    trainer.fit(gan)

    GAN.load_from_checkpoint()