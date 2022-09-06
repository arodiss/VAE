import os
from vae import VAE
from dataset import Dataset
from experiment import Experiment
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.plugins import DDPPlugin
import random


NUM_EXPERIMENTS = 20


def run_experiment(latent_params, kld_weight, learning_rate, scheduler_gamma):
    kld_weight = int(kld_weight * 100000) / 100000
    learning_rate = int(learning_rate * 10000) / 10000
    scheduler_gamma = int(scheduler_gamma * 100) / 100
    version_name = f"latent_{latent_params}___kld_{kld_weight}___lr_{learning_rate}___gamma_{scheduler_gamma}"
    tb_logger = TensorBoardLogger(save_dir="tensorboard", version=version_name)
    runner = Trainer(
        logger=tb_logger,
        callbacks=[
            LearningRateMonitor(),
            ModelCheckpoint(
                save_top_k=1,
                dirpath=os.path.join(tb_logger.log_dir, "checkpoints"),
                monitor="val_loss",
                save_last=True
            ),
        ],
        strategy=DDPPlugin(find_unused_parameters=False),
        log_every_n_steps=10,
        max_epochs=200,
    )
    runner.fit(
        Experiment(
            model=VAE(latent_dim=latent_params, interim_dim=8 * latent_params),
            kld_weight=kld_weight,
            learning_rate=learning_rate,
            scheduler_gamma=scheduler_gamma,
        ),
        datamodule=Dataset()
    )


for i in range(NUM_EXPERIMENTS):
    print(f"Running the experiment {i}/{NUM_EXPERIMENTS}...")
    run_experiment(
        latent_params=random.randint(6, 30),
        kld_weight=0.0002 + 0.0001 * random.random(),
        learning_rate=0.0003 + 0.0003 * random.random(),
        scheduler_gamma=0.99 - 0.03*random.random()
    )
