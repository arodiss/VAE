import os
import pathlib
import torch
from torch import optim
import pytorch_lightning as pl
import torchvision.utils as vutils


class Experiment(pl.LightningModule):
    def __init__(self, model, kld_weight=0.00025, learning_rate=0.005, scheduler_gamma=0.95) -> None:
        super(Experiment, self).__init__()
        self.model = model
        self.kld_weight = kld_weight
        self.learning_rate = learning_rate
        self.scheduler_gamma = scheduler_gamma
        self.step = 0

    def forward(self, data):
        return self.model(data)

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        results = self.forward(batch)
        train_loss = self.model.loss_function(
            *results,
            kld_weight=self.kld_weight,
        )
        z = train_loss['z']
        del train_loss['z']
        self.log_dict({key: val.item() for key, val in train_loss.items()}, sync_dist=True)
        self.step += 1
        self.logger.experiment.add_histogram('z distribution', z, global_step=self.step)
        return train_loss['loss']

    def validation_step(self, batch, batch_idx, optimizer_idx=0):
        results = self.forward(batch)
        val_loss = self.model.loss_function(
            *results,
            kld_weight=self.kld_weight,
        )
        del val_loss['z']
        self.log_dict({f"val_{key}": val.item() for key, val in val_loss.items()}, sync_dist=True)

    def on_validation_end(self) -> None:
        self.sample_images()

    def sample_images(self):
        pathlib.Path(os.path.join(self.logger.log_dir, "Reconstructions")).mkdir(parents=True, exist_ok=True)
        real_image = next(iter(self.trainer.datamodule.val_dataloader()))
        reconstructed_image = self.model.reconstruct(real_image)
        vutils.save_image(
            torch.cat((real_image.data, reconstructed_image.data), 0),
            os.path.join(
                self.logger.log_dir,
                "Reconstructions",
                f"recons_{self.logger.name}_Epoch_{self.current_epoch}.png"
            ),
            normalize=True,
            nrow=8,
            ncol=2,
        )

        pathlib.Path(os.path.join(self.logger.log_dir, "Generated")).mkdir(parents=True, exist_ok=True)
        samples = self.model.generate(8)
        vutils.save_image(
            samples.data,
            os.path.join(
                self.logger.log_dir,
                "Generated",
                f"{self.logger.name}_Epoch_{self.current_epoch}.png"
            ),
            normalize=True,
            nrow=4
        )

    def configure_optimizers(self):
        optimizers = [
            optim.Adam(self.model.parameters(), lr=self.learning_rate)
        ]
        schedulers = [
            optim.lr_scheduler.ExponentialLR(optimizers[0], gamma=self.scheduler_gamma)
        ]
        return optimizers, schedulers
