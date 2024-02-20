# This file is used to train the VQVAE model. It is a simple script that uses the VQVAE class to train the model.

import torch
import torch.nn.functional as F
import torch.utils.data.dataloader as D
from vqvae import VQVAE, Discriminator
from dataset import Dataset
from utils import *
from sklearn.model_selection import train_test_split
import atexit
import time
from tqdm import tqdm
import torchvision.models as models
import torch.nn as nn
from ranger import Ranger

def train_test_data(dataset, test_split):
    train_idx, test_idx = train_test_split(range(len(dataset)), test_size=test_split)
    return torch.utils.data.Subset(dataset, train_idx), torch.utils.data.Subset(dataset, test_idx)

class Trainer:
    def __init__(self, nlayers=4,
                       epochs=1,
                       batch_size=8,
                       max_ctx_length=0,
                       lr=3e-4,
                       halting_iteration=4096,
                       dmodel=256,
                       nheads=16,
                       save_every=100,
                       alpha=0.2):
        
        self.nlayers = nlayers
        self.epochs = epochs
        self.max_ctx_length = max_ctx_length
        self.batch_size = batch_size
        self.halting_iteration = halting_iteration
        self.lr = lr
        self.alpha = alpha
        print("Loading Nets...")
        self.engine = VQVAE(3, 16, 512, 4).to("mps")
        self.engine.load_state_dict(torch.load("../saves/vqvae_epoch1_80000its")) # Not training from scratch; too expensive.
        self.discriminator = Discriminator(3, 16).to("mps")
        
        self.discriminator_optim = torch.optim.RAdam(self.discriminator.parameters(), lr=self.lr)
        self.dataset = Dataset(max_ctx_length=max_ctx_length)
        print(get_nparams(self.engine), "params in generator net.")
        self.train_dataset, self.test_dataset = train_test_data(self.dataset, .2)
        self.train_dataloader = D.DataLoader(self.train_dataset, shuffle=True, batch_size=batch_size)
        self.test_dataloader = D.DataLoader(self.test_dataset, shuffle=True, batch_size=batch_size)
        _, self.display, self.surface = init_camera_and_window() 
        self.optim = Ranger(self.engine.parameters(), lr=self.lr, num_batches_per_epoch=len(self.train_dataloader), num_epochs=self.epochs)

        # self.schedule = torch.optim.lr_scheduler.OneCycleLR(self.optim, self.lr, len(self.train_dataset))


        self.save_every = save_every
        self.step = 0
    def train(self):
        """Train the model."""
    
        print("Beginning Training...")
        running_engine_loss = 0
        running_critic_loss = 0

        self.engine.train()
        for epoch in range(self.epochs):
            bar = tqdm(range(len(self.train_dataloader)))
            for i, (x) in enumerate(self.train_dataloader):
                if i % self.save_every != 0 or i == 0:
                    engine_loss = self.training_step(x)
                    running_engine_loss += engine_loss
                    bar.set_description(f'Loss: {engine_loss:.5f} | Running Loss: {running_engine_loss / (self.step + 1):.5f}')
                else:
                    self.save()
                bar.update(1)
                self.step += 1


    def training_step(self, x):
        self.optim.zero_grad()
        x = x.squeeze(-1)
        x = x.to("mps")
        x_hat, vq = self.engine(x)

        recon = torch.abs(x - x_hat).mean()
        recon = recon / x.size(0)

        vq = vq.mean()

        disc = self.discriminator(x_hat)
        disc = disc.mean()
        disc = -disc

        gloss = recon + self.alpha_penalty(disc, recon) * disc + vq
        gloss.backward()
        self.optim.step()

        if self.step > 100:
            self.discriminator_optim.zero_grad()

            real = self.discriminator(x)
            fake = self.discriminator(x_hat.detach())

            loss = torch.relu(1 - real).mean() + torch.relu(1 + fake).mean()
            loss = loss / 2
            loss.backward()

            self.discriminator_optim.step()
        return gloss.item()
    
    def alpha_penalty(self, disc, recon):
        if self.step > 100:
            gloss_grads = torch.autograd.grad(recon, self.engine.out.weight, retain_graph=True)[0]
            disc_grads = torch.autograd.grad(disc, self.engine.out.weight, retain_graph=True)[0]
            d_weight = torch.norm(gloss_grads) / (torch.norm(disc_grads) + 1e-6)
            d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        else:
            d_weight = 0.0

        return d_weight    
    def save(self):
        torch.save(self.engine.state_dict(), "../saves/vqvae.pth")


if __name__ == "__main__":
    trainer = Trainer()
    trainer.train()
    trainer.save()
    print("Model Saved.")
    print("Training Complete.")