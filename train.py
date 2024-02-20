import torch
import torch.nn.functional as F
import torch.utils.data.dataloader as D
from vqvae import VQVAE
from net import Transformer3d
from dataset import Dataset
from utils import *
from sklearn.model_selection import train_test_split
import atexit
import time
from tqdm import tqdm
from einops import rearrange
from ranger import Ranger

def train_test_data(dataset, test_split):
    train_idx, test_idx = train_test_split(range(len(dataset)), test_size=test_split)
    return torch.utils.data.Subset(dataset, train_idx), torch.utils.data.Subset(dataset, test_idx)

class Trainer:
    def __init__(self, nlayers=6,
                       epochs=1,
                       batch_size=1,
                       max_ctx_length=25,
                       lr=3e-4,
                       halting_iteration=4096,
                       dmodel=32,
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
        self.vqvae = VQVAE(3, 16, 512, 4).to("mps").eval()
        self.vqvae.load_state_dict(torch.load('../saves/vqvae.pth'))
        self.engine = Transformer3d(512, dmodel, nheads, nlayers).to("mps")
        self.dataset = Dataset(max_ctx_length=max_ctx_length)
        print(get_nparams(self.engine), "params in generator net.")
        self.train_dataset, self.test_dataset = train_test_data(self.dataset, .2)
        self.train_dataloader = D.DataLoader(self.train_dataset, shuffle=True, batch_size=batch_size)
        self.test_dataloader = D.DataLoader(self.test_dataset, shuffle=True, batch_size=batch_size)
        _, self.display, self.surface = init_camera_and_window() 
        
        self.optim = Ranger(self.engine.parameters(), lr=self.lr,  num_batches_per_epoch=len(self.train_dataloader), num_epochs=self.epochs, use_warmup=False)
        # self.schedule = torch.optim.lr_scheduler.OneCycleLR(self.optim, self.lr, len(self.train_dataset))


        self.save_every = save_every
        self.step = 0

    def train(self):
        """Train the model."""
    
        print("Beginning Training...")
        running_engine_loss = 0
        # running_critic_loss = 0

        self.engine.train()
        for epoch in range(self.epochs):
            print("Starting Epoch:", epoch)
            self.engine.train()
            bar = tqdm(range(len(self.train_dataloader)))
            for i, (x) in enumerate(self.train_dataloader):
                if i  % self.save_every != 0 or i == 0:
                    engine_loss = self.training_step(x)
                    bar.set_description(f'Loss: {engine_loss:.4f}')

                else:
                    self.validation_step(x, i)
                    self.dream_step(x, i)
                    self.save()
                bar.update(1) 

    def training_step(self, x):
        """
        One optimization step
        :param x: Input data
        :param y: Target data
        :param step: Current training step
        :return: loss
        """
        self.engine.train()
        x = x.to("mps")        

        self.optim.zero_grad()
        x, y = x[:, :, :, :, :-1], x[:, :, :, :, 1:]


        with torch.no_grad():
            ind = self.vqvae.encode(x)
            y_ind = self.vqvae.encode(y) #B H W T (...512)


        logits = self.engine(ind) #B 512 H W T (0...1)
        # logits = rearrange(logits, 'b c h w t -> (b h w t) c')
        # y_ind = rearrange(y_ind, 'b h w t -> (b h w t)')

        gloss = torch.nn.CrossEntropyLoss()(logits, y_ind)

        gloss.backward()
        self.optim.step()

        self.step += 1
        
        return gloss.item()

    def weight(self, gloss, disc):
        if self.step > 5000:
            gloss_grads = torch.autograd.grad(gloss, self.engine.conv_output.conv.weight, retain_graph=True)[0]
            disc_grads = torch.autograd.grad(disc, self.engine.conv_output.conv.weight, retain_graph=True)[0]
            d_weight = torch.norm(gloss_grads) / (torch.norm(disc_grads) + 1e-6)
            d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        else:
            d_weight = 0
        return d_weight

    def validation_step(self, x, step):
        """
        One validation step
        :param x: Input data
        :param y: Target data
        :param step: Current training step
        :return: loss
        """
        x, y = x[:, :, :, :, :-1], x[:, :, :, :, 1:]
        self.engine.eval()
        x, y = x.to("mps"), y.to("mps")
        x_ind = self.vqvae.encode(x)

        y = self.vqvae.encode(y)
        y = self.vqvae.decode(y)

        y_false = self.engine(x_ind)
        y_false = torch.argmax(y_false, dim=1)

        y_false = self.vqvae.decode(y_false)

        y_seq = torch.cat([y, y_false], 2)
        for i in y_seq[0].unsqueeze(0).split(1, -1):
            show_tensor(i.cpu().squeeze(-1), self.display, self.surface)
            time.sleep(1./8.)

        # return loss.item()

    def dream_step(self, x, step):
        """
        One dream step
        :param x: Input data
        :param y: Target data
        :param step: Current training step
        :return: loss
        """

        #ISSUE: The model is not learning to dream. It is just outputting random noise.
        #POSSIBLE SOLUTION: Use contrastive decoding to generate the next frame.

        
        with torch.no_grad():
            x, y = x[:, :, :, :, :-1], x[:, :, :, :, 1:]
            x, y = x.to("mps"), y.to("mps")

            x_ind = self.vqvae.encode(x[:, :, :, :, 0])

            memory = [x_ind]
            for i in range(self.max_ctx_length - 1):
                memory_stacked = torch.stack(memory, -1)
                y_mem = self.engine(memory_stacked)[:, :, :, :, -1]

                #Sampling logic.
                B, PROB, H, W = y_mem.shape
                
                #Top-k Sampling
                y_mem = rearrange(y_mem, 'b c h w -> b (h w) c')

                # #Nucleus Sampling (Top-p)
                # probs = torch.softmax(y_mem, dim=-1)
                # sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                # cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                # nucleus = cumulative_probs > 0.9
                # nucleus = torch.cat([nucleus.new_ones(nucleus.shape[:-1] + (1,)), nucleus[..., :-1]], dim=-1)
                
                # sorted_log_probs = torch.log(sorted_probs)
                # sorted_log_probs[~nucleus] = float('-inf')

                y_temp = torch.softmax(y_mem/.3, dim=1)
                y_mem = torch.multinomial(y_temp[0], 1).unsqueeze(0).squeeze(-1)

                # #Reshape the output
                y_false = rearrange(y_mem, 'b (h w) -> b h w', h=H, w=W)
                
                memory.append(y_false)
                y_decoded = self.vqvae.decode(y_false)
                show_tensor(y_decoded.cpu().squeeze(-1), self.display, self.surface)

                time.sleep(1./8.)


    #         memory = [x[:, :, :, :, 0]]
    #         for i in memory:
    #             i.squeeze_(-1)

    #         for i in range(self.max_ctx_length - 1):
    #             memory_stacked = torch.stack(memory, -1)
    #             y_mem = self.engine(memory_stacked)
    #             y_mem = y_mem[:, :, :, :, -1]
    #             memory.append(y_mem)
    #             show_tensor(y_mem.cpu(), self.display, self.surface)
    #             time.sleep(1./8.)

    def save(self, path='../saves/checkpoint.pt'):
        """Save the model to disk."""
        torch.save({
            'optim':self.optim.state_dict(),
            'engine':self.engine.state_dict(),
            }, path)

    def load(self, path='../saves/checkpoint.pt'):
        """Load the model from disk."""

        checkpoint = torch.load(path, map_location='cpu')
        self.engine.load_state_dict(checkpoint['engine'])
        del checkpoint['engine']
        self.optim.load_state_dict(checkpoint['optim'])
        del checkpoint['optim']
        
if __name__ == '__main__':
    trainer = Trainer()
    # try:
    #     trainer.load('../saves/checkpoint.pt')
    # except:
    #     print("No checkpoint found. Training from scratch...")
    

    trainer.train()

