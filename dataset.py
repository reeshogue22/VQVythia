from collections import deque
import glob
import torch
import torch.nn.functional as F
import torchvision.io as Tvio
import multiprocessing as mp
import random
import itertools

from torchvision.transforms import v2 as transforms

class Dataset(torch.utils.data.Dataset):
    def __init__(self, directory='../data/*', get_subdirs=True, max_ctx_length=32, data_aug=False):
        print("Loading dataset...")
        self.data_aug = data_aug
        self.data = glob.glob(directory)
        if get_subdirs:
            with mp.Pool(mp.cpu_count()) as data_procs:
                self.data_temp = [i for i in data_procs.imap_unordered(self.extract_from_data, list(range(len(self.data))))]
                self.data_temp = list(itertools.chain(*self.data_temp))
            self.data = self.data_temp
            self.max_ctx_length = max_ctx_length
    def extract_from_data(self, i):
        i_data = self.data[i]
        file_data = glob.glob(i_data+"/*")
        len_data = len(file_data)
        sorted_data = [i_data + "/" + str(i).zfill(7) + ".jpg" for i in range(1, len_data+1)]
        return sorted_data

    def __len__(self):
        return len(self.data) - self.max_ctx_length - 1

    def __getitem__(self, key):
        try:
            frame_start = key
            frames = []
            i_frame = frame_start
            
            while len(frames) <= self.max_ctx_length:
                frame = ((Tvio.read_image(self.data[i_frame], mode=Tvio.ImageReadMode.RGB).float() / 255))
                frames.append(frame)
                i_frame += 1
        except RuntimeError:
            print("Could not read file", self.data[i_frame])
            return self.__getitem__(random.randint(0, len(self)))
        return torch.stack(frames, dim=-1)

if __name__ == "__main__":
    dataset = Dataset()
    print(len(dataset))
    print(dataset.__getitem__(8)[0].shape)