import torch
import numpy as np
from torch.utils.data import Dataset


class AIGS10(Dataset):
    def __init__(self, train, base_path='AIGS10/', transform=None):

        self.train = train
        self.labels = ['aeroplane', 'car', 'bird', 'cat', 'sheep', 'dog', 'chair', 'horse', 'boat', 'train']
        self.transform = transform

        self._load_data(base_path)

    def compute_mean_std(self):
        self.means=np.zeros(3)
        self.stds=np.zeros(3)
        for i in range(3):
            self.means[i]=np.mean(self.xs[:,:,:,i])
            self.stds[i]=np.std(self.xs[:,:,:,i])
        assert self.means.shape==(3,), (self.means.shape,self.xs.shape)
        return self.means,self.stds



    def _load_data(self, base_path):
        self.mode = 'train' if self.train else 'test'

        #xs = torch.tensor([], dtype=int)
        #ys = torch.tensor([], dtype=int)
        xs=np.empty((0,32,32,3),np.uint8)
        ys=np.empty((0))
        for i, label in enumerate(self.labels):
            path = f'{base_path}{self.mode}data/{label}_{self.mode}.npy'
            data=np.load(path)
            y=np.array([i]*len(data))
            xs=np.append(xs,data,axis=0)
            ys=np.append(ys,y,axis=0)
            #data = torch.tensor(np.load(path), dtype=int)
            #y = torch.tensor([i] * len(data))
            #xs = torch.cat([xs, data])
            #ys = torch.cat([ys, y])
        self.xs=xs
        self.ys=ys
        #self.xs = xs.transpose(1,2,0)
        #self.ys = ys

    def __len__(self):
        return len(self.xs)

    def __getitem__(self, idx):
        x, y = self.xs[idx], self.ys[idx]
        if self.transform:
            x = self.transform(x)
        return x, y



if __name__=="__main__":
    import torchvision.transforms as transforms

    means = (0.4914, 0.4822, 0.4465)
    transform_train = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(means, (0.2023, 0.1994, 0.2010)),
    ])
    train_dataset=AIGS10(train=True,transform=transform_train)
    test_dataset=AIGS10(train=False)
    print(f'Load train & test dataset, train size:{len(train_dataset)},test_size:{len(test_dataset)}')
    #print(train_dataset[0][0])