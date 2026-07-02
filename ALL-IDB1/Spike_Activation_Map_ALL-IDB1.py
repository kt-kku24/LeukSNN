import torch
import torchvision
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, Subset
import os
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
import Model_ALL_IDB1 as Model
from torch import optim
import math
from matplotlib import pyplot as plt
import numpy as np
import shap
import torch.nn.functional as F

time_window=3

class LeukLoader(Dataset):
    def __init__(self, path, transform=None):

     
        self.benign_class = 0
 
        self.dataList = []
        self.path = path
        self.transform = transform

        
        for file in os.listdir(f'{path}/'):
            f = f'{path}/{file}'
            class_ = file.split('_')[1].split('.')[0]

            self.dataList.append([f, int(class_)])
                


    def get_benign_class(self):
        return self.benign_class
    
    def __len__(self):
        return len(self.dataList)

    def __getitem__(self, index):
    
        return self.dataList[index]
    
    
        
class trainDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform
        self.sub_dataset = []
        for data in dataset:
            file, target = data
            bitmap = torchvision.io.read_image(file)
            self.sub_dataset.append([bitmap, target])
        
    def __len__(self):
        return len(self.sub_dataset)
        
    def __getitem__(self, index):
        image, target = self.sub_dataset[index]
        image = image.to('cuda')

        if(image.max() != 1):
            image = image/image.max()
            
        if(self.transform != None):
            image = self.transform(image)
            return image, target
        
        else:
            return image, target

            
class testDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform
        
        self.sub_dataset = []
        for data in dataset:
            file, target = data
            bitmap = torchvision.io.read_image(file)
            self.sub_dataset.append([bitmap, target])
        
    def __len__(self):
        return len(self.sub_dataset)
        
    def __getitem__(self, index):
        image, target = self.sub_dataset[index]
        
        image = image.to('cuda')
        
        if(image.max() != 1):
            image = image/image.max()
            
        if(self.transform != None):
            image = self.transform(image)
            return image, target
        
        else:
            return image, target
            
           
    
spike_activations = []

def spike_hook(module, input, output):

    spike_activations.append(output.detach())
        
    
def main():
   
    dataset = LeukLoader(r'Path to the ALL-IDB1 dataset on your computer')
    print(len(dataset))
    b_class = dataset.get_benign_class()    
    batch_size = 22

    torch.cuda.empty_cache()
    fold = 1

    val_idxs = np.loadtxt(f'val_idxs_T{time_window}_{fold}.txt')

    val_idx = val_idxs.astype(int)

    val_subset = Subset(dataset, val_idx)

    print('len of val', len(val_subset))

    val_dataset = testDataset(val_subset)

    validation_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

                                                    
    model = Model.Model()
    model = model.to('cuda')
    
   
    prev_epoch = 0
    model_file = f'best_all_idb1_fold{fold}.pth'
    if os.path.exists(model_file):
        checkpoint = torch.load(model_file, map_location="cuda:0")
        model.load_state_dict(checkpoint['model_state_dict'])
        prev_epoch = checkpoint['epoch']
        print('warmed up model loaded')
        
    model.cuda()

    model.eval()
    image = torch.zeros(1)
    label = torch.zeros(1)

    image, label = next(iter(validation_loader))
    
    image = image.cuda()

    hook = model.conv3.lif3.register_forward_hook(spike_hook)
   
    spike_activations.clear()

    output, spk = model(image)
    prediction = torch.sigmoid(output) > 0.5

    hook.remove()
    
    spikes = torch.stack(spike_activations)

    # Sum over time    
    spikes_sum = spikes.sum(dim=0)**2
    #sum over channels
    sam = spikes_sum.sum(dim=1)
    sam = sam - sam.amin(dim=(1,2), keepdim=True)
    sam = sam / (sam.amax(dim=(1,2), keepdim=True) + 1e-8)

    sam = F.interpolate(
            sam.unsqueeze(1), 
            size=(224, 224),
            mode='bilinear',
            align_corners=False
        ).squeeze(1)
        
    classes = ['Benign', 'Malignant']
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300
        
    for i in range(sam.shape[0]):
        if prediction[i] == 0:
            continue
        input_np = image[i].permute(1, 2, 0).cpu().numpy()
        heatmap = sam[i].cpu().numpy()
        
        
        fig, ax = plt.subplots(1, 2)
        for a in ax.flat:
            a.set_axis_off()
        ax[0].imshow(input_np)
        ax[0].set_title('Original Image')
        ax[1].imshow(input_np)
        ax[1].imshow(heatmap, cmap='jet', alpha=0.5)
        ax[1].set_title(f'Heat Map for {classes[prediction[i]]}')
        plt.axis('off')
        #plt.constrained_layout
        plt.show()

    
if __name__ == '__main__':
    main()