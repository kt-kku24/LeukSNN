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
from sklearn.model_selection import KFold
import random

time_window = 3

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
            
            
                   
    
def main():
    transform = torchvision.transforms.Compose([
            torchvision.transforms.RandomHorizontalFlip(p=0.5),
            torchvision.transforms.RandomVerticalFlip(p=0.5),
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.RandomRotation(30),
            ])
    
    val_transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((224, 224)),
            ])
    dataset = LeukLoader(r'Path to ALL-IDB1 dataset on your computer')
    
    print(len(dataset))
    b_class = dataset.get_benign_class()    
    batch_size = 8
    validation_split = .1
    shuffle_dataset = True
    
    dataset_size = len(dataset)
    
    k = 5

    fold_results = []
    max_acc = 0

    for fold in range(k):
       
        train_idxs = np.loadtxt(f'train_idxs_T{time_window}_{fold}.txt')
        val_idxs = np.loadtxt(f'val_idxs_T{time_window}_{fold}.txt')
        train_idx = train_idxs.astype(int)
        val_idx = val_idxs.astype(int)
           
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)
        print('len of train', len(train_subset))
        print('len of val', len(val_subset))

        train_dataset = trainDataset(train_subset, transform=transform)
        val_dataset = testDataset(val_subset, transform=val_transform))

        train_drop = False
        
        if len(train_dataset)%16 == 1:
            print('last batch will have one image in train set. Dropping from dataset')
            train_drop = True


        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=train_drop)
        validation_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

                                                        
        model = Model.Model()
        model = model.to('cuda')
        
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        loss_fn = torch.nn.BCEWithLogitsLoss()
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
        max_acc = 0
        max_train_acc = 0
        best_epoch = 0
        train_hist = torch.zeros(60)
        val_hist = torch.zeros(60)
        f = open(f'ALLIDB1_T{time_window}_noAffine_{fold}.txt', 'a')
        file = f'best_all_idb1_fold{fold}.pth'
        for e in range(60):
            
            
            avg_val_spk_rate = np.zeros((10,time_window))
            print('starting epoch', e)
            f.write(f'starting epoch {e}\n')

            
            print('learning rate', optimizer.param_groups[0]['lr'])
            lr_temp = optimizer.param_groups[0]['lr']
            f.write(f'learning rate {lr_temp}\n')
            
            train_acc = train(model, train_loader, optimizer, loss_fn)
            scheduler.step()
            
            f.write(f'train accuracy: {train_acc}\n')
            train_hist[e] = train_acc

            acc, val_loss, val_spk_rate, conf_mat = val(model, validation_loader, loss_fn, b_class) 

            avg_val_spk_rate = val_spk_rate
            f.write(f'val accuracy: {acc}\n')
            f.write(f'avg val spk rates: {avg_val_spk_rate}\n')
            f.write(f'confusion matrix: {conf_mat}\n\n')

            val_hist[e] = acc
            if((acc > max_acc) or (acc >= max_acc and train_acc > max_train_acc) or (acc > 99.99 and train_acc > 99.99)):
                max_acc = acc
                max_train_acc = train_acc
                best_epoch = e
                torch.save({'epoch':e, 'model_state_dict':model.state_dict(),'optimizer_state_dict': optimizer.state_dict()}, file) 
 
        fold_results.append(max_acc)
           
        print(f'max accuracy for folds {max_acc}')
        print(f'best epoch {best_epoch}')
        f.write(f'max accuracy for folds {max_acc}')
        f.write(f'Best epoch {best_epoch}')
        f.close()    
    print('fold results', fold_results)   



def confusion_matrix(preds, targets, num_classes):
        
        
        class_counter = torch.zeros((num_classes, 1), device=preds.device)
        conf_mat = torch.zeros((num_classes, num_classes), device=preds.device)
        if targets.shape[0] > 1:
            for i in range(len(targets)):
                
                conf_mat[targets[i]][preds[i]] += 1
                
            for j in range(num_classes):
                class_counter[j][0] += (targets == j).sum()
        else:
            conf_mat[targets.item()][preds.item()] += 1
            for j in range(num_classes):
                class_counter[j][0] += (targets == j).sum()
        
        return conf_mat, class_counter
        
        
        
def train(model, train_loader, optimizer, loss_fn):
    model.train()
    correct = 0
    num_images = 0
    for index, (images, targets) in enumerate(train_loader):

        num_images += images.shape[0]
        optimizer.zero_grad()
        
       
        #images = images.to('cuda')

        targets = targets.to('cuda')
        
       
        output, spk_rate = model(images)
            
        preds = torch.sigmoid(output) > 0.5 
        
        correct += preds.squeeze().eq(targets).sum()
        loss = loss_fn(output.squeeze(), targets.float())
       
        loss.backward()
        optimizer.step()
       
    
    print('Accuracy for train set:', correct/num_images * 100)
    return correct/num_images * 100
    
    
    
@torch.no_grad()           
def val(model, validation_loader, loss_fn, benign_class):
    model.eval()
    correct = 0
    num_images = 0
    num_batches = 0
    avg_val_spk_rate = torch.zeros((10, time_window), device='cuda')
   
    num_classes=2
    conf_mat = torch.zeros((num_classes, num_classes), device='cuda')
    class_counter = torch.zeros((num_classes, 1), device='cuda')
    val_loss = 0
    for index, (images, targets) in enumerate(validation_loader):
        num_images += images.shape[0]
        num_batches += 1
        #images = images.to('cuda')
        targets = targets.to('cuda')
 
        output, spk_rate = model(images)

        avg_val_spk_rate += spk_rate
        
        preds = (torch.sigmoid(output) > 0.5).int() 
        correct += preds.squeeze().eq(targets).sum()
        
        val_loss += loss_fn(output.squeeze(), targets.float()).item()
        conf_m, c_count = confusion_matrix(preds.squeeze(), targets, num_classes)
        conf_mat += conf_m
        class_counter += c_count
        
    
    print('Accuracy for val set:', correct/num_images * 100)
    print('Loss for val:', val_loss/num_batches)
    print('avg val spk rate', avg_val_spk_rate/num_batches)
    print('val confusion matrix:', conf_mat)

    return correct/num_images * 100, val_loss/num_batches, avg_val_spk_rate/num_batches, conf_mat
    
    


if __name__ == '__main__':
    main()
