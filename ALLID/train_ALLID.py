import torch
import torchvision
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, Subset
import os
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
import Model_ALLID as Model
from torch import optim
import math
from matplotlib import pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import KFold


time_window = 3


class LeukLoader(Dataset):
    def __init__(self, path, transform=None):

        self.classes = os.listdir(path)
        print('class order', self.classes)
        self.benign_class = 0
        self.dataList = []
        self.path = path
        self.transform = transform

        for i in range(len(self.classes)):
            if('benign' in self.classes[i].lower() or 'control' in self.classes[i].lower()):
                print('found benign class', i)
                self.benign_class = i
            for file in os.listdir(f'{path}/{self.classes[i]}'):
                f = f'{path}/{self.classes[i]}/{file}'
                
                self.dataList.append([f, i])

                


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
            bitmap = bitmap/bitmap.max()
            self.sub_dataset.append([bitmap, target])
        
    def __len__(self):
        return len(self.sub_dataset)
        
    def __getitem__(self, index):
        image, target = self.sub_dataset[index]
        image = image.to('cuda')
    
        if(image.max() != 1):
            image = image/image.max()
            
        if(self.transform != None):
            
            return self.transform(image), target
        
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
            bitmap = bitmap/bitmap.max()
            self.sub_dataset.append([bitmap, target])
        
    def __len__(self):
        return len(self.sub_dataset)
        
    def __getitem__(self, index):
        image, target = self.sub_dataset[index]
        #image = torchvision.io.read_image(file)
                #bitmap = bitmap.to('cuda')
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

            torchvision.transforms.RandomCrop(size=224, padding=8),
            torchvision.transforms.RandomRotation(20),
            ])
            
    dataset = LeukLoader(r'Path to the ALLID dataset on your computer')

    b_class = dataset.get_benign_class()    
    batch_size = 16
    num_classes = 4
    dataset_size = len(dataset)

    k=5
    fold_results = []
    
    for fold in range(k):
        
        torch.cuda.empty_cache()

        train_idxs = np.loadtxt(f'train_idxs_T{time_window}_{fold}.txt')
        val_idxs = np.loadtxt(f'val_idxs_T{time_window}_{fold}.txt')
        train_idx = train_idxs.astype(int)
        val_idx = val_idxs.astype(int)
      
            
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)
        
        print('len of train', len(train_subset))
        print('len of val', len(val_subset))

        train_dataset = trainDataset(train_subset, transform=transform)
        val_dataset = testDataset(val_subset)

        
        train_drop = False
        test_drop = False
        
        if len(train_dataset)%batch_size == 1:
            print('last batch will have one image in train set. Dropping from dataset')
            train_drop = True
            
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=train_drop)
        validation_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
                                                        
        model = Model.Model()
        model = model.to('cuda')
        optimizer = torch.optim.Adam([{
            'params': model.parameters(),
            'initial_lr': 1e-3
        }], lr=1e-3)
        
        loss_fn = torch.nn.CrossEntropyLoss()
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-5)
        
        max_acc = 0
        max_train_acc = 0
        best_epoch = 0
        train_hist = torch.zeros(100)
        val_hist = torch.zeros(100)
        f = open(f'ALL_5FoldCV_NoAffine_T{time_window}_fold{fold}.txt', 'a')
        file = f'best_all_run{fold}.pth'
        torch.cuda.empty_cache()
        
        for e in range(100):
           
            avg_val_spk_rate = np.zeros((10, time_window))
            
            print('starting epoch', e)
            f.write(f'starting epoch {e}\n')

            
            print('learning rate', optimizer.param_groups[0]['lr'])
            lr_temp = optimizer.param_groups[0]['lr']
            f.write(f'learning rate {lr_temp}\n')
            train_acc = train(model, train_loader, optimizer, loss_fn)
            
            f.write(f'train accuracy: {train_acc}\n')
            train_hist[e] = train_acc
            
            acc, val_loss, val_spk_rate, conf_mat = val(model, validation_loader, loss_fn, num_classes) 
            scheduler.step(val_loss)

            avg_val_spk_rate = val_spk_rate
            f.write(f'test accuracy: {acc}\n')
            f.write(f'avg test spk rates: {avg_val_spk_rate}\n')
            f.write(f'confusion matrix: {conf_mat}\n\n')

            val_hist[e] = acc
            
            if(acc > max_acc or (acc == max_acc and train_acc > max_train_acc)):
                max_acc = acc
                max_train_acc = train_acc
                best_epoch = e
                torch.save({'epoch':e, 'model_state_dict':model.state_dict(),'optimizer_state_dict': optimizer.state_dict()}, file)
            
            if acc > 99.99:
                    break
            torch.cuda.empty_cache()
            
        fold_results.append(max_acc)
           
        print(f'max accuracy for fold {max_acc}')
        
        f.write(f'max accuracy for fold {max_acc}')
        f.write(f'Best epoch {best_epoch}')
        
        f.close() 
    
    print('Fold Accuracies', fold_results)
    print('Avg for all folds', sum(fold_results)/len(fold_results))


      
      
def confusion_matrix(preds, targets, num_classes):
      
        class_counter = torch.zeros((num_classes, 1), device=preds.device)
        conf_mat = torch.zeros((num_classes, num_classes), device=preds.device)
        for i in range(len(targets)):
            conf_mat[targets[i]][preds[i]] += 1
            
        for j in range(4):
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
        
        _, preds = output.max(1)
        correct += preds.eq(targets).sum()
        loss = loss_fn(output, targets)

        loss.backward()
        optimizer.step()
       
    
    print('Accuracy for train set:', correct/num_images * 100)
    return correct/num_images * 100
    
    
    
@torch.no_grad()           
def val(model, validation_loader, loss_fn, num_classes):
    model.eval()
    correct = 0
    num_images = 0
    num_batches = 0
    avg_test_spk_rate = torch.zeros((10, time_window), device='cuda')
    
    conf_mat = torch.zeros((4,4), device='cuda')
    class_counter = torch.zeros((4, 1), device='cuda')
    
    val_loss = 0
    for index, (images, targets) in enumerate(validation_loader):
        num_images += images.shape[0]
        num_batches += 1
        #images = images.to('cuda')
        targets = targets.to('cuda')
        output, spk_rate = model(images)

        avg_test_spk_rate += spk_rate
        
        _, preds = output.max(1)

        correct += preds.eq(targets).sum()
        val_loss += loss_fn(output, targets).item()
        conf_m, c_count = confusion_matrix(preds, targets, num_classes)
        conf_mat += conf_m
        class_counter += c_count
        
    
    print('Accuracy for test set:', correct/num_images * 100)
    print('Loss for val:', val_loss/num_batches)
    print('avg test spk rate', avg_test_spk_rate/num_batches)
    print('confusion matrix:', conf_mat)
    
    return correct/num_images * 100, val_loss/num_batches, avg_test_spk_rate/num_batches, conf_mat


if __name__ == '__main__':
    main()