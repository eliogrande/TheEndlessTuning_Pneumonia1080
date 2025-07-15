"""Dataloader per Endless Tuning su Pneumonia"""

import shutil
import os
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import torchvision as tv
from torchvision import transforms,datasets
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
import random



def resize_and_pad(image, target_size):
    """Ridimensiona l'immagine mantenendo il rapporto e aggiunge padding nero per adattarla a target_size."""
    h, w = image.shape[:2]

    # Calcola il rapporto di ridimensionamento
    scale = min(target_size[0] / w, target_size[1] / h)
    new_w, new_h = int(w * scale), int(h * scale)

    # Ridimensiona mantenendo le proporzioni
    resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Calcola il padding necessario
    top = (target_size[1] - new_h) // 2
    bottom = target_size[1] - new_h - top
    left = (target_size[0] - new_w) // 2
    right = target_size[0] - new_w - left

    # Aggiunge il padding nero
    padded_image = cv2.copyMakeBorder(resized_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    return padded_image


class ImageFolderWithPaths(datasets.ImageFolder):
    def __getitem__(self, index):
        # Ottieni il dato dalla ImageFolder originale
        sample, label = super().__getitem__(index)
        # Recupera il percorso dell'immagine
        img_path = self.imgs[index][0]  # self.imgs è una lista di tuple (path, label)
        return sample, label, img_path


def create_dataset(num_samples,data_path,resize):
    torch.manual_seed(7)
    random.seed(7)

#TRAINING SET
    if os.path.exists('./Pneumonia_Or_Not'):
        shutil.rmtree('./Pneumonia_Or_Not') 
    prohibited_idx = {} 
    print('Selecting training data...')    
    for subfolder in os.listdir(data_path):
        prohibited_idx[subfolder]=[]
        print(f'Class {subfolder} loading...')
        label = subfolder.split('_')[0]
        img_dir = './Pneumonia_Or_Not/'+str(label)
        if not os.path.exists(img_dir):
            os.makedirs(img_dir)
        for idx in random.sample(range(len(os.listdir(os.path.join(data_path,subfolder)))),num_samples//len(os.listdir(data_path))):
            prohibited_idx[subfolder].append(idx)
            file = (os.listdir(os.path.join(data_path,subfolder)))[idx]
            img_bgr = cv2.imread(os.path.join(data_path, subfolder, file))
            img_bgr = resize_and_pad(img_bgr, target_size=resize)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img_rgb)
            img.save(img_dir+'/'+str(idx)+'.jpg')
        

#CASE STUDIES
    if os.path.exists('./case_studies'):
        shutil.rmtree('./case_studies') 
    print('Selecting case_study images...')
    for subfolder in os.listdir(data_path):
        print(f'Class {subfolder} loading...')
        label = subfolder.split('_')[0]
        img_dir = './case_studies/'+str(label)
        if not os.path.exists(img_dir):
            os.makedirs(img_dir)
        for idx in random.sample([i for i in range(len(os.listdir(os.path.join(data_path,subfolder)))) if i not in prohibited_idx[subfolder]],
                                 50):
            prohibited_idx[subfolder].append(idx)
            file = (os.listdir(os.path.join(data_path,subfolder)))[idx]
            img_bgr = cv2.imread(os.path.join(data_path, subfolder, file))
            img_bgr = resize_and_pad(img_bgr, target_size=resize)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img_rgb)
            img.save(img_dir+'/'+str(idx)+'.jpg')

#TUNING SET
    if os.path.exists('./temporary_images'):
        shutil.rmtree('./temporary_images') 
    print('Selecting tuning data for the experiment...')
    for subfolder in os.listdir(data_path):
        print(f'Class {subfolder} loading...')
        label = subfolder.split('_')[0]
        img_dir = './temporary_images/'+str(label)
        if not os.path.exists(img_dir):
            os.makedirs(img_dir)
        for idx in random.sample([i for i in range(len(os.listdir(os.path.join(data_path,subfolder)))) if i not in prohibited_idx[subfolder]],
                                 100):
            file = (os.listdir(os.path.join(data_path,subfolder)))[idx]
            img_bgr = cv2.imread(os.path.join(data_path, subfolder, file))
            img_bgr = resize_and_pad(img_bgr, target_size=resize)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img_rgb)
            img.save(img_dir+'/'+str(idx)+'.jpg')

    return None

def load_data(batch_size,dataset_path):

    transform = transforms.Compose([
        #transforms.RandomHorizontalFlip(),
        #transforms.RandomRotation(20),
        #transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    if dataset_path == 'Pneumonia_Or_Not':
        data_path = './Pneumonia_Or_Not'
        dataset = ImageFolderWithPaths(root=data_path, transform=transform)
        train_size = int(0.7 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])  
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        dataloader = DataLoader(dataset, batch_size=batch_size,shuffle=False)

        classes = len(os.listdir(data_path))
        print('The dataset is ready')
        return train_loader,val_loader,classes,dataloader

    elif dataset_path == 'Case Study':
        data_path = './case_studies'
        case_study_dataset = ImageFolderWithPaths(root=data_path, transform=transforms.ToTensor())
        case_study_dataloader = DataLoader(case_study_dataset, batch_size=batch_size, shuffle=False)
        classes = len(os.listdir(data_path))
        print('The dataset is ready')
        return case_study_dataloader,classes

    elif dataset_path == 'Tuning':
        data_path = './temporary_images'
        tuning_images = datasets.ImageFolder(root=data_path, transform=transform)
        classes = len(os.listdir(data_path))
        print('The dataset is ready')
        return classes,tuning_images

    else:
        print('Error: select a correct path')
        return None

if __name__ == '__main__':
    
    create_dataset(num_samples=2500,data_path='./Data_Source',resize=(1080,1080))