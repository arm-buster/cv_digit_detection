from torch.utils.data import Dataset
from stage1.utils import *
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import cv2
import numpy as np
import json
import pdb




class ConvLayer(nn.Module):

    def __init__(self, cin, cout, ksize, padding, maxpool = True):
        super().__init__()

        self.layer = [
            nn.Conv2d(cin, cout, ksize, 1, padding),
            nn.BatchNorm2d(cout),
            nn.ReLU()
        ]
        if maxpool:
            self.layer.append(nn.MaxPool2d(2,2))
        
        self.layer = nn.Sequential(*self.layer)
    
    def forward(self, x):
        return self.layer(x)



class MySmallNet(nn.Module):
    
    def __init__(self):
        super().__init__()
        #self.fc_size = 65536 # for input size 128x128
        self.fc_size = 16384
        self.conv1 = ConvLayer(3, 16, 5, 2)
        self.conv2 = ConvLayer(16, 32, 3, 1, maxpool= False)
        self.conv3 = ConvLayer(32, 64, 3, 1)
        self.fc1 = nn.Linear(self.fc_size, 1024) # warning, this will break
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, 11)
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.1)
        

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(-1, self.fc_size)
        x = self.fc1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x




class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
    def forward(self, X_out, y):
        loss=F.binary_cross_entropy_with_logits(X_out, y, reduce=False)
        focal_loss = self.alpha * ((1-torch.exp(-loss))**self.gamma) * loss
        return torch.mean(focal_loss)


class ModelEnsemble:

    def __init__(self, models, transformations, flips):
        """
        models: list of pytorch models
        transformations: pytorch transformations
        flip: same length as models, ith flip
        is a function to be called on each image before passing the
        image to the ith model
        """
        self.models = models
        self.trans = transformations
        self.flips = flips

    def __call__(self, original_cutout):
        """
        cutouts: list of mser cutouts (or another other images)
        to run inference on
        """
        preds = []
        conf = []

        for f, m in zip(self.flips, self.models):
            cutouts = [f(im.copy()) for im in original_cutout]
            cutouts = [self.trans(i) for i in cutouts]
            inp = torch.stack(cutouts)
            out = m(inp)
            predictions, confidence_scores = get_model_prediction_and_confidence(out)
            preds.append(predictions)
            conf.append(confidence_scores)

        pred = np.stack(preds, axis=1)
        conf = np.mean(np.stack(conf, axis=1), axis=1)
        row_equal = np.logical_and(pred[:,0] == pred[:,1], pred[:,0] == pred[:,2])
        row_equal = np.logical_and(row_equal, pred[:,0] == pred[:,3])
        row_equal = np.logical_and(row_equal, pred[:,2] != 10)
        isnumber = np.logical_and(conf > 2.5, row_equal)
        indices = np.where(isnumber)[0]
        return indices, pred[:,0], conf
        
        
        

        
class SvhnDatasetDigits(Dataset):

    def __init__(self, data_path, label_path, transform = None):
        self.data = np.load(data_path)
        with open(label_path, "r") as fp:
            self.labels = json.load(fp)
        self.transform = transform

    def __len__(self):
        return self.data.shape[0]

        
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img = self.data[idx].copy()
        lab = int(self.labels[idx])
        
        if self.transform:
            img = self.transform(img)
            #img = np.array(img, dtype = np.float32)

        full_label = np.zeros((11), dtype = np.float32)
        full_label[lab] = 1
        return img, full_label, lab
        


    


        