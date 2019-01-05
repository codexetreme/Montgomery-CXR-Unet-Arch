
# from torch.utils.data.dataset import Dataset
# from torchvision import transforms
# from skimage import io, transform
# from PIL import Image
# import os
# import numpy as np
# from Transforms import *


# class NucleiSeg(Dataset):
#     def __init__(self, path='/home/Drive2/deepak/New_whole_image_input/Final_journal_model/Training/Train_40_x_HE/', transforms=None):
#         self.path = path
#         self.list = os.listdir(self.path)
        
#         self.transforms = transforms
        
#     def __getitem__(self, index):
#         # stuff
#         image_path = '/home/Drive2/deepak/New_whole_image_input/Final_journal_model/Training/Train_40_x_HE/'
#         mask_path = '/home/Drive2/deepak/New_whole_image_input/Final_journal_model/Training/Train_40_y_HE/'
#         image = Image.open(image_path+self.list[index])
#         image = image.convert('RGB')
#         mask = Image.open(mask_path+self.list[index])
#         mask = mask.convert('L')
         
#         if self.transforms is not None:
#             image = self.transforms(image)
#             mask = self.transforms(mask)
#         # If the transform variable is not empty
#         # then it applies the operations in the transforms with the order that it is created.
#         return (image, mask)

#     def __len__(self):
#         return len(self.list) # of how many data(images?) you have

    
# class NucleiSegVal(Dataset):
#     def __init__(self, path='/home/Drive2/deepak/New_whole_image_input/Final_journal_model/Validation/Valid_40_x_HE/', transforms=None):
#         self.path = path
#         self.list = os.listdir(self.path)
        
#         self.transforms = transforms
        
#     def __getitem__(self, index):
#         # stuff
#         image_path = '/home/Drive2/deepak/New_whole_image_input/Final_journal_model/Validation/Valid_40_x_HE/'
#         mask_path = '/home/Drive2/deepak/New_whole_image_input/Final_journal_model/Validation/Valid_40_y_HE/'
#         image_name = self.list[index]
#         image = Image.open(image_path+self.list[index])
#         image = image.convert('RGB')
#         mask = Image.open(mask_path+self.list[index])
#         mask = mask.convert('L')
#         if self.transforms is not None:
#             image = self.transforms(image)
#             mask = self.transforms(mask)
#         # If the transform variable is not empty
#         # then it applies the operations in the transforms with the order that it is created.
#         return (image, mask, image_name )
        
#     def __len__(self):
#         return len(self.list)
        
# class NucleiSegTest(Dataset):
#     def __init__(self, path='/home/Drive2/deepak/New_whole_image_input/Final_journal_model/Testing/Test_40_x_HE/', transforms=None):
#         self.path = path
#         self.list = os.listdir(self.path)
        
#         self.transforms = transforms
        
#     def __getitem__(self, index):
#         # stuff
#         image_path = '/home/Drive2/deepak/New_whole_image_input/Final_journal_model/Testing/Test_40_x_HE/'
#         mask_path = '/home/Drive2/deepak/New_whole_image_input/Final_journal_model/Testing/Test_40_y_HE/'
#         image_name = self.list[index]
#         image = Image.open(image_path+self.list[index])
#         image = image.convert('RGB')
#         mask = Image.open(mask_path+self.list[index])
#         mask = mask.convert('L')
#         if self.transforms is not None:
#             image = self.transforms(image)
#             mask = self.transforms(mask)
#         # If the transform variable is not empty
#         # then it applies the operations in the transforms with the order that it is created.
#         return (image, mask, image_name )

#     def __len__(self):
#         return len(self.list)




import os

import torch
import torch.utils.data as datautil
from PIL import Image
import torchvision

def make_dataset(path):
    imgs = []
    labels = []
    # path = ...../train/
    loaded_path = os.listdir(path)
    #print (loaded_path)
    img_and_label = (os.path.join(path,'CXR_png'),os.path.join(path,'ManualMask'))
    for p in os.listdir(img_and_label[0]):
        imgs.append(os.path.join(img_and_label[0], p))
    pass
    masks = os.listdir(img_and_label[1])
    left_mask = os.path.join(img_and_label[1], masks[0])
    right_mask = os.path.join(img_and_label[1], masks[1])
    for p in os.listdir(left_mask):
        labels.append(os.path.join(left_mask, p))
    
    for i, p in enumerate(os.listdir(right_mask)):
        labels[i] = (labels[i], os.path.join(right_mask, p))
    return imgs, labels


#make_dataset(r"/home/yash/lung_data/train")
# addon = '/CXR_png/MCUCXR_0001_0.png'

class CustomDataset(datautil.Dataset):

    def __init__(self, path, transforms=None, variant="Train"):
        self.transforms = transforms
        self.variant = variant
        self.data, self.masks = make_dataset(path)

    def __getitem__(self, index):
        label_left = Image.open(self.masks[index][0])
        label_right = Image.open(self.masks[index][1])



        name = os.path.split(self.masks[index][0])[1].split('.')[0]
        print (name)
        image = self.data[index]
        image = Image.open(image).convert('RGB')
        if self.transforms is not None:
            image = self.transforms(image)
            label_left = self.transforms(label_left)
            label_right = self.transforms(label_right)
            # label_left.unsqueeze_(0)
            # label_right.unsqueeze_(0)

            label_left.add_(label_right)
            label_left = torch.Tensor(label_left)
            
        return image, label_left

        pass

    def __len__(self):
        return len(self.data)
        pass

