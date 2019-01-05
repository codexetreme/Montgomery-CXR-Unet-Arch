import torch
import os
import cv2
import numpy as np
def make_dataset():
    left = '/home/yash/lung_data/test_image/ManualMask/left'
    right = '/home/yash/lung_data/test_image/ManualMask/right'
    masks_l = os.listdir(left)
    masks_r = os.listdir(right)
    # left_mask = os.path.join(img_and_label, masks[0])
    # right_mask = os.path.join(img_and_label, masks[1])
    # k = os.listdir(masks_l)
    # print (k)
    for i,p in enumerate(masks_l):
        if p !=masks_r[i]:
            print (os.path.join(left,p))
            print (masks_r[i])
        else:
            img_l = cv2.imread(os.path.join(left,p))
            img_r = cv2.imread(os.path.join(right,masks_r[i]))
            img_l = np.clip(img_l+img_r,0,255)
            cv2.imwrite('gt/{}'.format(p),img_l)
            img = Image.open('gt/{}'.format(p),img_l)


make_dataset()