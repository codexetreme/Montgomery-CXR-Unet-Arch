import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
import torchvision.utils as vutils
from torchvision import transforms
from torch.autograd import Variable
from build_model import *
import os
from tqdm import tqdm
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import MultiStepLR



os.environ["CUDA_VISIBLE_DEVICES"]="0"

transformations_train = transforms.Compose([
    transforms.Resize((512,512)),
    transforms.ToTensor()])
    
transformations_val = transforms.Compose([transforms.ToTensor()])     
#-------------------------------------------------------------                                      
writer = SummaryWriter()
from data_loader import *
test_set = CustomDataset(path="/home/codexetreme/Documents/internwork/lung_data/train",transforms = transformations_train)  
batch_size = 2


def test():
    cuda = torch.cuda.is_available()
    net = UNet11(1,32,3)
    if cuda:
        net = net.cuda()
    # net.load_state_dict(torch.load('/home/yash/models/vggunet_he/Weights/cp_bce_interchange_custom_lr_460.pth.tar'))
    net.eval()

    print("preparing training data ...")
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    print("done ...")    

    for i, (images, masks) in tqdm(enumerate(test_loader)):
        images = Variable(images)
        masks = Variable(masks)
        if cuda:
            images = images.cuda()
            masks = masks.cuda()
        print ("images.shape ",images.shape)
        outputs = net(images)
        outputs = F.sigmoid(outputs)>0.5
        x = vutils.make_grid(images, normalize=True, scale_each=True)
        y = vutils.make_grid(outputs, normalize=True, scale_each=True)
        print ("x.shape ",x.shape)
        print ("x.shape ",y.shape)
        writer.add_image('Testing Input',x)
        writer.add_image('Testing Pred',y)

def main():
    test()



if __name__ == '__main__':
    main()