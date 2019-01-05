import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
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
test_set = CustomDataset(path="/home/yash/lung_data/test_image/",transforms = transformations_train)  
batch_size = 1


def test():
    cuda = torch.cuda.is_available()
    net = UNet11(1,32,3)
    if cuda:
        net = net.cuda()
    net.load_state_dict(torch.load('/home/yash/models/vggunet_he/Weights/cp_bce_interchange_custom_lr_460.pth.tar'))
    net.eval()

    print("preparing training data ...")
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    print("done ...")    

    for i, (images, masks,name) in tqdm(enumerate(test_loader)):
        images = Variable(images)
        masks = Variable(masks)
        if cuda:
            images = images.cuda()
            masks = masks.cuda()

        outputs = net(images)
        # print ('name is {}'.format(name[0]))
        torchvision.utils.save_image(outputs,'results/{}.png'.format(name[0]))
        # writer.add_image('Testing Input',images)
        # writer.add_image('Testing Pred',F.sigmoid(outputs)>0.5)

def main():
    test()



if __name__ == '__main__':
    main()