import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.autograd import Variable
from build_model import *
import os
from tqdm import tqdm
from torch.optim.lr_scheduler import MultiStepLR
from Transforms import *

os.environ["CUDA_VISIBLE_DEVICES"]="0"
import csv

class SoftDiceLoss(nn.Module):
    '''
    Soft Dice Loss
    '''        
    def __init__(self, weight=None, size_average=True):
        super(SoftDiceLoss, self).__init__()

    def forward(self, logits, targets):
        smooth = 1.
        logits = F.sigmoid(logits)
        iflat = logits.view(-1)
        tflat = targets.view(-1)
        intersection = (iflat * tflat).sum()
        return 1 - ((2. * intersection + smooth) /(iflat.sum() + tflat.sum() + smooth))

class SoftDicescore(nn.Module):
    '''
    Soft Dice Loss
    '''        
    def __init__(self, weight=None, size_average=True):
        super(SoftDicescore, self).__init__()

    def forward(self, logits, targets):
        smooth = 1.
        logits = F.sigmoid(logits)
        iflat = logits.view(-1)
        tflat = targets.view(-1)
        intersection = (iflat * tflat).sum()
        return ((2. * intersection + smooth) /(iflat.sum() + tflat.sum() + smooth))        

class W_bce(nn.Module):
    '''
    weighted crossentropy per image
    '''
    def __init__(self, weight=None, size_average=True):
        super(W_bce, self).__init__()
        
    def forward(self, logits, targets):
        eps = 1e-6
        total_size = targets.view(-1).size()[0]
        #print "total_size", total_size
        ones_size = torch.sum(targets.view(-1,1)).item()
        #print "one_size", ones_size
        zero_size = total_size - ones_size
        #print "zero_size", zero_size
        #assert total_size == (ones_size + zero_size)
        #print "crossed assertion"
        loss_1 = torch.mean(-(targets.view(-1)* ( total_size/ones_size) * torch.log(torch.clamp(F.sigmoid(logits).view(-1),eps,1.-eps))))#.sum(axis=1)
        #print "crossed loss1"
        loss_0 = torch.mean(-((1.-targets.view(-1))* ( total_size/zero_size) * torch.log((1.-torch.clamp(F.sigmoid(logits).view(-1),eps,1.-eps)))))#.sum(axis=1)
        #print "crossed loss0"
        return loss_1 + loss_0

      
class InvSoftDiceLoss(nn.Module):
    '''
    Inverted Soft Dice Loss
    '''
       
    def __init__(self, weight=None, size_average=True):
        super(InvSoftDiceLoss, self).__init__()

    def forward(self, logits, targets):
        smooth = 1.
        logits = F.sigmoid(logits)
        iflat = 1-logits.view(-1)
        tflat = 1-targets.view(-1)
        intersection = (iflat * tflat).sum()    
        return 1 - ((2. * intersection + smooth) /(iflat.sum() + tflat.sum() + smooth))

class InvSoftDicescore(nn.Module):
    '''
    Inverted Soft Dice Loss
    '''
       
    def __init__(self, weight=None, size_average=True):
        super(InvSoftDicescore, self).__init__()

    def forward(self, logits, targets):
        smooth = 1.
        logits = F.sigmoid(logits)
        iflat = 1-logits.view(-1)
        tflat = 1-targets.view(-1)
        intersection = (iflat * tflat).sum()    
        return ((2. * intersection + smooth) /(iflat.sum() + tflat.sum() + smooth))

class custom_loss(nn.Module):
    '''
    custom loss
    '''
    def __init__(self, weight=None, size_average=True):
        super(custom_loss, self).__init__()
        
    def forward(self, logits, targets):
        #eps = 1e-8
        loss_inv_dice = InvSoftDicescore()
        loss_dice = SoftDicescore()
        total_size = targets.view(-1).size()[0]
        #print "total_size", total_size
        ones_size = torch.sum(targets.view(-1,1)).item()
        #print "one_size", ones_size
        #zero_size = total_size - ones_size
        #print "zero_size", zero_size
        #assert total_size == (ones_size + zero_size)
        #print "crossed assertion"
        th = 0.2 * total_size
        #loss_1 = torch.mean(-(targets.view(-1)* ( total_size/ones_size) * torch.log(torch.clamp(F.sigmoid(logits).view(-1),eps,1.-eps))))#.sum(axis=1)
        #print "crossed loss1"
        #loss_0 = torch.mean(-((1.-targets.view(-1))* ( total_size/zero_size) * torch.log((1.-torch.clamp(F.sigmoid(logits).view(-1),eps,1.-eps)))))#.sum(axis=1)
        #print "crossed loss0"
        if(ones_size > th):
            return (- 0.2*torch.log(loss_dice(logits,targets))-0.8*torch.log(loss_inv_dice(logits, targets)))
        else:
            return(-0.8*torch.log(loss_dice(logits, targets))-0.2*torch.log(loss_inv_dice(logits, targets)))

        
        
    

transformations_train = transforms.Compose([transforms.ToTensor()])
    
transformations_val = transforms.Compose([transforms.ToTensor()])     
                                      
    
  
from data_loader import NucleiSeg
from data_loader import NucleiSegVal
train_set = NucleiSeg(transforms = transformations_train)  
batch_size = 4   
num_epochs = 100
    
class Average(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.sum += val
        self.count += n

    @property
    def avg(self):
        return self.sum / self.count

def train():
    cuda = torch.cuda.is_available()
    net = UNet11(1,32,3)#UNet(3,1)
    if cuda:
        net = net.cuda()
    # net.load_state_dict(torch.load('/home/sahyadri/vggunet_he/New_Weights/cp_all_lr_525_0.0856291705689.pth.tar'))
    criterion1 = nn.BCEWithLogitsLoss().cuda()
    criterion2 = SoftDiceLoss().cuda()
    criterion3 = InvSoftDiceLoss().cuda()
    criterion4 = W_bce().cuda()
    criterion5 = custom_loss()
    
    
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)
    scheduler = MultiStepLR(optimizer, milestones=[25,50,75,100], gamma=0.5)
    
    
    print("preparing training data ...")
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    print("done ...")
    if not(os.path.exists('./Log_initial_setting_after25.csv')):
        with open('Log_initial_setting_after25.csv', 'w') as logFile:
            FileWriter = csv.writer(logFile)
            FileWriter.writerow(['Epochs', 'TrainLoss', 'ValLoss', 'ValLoss2', 'ValLoss3', 'ValLoss4'])
    else:            
        with open('Log_initial_setting_after25.csv', 'a') as logFile:
            FileWriter = csv.writer(logFile)
            FileWriter.writerow(['Epochs', 'TrainLoss', 'ValLoss', 'ValLoss2', 'ValLoss3', 'ValLoss4'])    

    val_set = NucleiSegVal(transforms = transformations_val)   
    val_loader = DataLoader(val_set, batch_size=batch_size)
    for epoch in tqdm(range(num_epochs)):
        scheduler.step()
        train_loss = Average()
        net.train()
        for i, (images, masks) in tqdm(enumerate(train_loader)):
            images = Variable(images)
            masks = Variable(masks)
            if cuda:
                images = images.cuda()
                masks = masks.cuda()

            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion1(outputs, masks) + criterion5(outputs, masks) #+ criterion2(outputs, masks)
            loss.backward()
            optimizer.step()
            train_loss.update(loss.item(), images.size(0))
        
        #val_loss = Average()
        val_loss1 = Average()
        val_loss2 = Average()
        val_loss3 = Average()
        #val_loss4 = Average()
        net.eval()
        for images, masks,_ in tqdm(val_loader):
            images = Variable(images)
            masks = Variable(masks)
            if cuda:
                images = images.cuda()
                masks = masks.cuda()

            outputs = net(images)
            vloss1 = criterion1(outputs, masks)
            vloss2 = criterion2(outputs, masks)  
            vloss3 = criterion3(outputs, masks) #+ criterion2(outputs, masks)
            #vloss4 = criterion4(outputs, masks)
            #vloss = vloss2 + vloss3
            val_loss1.update(vloss1.item(), images.size(0))
            val_loss2.update(vloss2.item(), images.size(0))
            val_loss3.update(vloss3.item(), images.size(0))
            #val_loss4.update(vloss4.item(), images.size(0))

        #print("Epoch {}, Loss: {}, Validation Loss1: {}, Validation Loss2: {}, Validation Loss3: {}, Validation Loss4: {}".format(epoch+1, train_loss.avg, val_loss1.avg, val_loss2.avg, val_loss3.avg, val_loss4.avg))
        print("Epoch {}, Loss: {}, Validation Loss1: {}, Validation Loss2: {}, Validation Loss3: {}".format(epoch+1, train_loss.avg, val_loss1.avg, val_loss2.avg, val_loss3.avg))
        with open('Log_initial_setting_after25.csv', 'a') as logFile:
            FileWriter = csv.writer(logFile)
            FileWriter.writerow([epoch+1, train_loss.avg, val_loss1.avg, val_loss2.avg, val_loss3.avg])        
        
        torch.save(net.state_dict(), 'New_Weights/cp_all_lr_5_after25{}_{}.pth.tar'.format(epoch+1, val_loss2.avg))    
            
    return net

def test(model):
    model.eval()



if __name__ == "__main__":
    train()
