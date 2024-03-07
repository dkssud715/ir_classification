import torch
import torchvision
from torchvision import transforms # 이미지 데이터 transform
from torch.utils.data import DataLoader # 이미지 데이터 로더
from PIL import Image
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import os
import glob
from Dataset import GasVidDataset
from video_dataset import  VideoFrameDataset, ImglistToTensor
from ResNet import ResNet50, ResNet18, ResNet101, ResNet152
from config import Config
import time

device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(777)
if device == 'cuda': 
  torch.cuda.manual_seed_all(777)

train_path = '/home/linux_hy/anaconda3/envs/mc/Gas/Images2'
test_path = '/home/linux_hy/anaconda3/envs/mc/Gas/Images2'
annotation_train = '/home/linux_hy/anaconda3/envs/mc/Gas/Images2/train.txt'
annotation_test = '/home/linux_hy/anaconda3/envs/mc/Gas/Images2/test.txt'

preprocess = transforms.Compose([
    ImglistToTensor(),
    transforms.Resize((224,224)),
])
#dataset = GasVidDataset()
traindataset = VideoFrameDataset(
    root_path=train_path,
    annotationfile_path=annotation_train,
    num_segments=1,
    frames_per_segment=5,
    imagefile_template='{:05d}.jpg',
    transform=preprocess,
    test_mode=False
)
trainloader = DataLoader(traindataset, batch_size=128, shuffle=True, num_workers=2)
testdataset = VideoFrameDataset(
    root_path=test_path,
    annotationfile_path=annotation_test,
    num_segments=1,
    frames_per_segment=5,
    imagefile_template='{:05d}.jpg',
    transform=preprocess,
    test_mode=False
)
testloader = DataLoader(testdataset, batch_size=128, shuffle=False, num_workers=2)



# move the input and model to GPU for speed if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

resnett = ResNet18().to(device)

lr = 1e-6
optimizer = torch.optim.Adam(resnett.parameters(), lr=lr)

class train_test():
    def __init__(self):
        # 파라미터 인자-+9+6
        self.trainloader = trainloader
        self.testloader = testloader
        self.model = resnett
        self.device = device
        self.optimizer = optimizer
        #self.lr_schedular = optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.5)
        self.criterion = nn.CrossEntropyLoss().to(device)
        self.globaliter = 0

    def train(self, epochs = 30, log_interval = 1000, test_interval = 1):
        self.model.train()
        self.globaliter = 0
        correct_train = 0
        total_train = 0 
        for epoch in range(1, epochs + 1 ):  # epochs 루프
            running_loss = 0.0
            for i, data in enumerate(self.trainloader, 0): # batch 루프
                # get the inputs            
                #self.lr_schedular.step()
                inputs, labels = data # input data, label 분리
                #print(inputs.shape)
                inputs = inputs.to(self.device).squeeze(2)
                labels = labels.to(self.device)
                # print(labels)
                # 가중치 초기화 -> 이전 batch에서 계산되었던 가중치를 0으로 만들고 최적화 진행
                self.optimizer.zero_grad() 

                # forward + backward + optimize
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_train += labels.size(0)
                correct_train += (predicted == labels).sum().item()
                # 30 iteration마다 acc & loss 출력
                if self.globaliter % log_interval ==0 : # i는 1에포크의 iteration
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tlearningLoss: {:.6f}\twhole_loss: {:.6f} Accuracy: {}/{}({:.0f}%)\n '.format(
                        epoch, i*len(inputs), len(self.trainloader.dataset),
                        100. * i*len(inputs) / len(self.trainloader.dataset), 
                        running_loss / log_interval,
                        loss.item(), correct_train, total_train, 100 * correct_train/total_train))
                    running_loss = 0.0
                    correct_train = 0
                    total_train = 0
                self.globaliter += 1
                #with train_summary_writer.as_default():
                #    summary.scalar('loss', loss.item() , step = self.globaliter)
            # if epoch % test_interval == 0 :
            with torch.no_grad():
                self.model.eval()
                correct = 0
                total = 0
                test_loss = 0
                acc = []
                for k, data in enumerate(self.testloader, 0):
                    images, labels = data
                    images = images.to(self.device).squeeze(2)
                    labels = labels.to(self.device)
                    outputs = self.model(images)

                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    test_loss += self.criterion(outputs, labels).item()
                    acc.append(100 * correct/total)

                print('\nTest set : Epoch {}, Average loss:{:.4f}, Accuracy: {}/{}({:.0f}%)\n'.format(
                    epoch, test_loss, correct, total, 100 * correct/total
                ))
                    #with test_summary_writer.as_default():
                    #    summary.scalar('loss', test_loss , step = self.globaliter)
                    #    summary.scalar('accuracy', 100 * correct/total , step = self.globaliter)  
        ##                      if acc [k] > 60 and acc[k] > acc[k-1]:
        #                         torch.save({
        #                                     'epoch': epoch,
        #                                     'model_state_dict': self.model.state_dict(),
        #                                     'optimizer_state_dict': self.optimizer.state_dict(),
        #                                     'loss': test_loss
        #                                     }, PATH)
                                
t1 = train_test()
t1.train()
