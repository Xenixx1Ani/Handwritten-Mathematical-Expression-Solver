import torch
import numpy
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import os
from PIL import Image
from torchvision import transforms


class InfixClassifier(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv1 = nn.Conv2d(1,32,5)
    self.layer1 = nn.Linear(12*12*32,32)
    #self.layer2 = nn.Linear(32,32)
    #self.layer3 = nn.Linear(32,32)
    self.output = nn.Linear(32,2)

  def forward(self,x):
    x = F.relu(self.conv1(x))
    x = F.max_pool2d(x,2,2)
    x = x.view(-1,12*12*32)
    x=F.relu(self.layer1(x))
    #x=F.relu(self.layer2(x))
    #x=F.relu(self.layer3(x))
    x = self.output(x)
    return torch.sigmoid(x)

class OperatorClassifier(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv1 = nn.Conv2d(1,32,5)
    self.conv2 = nn.Conv2d(32,64,5)
    self.layer1 = nn.Linear(4*4*64,32)
    #self.layer2 = nn.Linear(32,32)
    #self.layer3 = nn.Linear(32,32)
    self.output = nn.Linear(32,4)

  def forward(self,x):
    x = F.relu(self.conv1(x))
    x = F.max_pool2d(x,2,2)
    x = F.relu(self.conv2(x))
    x = F.max_pool2d(x,2,2)
    x = x.view(-1,4*4*64)
    x = F.relu(self.layer1(x))
    #x=F.relu(self.layer2(x))
    #x=F.relu(self.layer3(x))
    x = self.output(x)
    return torch.sigmoid(x)

class DigitsClassifier(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv1 = nn.Conv2d(1,32,5)
    self.conv2 = nn.Conv2d(32,64,5)
    self.layer1 = nn.Linear(4*4*64,64)
    self.layer2 = nn.Linear(64,32)
    #self.layer3 = nn.Linear(32,32)
    self.output = nn.Linear(32,10)

  def forward(self,x):
    x = F.relu(self.conv1(x))
    x = F.max_pool2d(x,2,2)
    x = F.relu(self.conv2(x))
    x = F.max_pool2d(x,2,2)
    x = x.view(-1,4*4*64)
    x = F.relu(self.layer1(x))
    x=F.relu(self.layer2(x))
    #x=F.relu(self.layer3(x))
    x = self.output(x)
    return torch.sigmoid(x)


if torch.cuda.is_available():  
  dev = "cuda:0" 
else:  
  dev = "cpu"  


pipclassifier = InfixClassifier().to(torch.device(dev))
digitai = DigitsClassifier().to(torch.device(dev))
operatorsai = OperatorClassifier().to(torch.device(dev))
pipclassifier.load_state_dict(torch.load("./pipclassifier.pt"))
digitai.load_state_dict(torch.load('./digitai.pt'))
operatorsai.load_state_dict(torch.load('./operatorsai.pt'))
pipclassifier.eval()
digitai.eval()
operatorsai.eval()

folderpath = str(input("Enter path to folder:"))

p = transforms.Compose([transforms.Resize(28)])

annotations_values = pd.DataFrame(columns=['Image_Name','Value'])
i=0
for filename in os.listdir(folderpath):
    f = os.path.join(folderpath,filename)
    img = Image.open(f)
    img = p(img)
    img1 = (transforms.ToTensor()(numpy.array(img.crop((0,0,28,28))))).to(torch.device(dev))
    img2 = (transforms.ToTensor()(numpy.array(img.crop((28,0,56,28))))).to(torch.device(dev))
    img3 = (transforms.ToTensor()(numpy.array(img.crop((56,0,84,28))))).to(torch.device(dev))
    if torch.argmax(pipclassifier(img1.view(1,1,28,28))) == 1:
        no1=torch.argmax(digitai(img2.view(1,1,28,28)))
        no2=torch.argmax(digitai(img3.view(1,1,28,28)))
        opid=torch.argmax(operatorsai(img1.view(1,1,28,28)))
    elif torch.argmax(pipclassifier(img2.view(1,1,28,28))) == 1:
        no1=torch.argmax(digitai(img1.view(1,1,28,28)))
        no2=torch.argmax(digitai(img3.view(1,1,28,28)))
        opid=torch.argmax(operatorsai(img2.view(1,1,28,28)))
    else:
        no1=torch.argmax(digitai(img1.view(1,1,28,28)))
        no2=torch.argmax(digitai(img2.view(1,1,28,28)))
        opid=torch.argmax(operatorsai(img3.view(1,1,28,28)))
    no1=no1.cpu()
    no2=no2.cpu()
    if opid==0:
        value = no1+no2
    elif opid == 1:
        value = no1-no2
    elif opid == 2:
        value = no1*no2
    else:
      if no2!=0:
        value = no1//no2
      else:
        value=no1
    value = value.item()
    annotations_values.loc[i]=[filename,value]
    i+=1

annotations_values.to_csv('AI_Hayasaka_2.csv',index=False)

