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



if torch.cuda.is_available():  
  dev = "cuda:0" 
else:  
  dev = "cpu"  


pipclassifier = InfixClassifier().to(torch.device(dev))
pipclassifier.load_state_dict(torch.load("./pipclassifier.pt"))
pipclassifier.eval()

folderpath = str(input("Enter path to folder:"))

p = transforms.Compose([transforms.Resize(28)])

annotations_labels = pd.DataFrame(columns=['Image_Name','Label'])
i = 0

for filename in os.listdir(folderpath):
    f = os.path.join(folderpath,filename)
    img = Image.open(f)
    img = p(img)
    img1 = (transforms.ToTensor()(numpy.array(img.crop((0,0,28,28))))).to(torch.device(dev))
    img2 = (transforms.ToTensor()(numpy.array(img.crop((28,0,56,28))))).to(torch.device(dev))
    img3 = (transforms.ToTensor()(numpy.array(img.crop((56,0,84,28))))).to(torch.device(dev))
    if torch.argmax(pipclassifier(img1.view(1,1,28,28))) == 1:
        lst = [filename,'prefix']
    elif torch.argmax(pipclassifier(img2.view(1,1,28,28))) == 1:
        lst = [filename,'infix']
    else:
        lst = [filename,'postfix']
    annotations_labels.loc[i] = lst
    i+=1

annotations_labels.to_csv('AI_Hayasaka_1.csv',index=False)

    
    
    
    

