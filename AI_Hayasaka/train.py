import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import pandas as pd
from tqdm import tqdm
import random

if torch.cuda.is_available():  
  dev = "cuda:0" 
else:  
  dev = "cpu"  

def get_acc(net, loader):
  total = 0
  correct = 0

  for images, labels in loader:
    images,labels = images.to(torch.device(dev)),labels.to(torch.device(dev))
    batch_size = images.shape[0]

    #images = images.view(batch_size, 28*28)
    output = net(images)
    
    predicted = torch.argmax(output, dim=1)
    #print(target)
    #print(predicted)
    correct += torch.sum(torch.argmax(labels, dim=1) == predicted)
    total += batch_size
  return correct / total

def train_ai(ai,dataset,epoch_no,step_size):
  criterion = nn.MSELoss()
  optimizer = torch.optim.Adam(ai.parameters(), lr=1e-3)
  scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)
  
  for epoch in range(epoch_no):
    avg_loss=0
    num_iters=0
    Training_dataset = []
    min = 50000
    for lst in dataset:
      if len(lst)<min:
        min=len(lst)
      random.shuffle(lst)
    for i in range(min):
      for j in range(len(dataset)):
        Training_dataset.append((dataset[j][i],torch.eye(len(dataset))[j]))
    Training_loader = torch.utils.data.DataLoader(Training_dataset, batch_size=20, shuffle=True)

    for images, targets in tqdm(Training_loader):
      images,targets = images.to(torch.device(dev)),targets.to(torch.device(dev))
      optimizer.zero_grad()
      batch_size = images.shape[0]

      output = ai(images)

      loss = criterion(output, targets)

      loss.backward()
      optimizer.step()

      avg_loss += loss.item()
      num_iters += 1
    scheduler.step()
    print("Loss: {}".format(avg_loss / num_iters))
    #print("Train acc: {}".format(get_acc(ai, Training_loader)))
  torch.cuda.empty_cache()

def isuncertain(output, minconfidence,factor):
  flag = False
  maxm = 0
  for val in output[0]:
    if val>maxm:
      maxm=val
  if maxm< minconfidence:
    flag = True
  for item in output[0]:
    if item == maxm:
      continue
    if item/maxm>factor:
      flag=True
  return flag

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

class addsubmulClassifier(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv1 = nn.Conv2d(1,32,5)
    self.layer1 = nn.Linear(12*12*32,32)
    #self.layer2 = nn.Linear(32,32)
    #self.layer3 = nn.Linear(32,32)
    self.output = nn.Linear(32,3)

  def forward(self,x):
    x = F.relu(self.conv1(x))
    x = F.max_pool2d(x,2,2)
    x = x.view(-1,12*12*32)
    x = F.relu(self.layer1(x))
    #x=F.relu(self.layer2(x))
    #x=F.relu(self.layer3(x))
    x = self.output(x)
    return torch.sigmoid(x)

class partialdigitsClassifier(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv1 = nn.Conv2d(1,32,5)
    self.conv2 = nn.Conv2d(32,64,5)
    self.layer1 = nn.Linear(4*4*64,64)
    self.layer2 = nn.Linear(64,32)
    self.output = nn.Linear(32,4)
    
  def forward(self,x):
    x = F.relu(self.conv1(x))
    x = F.max_pool2d(x,2,2)
    x = F.relu(self.conv2(x))
    x = F.max_pool2d(x,2,2)
    x = x.view(-1,4*4*64)
    x = F.relu(self.layer1(x))
    x = F.relu(self.layer2(x))
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



annotations = pd.read_csv("./SoML-50/annotations.csv")
annotations['Label'] = annotations['Label'].replace({'prefix' : 0})
annotations['Label'] = annotations['Label'].replace({'infix' : 1})
annotations['Label'] = annotations['Label'].replace({'postfix' : 2})

Samples_Multiplication = []
Samples_Addition = []
Samples_Subtraction = []    
Operator_List = []
List_primes = [11,13,17]
Dataset_Label = [[],[]]
Samples_Digits = {0:[], 5:[],7:[],8:[],9:[]}

p = transforms.Compose([transforms.Resize(28)])
Dataset_all = []

for i in range(1,45001):
    img = Image.open("./SoML-50/data/{}.jpg".format(i))
    img = p(img)
    img1 = (transforms.ToTensor()(numpy.array(img.crop((0,0,28,28))))).to(torch.device(dev))
    img2 = (transforms.ToTensor()(numpy.array(img.crop((28,0,56,28))))).to(torch.device(dev))
    img3 = (transforms.ToTensor()(numpy.array(img.crop((56,0,84,28))))).to(torch.device(dev))
    Dataset_all.append([img1,img2,img3])
    val = annotations.at[i-1,'Value']
    if(annotations.at[i-1,'Label'] == 0):
        Dataset_Label[1].append(img1)
        Dataset_Label[0].append(img2)
        Dataset_Label[0].append(img3)
        Operator_List.append(img1)
        if(val<0):
            Samples_Subtraction.append(img1)
        elif(val>18):
            Samples_Multiplication.append(img1)
        elif(val in List_primes):
            Samples_Addition.append(img1)
        
        if val == 81:
            Samples_Digits[9]+=[img2,img3]
        elif val ==64:
            Samples_Digits[8]+=[img2,img3]
        elif val == 49:
            Samples_Digits[7]+=[img2,img3]
        elif val == 25:
            Samples_Digits[5]+=[img2,img3]
        elif val == -9:
            Samples_Digits[0].append(img2)
            Samples_Digits[9].append(img3)

    elif(annotations.at[i-1,'Label']==1):
        Dataset_Label[0].append(img1)
        Dataset_Label[1].append(img2)
        Dataset_Label[0].append(img3)
        Operator_List.append(img2)
        if(val<0):
            Samples_Subtraction.append(img2)
        elif(val>18):
            Samples_Multiplication.append(img2)
        elif(val in List_primes):
            Samples_Addition.append(img2)
        
        if val == 81:
            Samples_Digits[9]+=[img1,img3]
        elif val ==64:
            Samples_Digits[8]+=[img1,img3]
        elif val == 49:
            Samples_Digits[7]+=[img1,img3]
        elif val == 25:
            Samples_Digits[5]+=[img1,img3]
        elif val == -9:
            Samples_Digits[0].append(img1)
            Samples_Digits[9].append(img3)
        
    else:
        Dataset_Label[0].append(img1)
        Dataset_Label[0].append(img2)
        Dataset_Label[1].append(img3)
        Operator_List.append(img3)
        if(val<0):
            Samples_Subtraction.append(img3)
        elif(val>18):
            Samples_Multiplication.append(img3)
        elif(val in List_primes):
            Samples_Addition.append(img3)
        
        if val == 81:
            Samples_Digits[9]+=[img1,img2]
        elif val ==64:
            Samples_Digits[8]+=[img1,img2]
        elif val == 49:
            Samples_Digits[7]+=[img1,img2]
        elif val == 25:
            Samples_Digits[5]+=[img1,img2]
        elif val == -9:
            Samples_Digits[0].append(img1)
            Samples_Digits[9].append(img2)
    if i%10000 == 0:
        print(".")
    

pipclassifier = InfixClassifier().to(torch.device(dev))
train_ai(pipclassifier,Dataset_Label,5,5)

Dataset_addsubmul = [Samples_Addition,Samples_Subtraction,Samples_Multiplication] 

asmclassifier = addsubmulClassifier().to(torch.device(dev))
train_ai(asmclassifier,Dataset_addsubmul,15,10)

Samples_Division = []
for op in Operator_List:
  op = op.to(torch.device(dev))
  prediction = asmclassifier(op.view(1,1,28,28))
  if isuncertain(prediction,1e-4,2e-1): 
    Samples_Division.append(op)
print("number of division samples generated by asmclassifier: ",len(Samples_Division))

Dataset_partialdigits =  [Samples_Digits[5]]+[Samples_Digits[7]]+[Samples_Digits[8]]+[Samples_Digits[9]]
pdclassifier = partialdigitsClassifier().to(torch.device(dev))
train_ai(pdclassifier,Dataset_partialdigits,40,20)

Extra_Div = []
for i in range(0,45000):
  if annotations.at[i,'Value'] !=1:
    continue
  img1 = Dataset_all[i][0]
  img2 = Dataset_all[i][1]
  img3 = Dataset_all[i][2]
  label = annotations.at[i,'Label']
  if label ==0:
    tnsr1=pdclassifier(img2.view(1,1,28,28))
    tnsr2=pdclassifier(img3.view(1,1,28,28))
    if isuncertain(tnsr1,5e-1,5e-4) or isuncertain(tnsr2,5e-1,5e-4):
      continue
    if torch.argmax(tnsr1) == torch.argmax(tnsr2):
      Extra_Div.append(img1)
  elif label == 1:
    tnsr1=pdclassifier(img1.view(1,1,28,28))
    tnsr2=pdclassifier(img3.view(1,1,28,28))
    if isuncertain(tnsr1,5e-1,5e-4) or isuncertain(tnsr2,5e-1,5e-4):
      continue
    if torch.argmax(tnsr1) == torch.argmax(tnsr2):
      Extra_Div.append(img2)
  else:
    tnsr1=pdclassifier(img1.view(1,1,28,28))
    tnsr2=pdclassifier(img2.view(1,1,28,28))
    if isuncertain(tnsr1,5e-1,5e-4) or isuncertain(tnsr2,5e-1,5e-4):
      continue
    if torch.argmax(tnsr1) == torch.argmax(tnsr2):
      Extra_Div.append(img3)
#print("number of division samples generated by pdclassifier: ",len(Extra_Div))

Samples_Division += Extra_Div
Dataset_operators = []
Dataset_operators = Dataset_addsubmul + [Samples_Division]
opclassifier = OperatorClassifier().to(torch.device(dev))
train_ai(opclassifier,Dataset_operators,30,20)

Samples_Digits_Full = []
for i in range(10):
  Samples_Digits_Full.append([])
for i in range(0,45000):
  img1 = Dataset_all[i][0]
  img2 = Dataset_all[i][1]
  img3 = Dataset_all[i][2]
  label = annotations.at[i,'Label']
  value = annotations.at[i,'Value']
  if label == 0:
    operatortensor = opclassifier(img1.view(1,1,28,28))
    digitonetensor = pdclassifier(img2.view(1,1,28,28))
    digitoneimg = img2
    digittwoimg = img3
    digittwotensor = pdclassifier(img3.view(1,1,28,28))
  elif label == 1:
    operatortensor = opclassifier(img2.view(1,1,28,28))
    digitonetensor = pdclassifier(img1.view(1,1,28,28))
    digittwotensor = pdclassifier(img3.view(1,1,28,28))
    digitoneimg = img1
    digittwoimg = img3
  else:
    operatortensor = opclassifier(img3.view(1,1,28,28))
    digitonetensor = pdclassifier(img1.view(1,1,28,28))
    digittwotensor = pdclassifier(img2.view(1,1,28,28))
    digitoneimg = img1
    digittwoimg = img2
  
  if isuncertain(operatortensor,5e-1,1e-6):
    continue
  operatorid = torch.argmax(operatortensor)
  
  unknowndigit = 0
  cntuncertain = 0
  
  if isuncertain(digitonetensor,7e-1,5e-4):
    unknowndigit = 1
    cntuncertain +=1
  
  if isuncertain(digittwotensor,7e-1,5e-4):
    unknowndigit = 2
    cntuncertain +=1
  
  if cntuncertain==2:
    continue
  
  digitidtonumber = {0:5,1:7,2:8,3:9}
  if unknowndigit == 1:
    knownnumber = digitidtonumber[torch.argmax(digittwotensor).item()]
    if operatorid == 0:
      if value-knownnumber>=0 and value-knownnumber<10:
        Samples_Digits_Full[value-knownnumber].append(digitoneimg)
    elif operatorid == 1:
      if value+knownnumber>=0 and value+knownnumber<10:
        Samples_Digits_Full[value+knownnumber].append(digitoneimg)
    elif operatorid == 2:
      if knownnumber!=0 and value//knownnumber in [0,1,2,3,4,5,6,7,8,9]:
        Samples_Digits_Full[value//knownnumber].append(digitoneimg)
    else:
      if knownnumber*value>=0 and knownnumber*value<10:
        Samples_Digits_Full[knownnumber*value].append(digitoneimg)
  elif unknowndigit == 2:
    knownnumber = digitidtonumber[torch.argmax(digitonetensor).item()]
    if operatorid == 0:
      if value-knownnumber>=0 and value-knownnumber<10:
        Samples_Digits_Full[value-knownnumber].append(digittwoimg)
    elif operatorid == 1:
      if knownnumber-value>=0 and knownnumber-value<10:
        Samples_Digits_Full[knownnumber-value].append(digittwoimg)
    elif operatorid == 2:
      if knownnumber!=0 and value//knownnumber in [0,1,2,3,4,5,6,7,8,9]:
        Samples_Digits_Full[value//knownnumber].append(digittwoimg)
    else:
      if value!=0 and knownnumber//value in [0,1,2,3,4,5,6,7,8,9]:
        Samples_Digits_Full[knownnumber//value].append(digittwoimg)
  else:
    no1 = digitidtonumber[torch.argmax(digitonetensor).item()]
    no2 = digitidtonumber[torch.argmax(digittwotensor).item()]
    if operatorid == 0 and no1+no2 == value:
      Samples_Digits_Full[no1].append(digitoneimg)
      Samples_Digits_Full[no2].append(digittwoimg)
    elif operatorid == 1 and no1-no2==value:
      Samples_Digits_Full[no1].append(digitoneimg)
      Samples_Digits_Full[no2].append(digittwoimg)
    elif operatorid == 2 and no1*no2==value:
      Samples_Digits_Full[no1].append(digitoneimg)
      Samples_Digits_Full[no2].append(digittwoimg)
    elif no1/no2 == value:
      Samples_Digits_Full[no1].append(digitoneimg)
      Samples_Digits_Full[no2].append(digittwoimg)
#print("\n number of samples of each digit:\n")
#for i in range(10):
#  print(i,len(Samples_Digits_Full[i]))

for i in [0,5,7,8,9]:
  Samples_Digits_Full[i] += Samples_Digits[i]
#print("Training Digit_classifier:\n\n")
digits_classifier = DigitsClassifier().to(torch.device(dev))
train_ai(digits_classifier,Samples_Digits_Full,50,40)


Dataset_digits2 = []
Dataset_operators2 = [[],[],[],[]]
for i in range(10):
  Dataset_digits2.append([])
for i in range(0,45000):
  img1 = Dataset_all[i][0]
  img2 = Dataset_all[i][1]
  img3 = Dataset_all[i][2]
  if annotations.at[i,'Label']==0:
    operatorimg = img1
    digitoneimg = img2
    digittwoimg = img3
    label = 0
  elif annotations.at[i,'Label'] == 1:
    operatorimg = img2
    digitoneimg = img1
    digittwoimg = img3
    label = 1
  else:
    operatorimg = img3
    digitoneimg = img1
    digittwoimg = img2
    label = 2
  operatortensor = opclassifier(operatorimg.view(1,1,28,28))
  numonetensor = digits_classifier(digitoneimg.view(1,1,28,28))
  numtwotensor = digits_classifier(digittwoimg.view(1,1,28,28))
  if isuncertain(operatortensor,5e-1,1e-5): #or isuncertain(numonetensor,5e-1,1e-3) or isuncertain(numtwotensor,5e-1,1e-3):
    continue
  numone = torch.argmax(numonetensor)
  numtwo = torch.argmax(numtwotensor)
  operatorid = torch.argmax(operatortensor)
  
  if operatorid == 0 and numone+numtwo==annotations.at[i,'Value']:
    Dataset_digits2[numone].append(digitoneimg)
    Dataset_digits2[numtwo].append(digittwoimg)
    Dataset_operators2[0].append(operatorimg)
  elif operatorid == 1 and numone-numtwo==annotations.at[i,'Value']:
    Dataset_digits2[numone].append(digitoneimg)
    Dataset_digits2[numtwo].append(digittwoimg)
    Dataset_operators2[1].append(operatorimg)
  elif operatorid == 2 and numone*numtwo==annotations.at[i,'Value']:
    Dataset_digits2[numone].append(digitoneimg)
    Dataset_digits2[numtwo].append(digittwoimg)
    Dataset_operators2[2].append(operatorimg)
  elif operatorid == 3 and numone/numtwo==annotations.at[i,'Value']:
    Dataset_digits2[numone].append(digitoneimg)
    Dataset_digits2[numtwo].append(digittwoimg)
    Dataset_operators2[3].append(operatorimg)
#print("\n number of samples generated for each digit:\n")
#for i in range(10):
#  print(i,len(Dataset_digits2[i]))
#print("\n number of samples generated for each operator:\n")
for i in range(4):
  Dataset_operators2[i]+=Dataset_operators[i]
  #print(i,len(Dataset_operators2[i]))

#print("Training Digitai:\n\n")
digitai = DigitsClassifier().to(torch.device(dev))
train_ai(digitai,Dataset_digits2,10,10)

#print("Training Operatorsai:\n\n")
operatorsai = OperatorClassifier().to(torch.device(dev))
train_ai(operatorsai,Dataset_operators2,10,10)

#testing accuracy of overall program

torch.save(operatorsai.state_dict(),"./operatorsai.pt")
torch.save(pipclassifier.state_dict(),"./pipclassifier.pt")
torch.save(digitai.state_dict(),"./digitai.pt")