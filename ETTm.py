
from autoformer import AutoFormer
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset,DataLoader
from sklearn.preprocessing import StandardScaler
import torch
import numpy as np
import math
class ETTdataset(Dataset):
    def __init__(self,input_length = 96,preds_length = 96 ,path = 'ETTm2.csv',train = 'train',split = (0.6,0.2,0.2)):
        super().__init__()
        df = pd.read_csv(path)
        #time series feature
        df_stamp = pd.DataFrame()
        df['date'] = pd.to_datetime(df.date)
        df_stamp['month'] = df.date.apply(lambda row:row.month,1)
        df_stamp['weekday'] = df.date.apply(lambda row:row.weekday(),1)
        df_stamp['hour'] = df.date.apply(lambda row:row.hour,1)
        df_stamp['day'] = df.date.apply(lambda row:row.day,1)
        df_stamp = df_stamp[['month','weekday','hour','day']].values
        
        # scaled data
        cols = ['HUFL','HULL','MUFL','MULL','LUFL','LULL','OT']
        df = df[cols]
        self.scale_fn = StandardScaler()
        train_length = int((len(df)-input_length-preds_length)*split[0])
        valid_length = int((len(df)-input_length-preds_length)*split[1])
        train_data = df[:train_length].values
        self.scale_fn.fit(train_data)
        df = self.scale_fn.transform(df.values)
        
        if train == 'train':
            self.df = df[:train_length]
            self.df_stamp = df_stamp[:train_length]
        elif train == 'valid':
            self.df = df[train_length:train_length+valid_length]
            self.df_stamp = df_stamp[train_length:train_length+valid_length]
        else:
            self.df = df[train_length+valid_length:]
            self.df_stamp = df_stamp[train_length+valid_length:]
            
        self.input_length = input_length
        self.preds_length = preds_length
        self.train = train
        
        self.df = torch.tensor(self.df,device = 'cuda').float()
        self.df_stamp = torch.tensor(self.df_stamp,device = 'cuda')
        
        
    def __inverse_transform__(self,X):
        return self.scale_fn.inverse_transform(X)
    
    def __len__(self):
        return len(self.df) - self.input_length - self.preds_length + 1
    
    def __getitem__(self,index):
        X = self.df[index:index+self.input_length]
        X_stamp = self.df_stamp[index:index+self.input_length]
        
        y = self.df[index+self.input_length:index+self.input_length+self.preds_length]
        y_stamp = self.df_stamp[index+self.input_length:index+self.input_length+self.preds_length]
        
        return X,X_stamp,y,y_stamp


epoch = 15
batch_size = 32
lr = 1e-4
input_length = 96
pred_length = 96

train_dataset = ETTdataset(input_length,pred_length,train = 'train')
valid_dataset = ETTdataset(input_length,pred_length,train = 'valid')
test_dataset = ETTdataset(input_length,pred_length,train = 'test')


train_dataloader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True,drop_last=False)
valid_dataloader = DataLoader(valid_dataset,batch_size=batch_size,shuffle=True,drop_last=False)
test_dataloader = DataLoader(test_dataset,batch_size=batch_size,shuffle=True,drop_last=False)


model = AutoFormer(pred_length,2,1,nfeat=7,nembed=8,nhid = 512,nhead = 8).cuda()
loss_fn = torch.nn.MSELoss().cuda()
optim = torch.optim.Adam(model.parameters(),lr=lr)
schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optim,T_max=len(train_dataloader)*epoch,eta_min=1e-7)


def train_per_epoch(dataloader):
    model.train()
    loss_report = 0.
    bar = tqdm(dataloader)
    for X,X_stamp,y,y_stamp in bar:
        optim.zero_grad()
        X = X.float().cuda()
        y = y.float().cuda() 
        pred = model(X,X_stamp,y_stamp)
        loss = loss_fn(y,pred)
        loss.backward()
        optim.step()
        schedule.step()
        loss_report += loss.clone().detach().cpu().item()
        bar.set_description("Train loss {:.4f} ".format(loss.clone().detach().cpu().item()))
    return loss_report/len(dataloader)

def valid_per_epoch(dataloader):
    model.eval()
    loss_report = 0.
    bar = tqdm(dataloader)
    preds = []
    labels = []
    for X,X_stamp,y,y_stamp in bar:
        with torch.no_grad():
            X = X.float().cuda()
            y = y.float().cuda() 
            pred = model(X,X_stamp,y_stamp)
            loss = loss_fn(y,pred)
            loss_report += loss.clone().detach().cpu().item()
            bar.set_description("Valid loss {:.4f}".format(loss.clone().detach().cpu().item()))
            
            preds.append(pred.clone().detach().cpu().numpy())
            labels.append(y.clone().detach().cpu().numpy())
    
    preds = np.concatenate(preds,axis = 0)
    labels = np.concatenate(labels,axis = 0)
    loss_report = {
        'loss': loss_report/len(dataloader),
        'mae':np.abs(preds-labels).mean(),
        'rmse':math.sqrt(((preds-labels)**2).mean())
    }
    return loss_report

for e in range(epoch):
    loss_train = train_per_epoch(train_dataloader)
    print('train loss is {:.4f}'.format(loss_train))    
    report = valid_per_epoch(valid_dataloader)
    print('valid loss is {:.4f} rmse is {:.4f} mae is {:4f}'.format(report['loss'],report['rmse'],report['mae']))
    report = valid_per_epoch(test_dataloader)
    print('test loss is {:.4f} rmse is {:.4f} mae is {:4f}'.format(report['loss'],report['rmse'],report['mae']))