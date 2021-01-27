#!/usr/bin/env python
# coding: utf-8

# In[1]:


import argparse
import glob
import os
import pathlib
import cv2
import numpy as np
import openslide
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision.models as models
import torchvision.transforms as transforms
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
from tifffile import memmap
from PIL import Image
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn import metrics
import logging.handlers
torch.manual_seed(777)


# In[2]:


with open('result/colon_result_temp', 'rb') as f:
    colon_result = pickle.load(f)
    
with open('result/sigmoid_result_temp', 'rb') as f:
    sigmoid_result = pickle.load(f)
    
with open('result/rectum_result_temp', 'rb') as f:
    rectum_result = pickle.load(f)


# In[3]:


info = '-'
if not os.path.exists('checkpoint/{}'.format(info)):
    os.mkdir('checkpoint/{}'.format(info))
    
log = logging.getLogger('log')
log.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
fileHandler = logging.FileHandler('checkpoint/{}/{}_log.txt'.format(info, info))
streamHandler = logging.StreamHandler()
fileHandler.setFormatter(formatter)
streamHandler.setFormatter(formatter)
log.addHandler(fileHandler)
log.addHandler(streamHandler)


# In[ ]:


data_all = np.concatenate([np.array(colon_result), np.array(sigmoid_result), np.array(rectum_result)])

# colon_result, sigmoid_result, rectum_result 안에 있는 모든 패치의 총합 데이터
data_final = np.zeros(0)
for i in range(len(data_all)):
    temp = np.array(data_all[i])
    data_final = np.concatenate([data_final, temp])


# In[ ]:


class Dataset(Dataset):
    
    def __init__(self, dataset, long_short=None, transform=False):
        self.dataset = dataset
        self.long_short = long_short
        self.transform = transform
        
    def __len__(self):
        return len(self.dataset)
        
    def __getitem__(self, idx):
        lines = self.dataset[idx]
        ID = lines['ID']
        image = lines['image']
        status = lines['status']
        duration = lines['duration']
        
        image = transforms.ToPILImage()(image)
        
        if self.transform:
            image = self.transform(image)
            
        image = transforms.RandomRotation(90)(image)
        
        image = transforms.ToTensor()(image)
        image = transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))(image)
        
        if self.long_short:
            long_short = lines['long_short']
        return {'ID' : ID ,'image' : image, 'status' : status, 'duration' : duration, 'long_short' : long_short}

def create_file(file_path,msg):
    msg = msg+'\n'
    f=open(file_path,"a")
    f.write(msg)
    f.close


# In[ ]:


# cutoff 기준 데이터 분할
output = []
for i in range(len(data_final)):
    output.append(data_final[i]['status'])
output = np.array(output)

alive_data = data_final[output==0]
dead_data = data_final[output==1]

alive_duration = []
for i in range(len(alive_data)):
    alive_duration.append(alive_data[i]['duration'])
alive_duration = np.array(alive_duration)

dead_duration = []
for i in range(len(dead_data)):
    dead_duration.append(dead_data[i]['duration'])
dead_duration = np.array(dead_duration)

cutoff = 365*5
three_up_alive = alive_data[alive_duration>=cutoff]
three_up_dead = dead_data[dead_duration>=cutoff]
three_down_dead = dead_data[dead_duration<cutoff]

Long_live_patient = np.concatenate([three_up_alive, three_up_dead])
Short_live_patient = three_down_dead

#three_years_all = np.concatenate([three_years_alive, three_years_dead])

#three_years_label = []
#for i in range(len(three_years_all)):
#    three_years_label.append(three_years_all[i]['status'])
#three_years_label = np.array(three_years_label)

for i in range(len(Long_live_patient)):
    Long_live_patient[i].update(long_short = 1)
    
for i in range(len(Short_live_patient)):
    Short_live_patient[i].update(long_short = 0) 
    
prog1 = []
for i in range(len(Long_live_patient)):
    prog1.append(Long_live_patient[i]['long_short'])
prog1 = np.array(prog1)

prog2 = []
for i in range(len(Short_live_patient)):
    prog2.append(Short_live_patient[i]['long_short'])
prog2 = np.array(prog2)

Long_Short_All = np.concatenate([Long_live_patient, Short_live_patient])
prog_All = np.concatenate([prog1, prog2])


# In[ ]:


out1 = []
out2 = []
out3 = []
out4 = []
for i in range(len(Long_live_patient)):
    out1.append(Long_live_patient[i]['ID'][:12])
    out2.append(Long_live_patient[i]['ID'])
out1 = np.array(out1)
out2 = np.array(out2)

for i in range(len(Short_live_patient)):
    out3.append(Short_live_patient[i]['ID'][:12])
    out4.append(Short_live_patient[i]['ID'])
out3 = np.array(out3)
out4 = np.array(out4)


# In[9]:


from sklearn.model_selection import KFold

out1_ = np.unique(out1.reshape(out1.shape[0], 1))
out3_ = np.unique(out3.reshape(out3.shape[0], 1))

kfold_long = KFold(n_splits=3, shuffle=True)
kfold_short = KFold(n_splits=3, shuffle=True)
kf_long = kfold_long.split(out1_, out1_)
kf_short = kfold_short.split(out3_, out3_)

X_train_Long = []
X_test_Long = []
for i, (train_idx, test_idx) in enumerate(kf_long):
    train, test = out1_[train_idx], out1_[test_idx]
    X_train_Long.append(train)
    X_test_Long.append(test)
    
X_train_Short = []
X_test_Short = []    
for i, (train_idx, test_idx) in enumerate(kf_short):
    train, test = out3_[train_idx], out3_[test_idx]
    X_train_Short.append(train)
    X_test_Short.append(test)


# In[ ]:


temp = []
for i in range(len(Long_Short_All)):
    temp.append(Long_Short_All[i]['ID'][:12])
temp = np.array(temp)


# In[ ]:


num_epochs=1

log.info("Long 환자 수 : {} ({} WSI)".format(np.unique(out1).shape[0], np.unique(out2).shape[0]))
log.info("Short 환자 수 : {} ({} WSI)".format(np.unique(out3).shape[0], np.unique(out4).shape[0]))
log.info("\n")

n_fold = 3
for fold in range(n_fold):
    create_file(f'checkpoint/{info}/result.csv', f'fold{fold+1}')
    
    
    log.info('-----  Fold {}  -----'.format(fold+1))
    
    train_idx = np.concatenate([X_train_Long[fold], X_train_Short[fold]]).reshape(-1, )
    test_idx = np.concatenate([X_test_Long[fold], X_test_Short[fold]]).reshape(-1, )
    
    temp = []
    for i in range(len(Long_Short_All)):
        temp.append(Long_Short_All[i]['ID'][:12])
    temp = np.array(temp)
    
    X_train = Long_Short_All[np.isin(temp, train_idx)]
    X_test = Long_Short_All[np.isin(temp, test_idx)]
    
    
    train_ID = []
    for i in range(len(X_train)):
        train_ID.append(X_train[i]['ID'][:12])
    train_ID = np.array(train_ID)

    test_ID = []
    for i in range(len(X_test)):
        test_ID.append(X_test[i]['ID'][:12])
    test_ID = np.array(test_ID)
    
    log.info("Train 환자 수 : {}".format(np.unique(train_ID).shape[0]))
    log.info("Test 환자 수 : {}".format(np.unique(test_ID).shape[0]))
    log.info("\n")
    
    transform = transforms.ColorJitter(brightness=0.25,#0.25
                                   contrast=0.75,#0.75
                                   saturation=0.25,#0.25
                                   hue=0.1)
    
    train_dataset = Dataset(X_train, long_short=True, transform=transform)
    train_dataloader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=16)

    test_dataset = Dataset(X_test, long_short=True, transform=transform)
    test_dataloader = DataLoader(test_dataset, batch_size=256, shuffle=True, num_workers=16)
    
    #model = models.resnet18(pretrained=True)
    #model.fc = nn.Linear(512, 2)
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(512, 2)   
    
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    optim = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999))
    criterion = torch.nn.CrossEntropyLoss().to(device)
    model = model.to(device)

    for epoch in range(num_epochs):
        log.info('Epoch  {}'.format(epoch+1))
        train_loss = []
        train_predicted = []
        train_label = []
        train_pos = []
        ID = []
        long_short = []
        label = []
    
        model.train()
        for i, batch in enumerate(train_dataloader):
            output = model(batch['image'].to(device))
        
            loss = criterion(output, batch['long_short'].to(device))
        
            optim.zero_grad()
            loss.backward()
            optim.step()
        
            loss = loss.cpu().detach().numpy()
        
            #evaluation result
            pred_label = np.argmax(output.cpu().detach().numpy(),1)
            pred_prob  = nn.Softmax(1)(output).cpu().detach().numpy()
            positive_prob = pred_prob[:, 1].tolist()   # 1이라고 예측한 확률 (True Positive?) -> auc 구하기 위한 코드
            
            train_label += batch['long_short'].tolist()          ## 1에폭의 785개 배치에 있는 6280개에 대한 true label
            train_loss += [loss.tolist()]    
            train_predicted += pred_label.tolist()
            train_pos += positive_prob
            label += pred_label.tolist()
            
            train_patient = []
            for j in range(len(batch['ID'])):
                train_patient.append(batch['ID'][j][:12])
                
            ID += train_patient
            long_short += batch['long_short'].tolist()
        
        
            batch_acc = np.sum(np.array(train_label) == np.array(train_predicted))/ len(train_label) 
            batch_mean_loss = sum(train_loss)/len(train_loss)
            fpr, tpr, thresholds = metrics.roc_curve(train_label, train_pos)
            auc = metrics.auc(fpr, tpr)

        
            if i % 300 == 0:
                log.info('Epoch : {}/{}, Batch : {}/{}, Loss : {:.3f}, ACC : {:.3f}, AUC : {:.3f}'.format(epoch+1, num_epochs, i+1, len(train_dataloader), batch_mean_loss, batch_acc, auc))
        log.info('Train Evaluation Result')
        log.info('Epoch : {}/{}, Batch : {}/{}, Loss : {:.3f}, ACC : {:.3f}, AUC:{:.3f}'.format(epoch+1, num_epochs, i+1, len(train_dataloader), batch_mean_loss, batch_acc, auc))
    
        number = []
        for i in range(len(ID)):
            number += [(train_ID == ID[i]).sum()]
        total = {'ID' : ID, 'long_short' : long_short, 'prob' : train_pos, 'pred_label' : label}    
    
        prob_mean = []
        y_true = []
        y_pred = []
        for i in range(len(np.unique(np.array(ID)))):
            index = np.array(total['ID']) == np.unique(np.array(ID))[i]
            prob_mean += [np.array(total['prob'])[index].sum() / index.sum()]
            y_true += [np.array(total['long_short'])[index][0]]
            y_pred += [np.array(total['pred_label'])[index][0]]
        
        acc = metrics.accuracy_score(y_true, y_pred)
        fpr, tpr, thresholds = metrics.roc_curve(y_true, prob_mean)
        auc = metrics.auc(fpr, tpr)    
    
    
        log.info('Train_Patient_Level_ACC : {:.3f} / Train_Patient_Level_AUC : {:.3f}'.format(acc, auc))    
        
        
        
        
        test_loss = []
        test_predicted = []
        test_label = []
        test_pos = []
        ID = []
        patch_ID = []
        long_short = []    
        label = []
             
        model.eval()
        for i, batch in enumerate(test_dataloader):
            output_ = model(batch['image'].to(device))
            loss_ = criterion(output_, batch['long_short'].to(device))
        
            #evaluation result
            pred_label = np.argmax(output_.cpu().detach().numpy(), 1)
            pred_prob  = nn.Softmax(1)(output_).cpu().detach().numpy()
            positive_prob = pred_prob[:, 1].tolist()
                
                
            test_label += batch['long_short'].tolist()   
            test_loss += [loss_.tolist()]
            test_predicted += pred_label.tolist()
            test_pos += positive_prob
            label += pred_label.tolist()
        
        
            test_patient = []
            for i in range(len(batch['ID'])):
                test_patient.append(batch['ID'][i][:12])
                
            test_patient_patch = []
            for i in range(len(batch['ID'])):
                test_patient_patch.append(batch['ID'][i])
                
            
                
                
            ID += test_patient
            patch_ID += test_patient_patch
            duration = batch['duration'].tolist()

            long_short += batch['long_short'].tolist()
        
            batch_acc = np.sum(np.array(test_label) ==  np.array(test_predicted))/ len(test_label)
            batch_mean_loss = sum(test_loss) / len(test_loss)
            fpr, tpr, thresholds = metrics.roc_curve(test_label, test_pos)
            auc = metrics.auc(fpr, tpr)
    
        total = {'ID' : ID, 'patch_ID' : patch_ID, 'long_short' : long_short, 'duration' : duration, 'prob' : test_pos, 'pred_label' : label}
        
        log.info('TEST Evaluation Result')
        log.info('Epoch : {}/{}, Batch : {}/{}, Loss : {:.3f}, ACC : {:.3f}, AUC : {:.3f}'.format(epoch+1, num_epochs, i+1, len(test_dataloader), batch_mean_loss, batch_acc, auc))    
        
        
        
            
     
        
        
        
        
        
        
        # 데이터 프레임 형성(환자별)
        ID_ = []
        patch_ = []
        prob_mean = []
        y_true = []
        y_pred = []
        #duration = []
        for i in range(len(np.unique(np.array(patch_ID)))):
            index = np.array(total['patch_ID']) == np.unique(np.array(patch_ID))[i]
            ID_ += np.array(total['ID'])[index].tolist()
            patch_ += np.array(total['patch_ID'])[index].tolist()
            prob_mean += np.array(total['prob'])[index].tolist()
            y_true += np.array(total['long_short'])[index].tolist()    
            y_pred += np.array(total['pred_label'])[index].tolist()
            #duration += np.array(total['duration'])[index].tolist()
    
    
    
        DataFrame_Patient_Level = pd.DataFrame({'ID' : np.array(ID_), 'patch_ID' : np.array(patch_), 'prob' : np.array(prob_mean), 
                                                'true_label' : np.array(y_true), 'pred_label' :np.array(y_pred)})
    
        DataFrame_Patient_Level_groupby = DataFrame_Patient_Level.groupby('ID').mean()
        DataFrame_Patient_Level_groupby_patch = DataFrame_Patient_Level.groupby('patch_ID').mean()
                                  
        
        
        
        
        
        
        ### 데이터 프레임 형성(patch별  ->  환자별)
        
        patchID = []
        for i in range(len(DataFrame_Patient_Level_groupby_patch.index.values)):
            patchID.append(DataFrame_Patient_Level_groupby_patch.index.values[i][:12])
        patchID = np.array(patchID)
    
        final_ID = []
        final_mean_prob = []
        final_max_prob = []
        final_min_prob = []
        final_true_label = []
        # final_pred_label = []
        for i in range(len(np.unique(np.array(ID)))):
    
            patient_index = (patchID == np.unique(np.array(ID))[i])  # patchID : patch ID ,  np.unique(ID) : patient ID
            mean_prob = np.mean(DataFrame_Patient_Level_groupby_patch[patient_index]['prob'].values)
            max_prob = np.max(DataFrame_Patient_Level_groupby_patch[patient_index]['prob'].values)
            min_prob = np.min(DataFrame_Patient_Level_groupby_patch[patient_index]['prob'].values)
            true_label = DataFrame_Patient_Level_groupby_patch[patient_index]['true_label'].values[0]
            # pred_label = DataFrame_Patient_Level_groupby_patch[patient_index]['pred_label'].values[0]
    
            final_ID += [np.unique(np.array(ID))[i]]
            final_mean_prob += [mean_prob]
            final_max_prob += [max_prob]
            final_min_prob += [min_prob]
            final_true_label += [true_label]
            
    
        final_df = {'ID' : final_ID, 'mean_prob' : final_mean_prob, 'max_prob' : final_max_prob, 
                    'min_prob' : final_min_prob, 'true_label' : final_true_label}
        
        
        fpr, tpr, thresholds = metrics.roc_curve(final_df['true_label'], final_df['mean_prob'])
        mean_auc = metrics.auc(fpr, tpr)  
        
        fpr, tpr, thresholds = metrics.roc_curve(final_df['true_label'], final_df['max_prob'])
        max_auc = metrics.auc(fpr, tpr)  
        
        fpr, tpr, thresholds = metrics.roc_curve(final_df['true_label'], final_df['min_prob'])
        min_auc = metrics.auc(fpr, tpr)  
        
        
        

        create_file(f'checkpoint/{info}/result.csv', f'epoch{epoch+1}')
    
        log.info('\n')
        # log.info('---------------------------------  Test_Patient_Level_AUC : {:.3f}'.format(auc))
        log.info('---------------------------------   Test_Patient_Level_Mean_AUC : {:.3f}'.format(mean_auc))
        log.info('---------------------------------  Test_Patient_Level_Max_AUC : {:.3f}'.format(max_auc))
        log.info('---------------------------------   Test_Patient_Level_Min_AUC : {:.3f}'.format(min_auc))
        log.info('\n')

        
        
        #duration_csv = []
        #for i in range(len(final_df['ID'])):
        #    duration_csv.append(DataFrame_Patient_Level[DataFrame_Patient_Level['ID'].values  == final_df['ID'][i]]['duration'].values[0])
        
        
        
        csv_file = np.hstack([np.array(final_df['ID']).reshape(-1, 1), 
                              np.array(final_df['mean_prob']).reshape(-1, 1), 
                              np.array(final_df['max_prob']).reshape(-1, 1), 
                              np.array(final_df['min_prob']).reshape(-1, 1),
                              np.array(final_df['true_label']).reshape(-1, 1),
                              #np.repeat(true_label, repeats=len(final_ID)).reshape(-1, 1),
                              np.repeat(mean_auc, repeats=len(final_ID)).reshape(-1, 1),
                              np.repeat(max_auc, repeats=len(final_ID)).reshape(-1, 1),
                              np.repeat(min_auc, repeats=len(final_ID)).reshape(-1, 1)])
        
        create_file(f'checkpoint/{info}/result.csv', 'ID, mean_prob, max_prob, min_prob, true_label, mean_auc, max_auc, min_auc')


        for i in range(len(csv_file)):
            create_file(f'checkpoint/{info}/result.csv', ','.join(csv_file[i]))
        
        
        torch.save(model.state_dict(), 'checkpoint/{}/fold {} epoch {}.pth'.format(info, fold+1, epoch+1))
    create_file(f'checkpoint/{info}/result.csv', '--------------------------------------------------')


# In[11]:


pd.DataFrame(csv_file)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




