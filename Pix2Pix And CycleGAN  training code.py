#!/usr/bin/env python
# coding: utf-8

# In[26]:


import argparse
import logging.handlers
import os
import sys
import datetime
import cv2
import matplotlib.pyplot as plt

import numpy as np
from scipy import linalg
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision

sys.path.append('./src')
from virtualstaining.model.networks import define_D, define_G, GANLoss, get_scheduler,     update_learning_rate
from virtualstaining.model.helper import StainingDataset, StainingDatasetAux


# In[2]:


def TorchRgb2hed(rgb, trans_mat):
    rgb = rgb.squeeze().permute(1, 2, 0)
    rgb = rgb + 2
    stain = -torch.log(rgb.view(-1, 3))
    stain = torch.matmul(stain, trans_mat, out=None)
    return stain.view(rgb.shape)


# In[2]:


run_info = 'model'

log = logging.getLogger('staining_log')
log.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
fileHandler = logging.FileHandler('./data/checkpoints/{}/log.txt'.format(run_info))
streamHandler = logging.StreamHandler()
fileHandler.setFormatter(formatter)
streamHandler.setFormatter(formatter)
#
log.addHandler(fileHandler)
log.addHandler(streamHandler)


# In[4]:


dataset_path = '../../../Projects/Pathology/Development/staining/data/patch_all_Train'


# In[5]:


train_dataset = StainingDatasetAux(dataset_dir=dataset_path, transform=True, crop=True)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4)


# In[ ]:





# In[15]:


model = 'unet'
input_nc = 3
output_nc = 3
device= torch.device("cpu")
ndf = 64


# In[16]:


net_g = define_G(model, input_nc, output_nc, gpu_id=device)

net_d = define_D(input_nc + output_nc, ndf, netD='basic', gpu_id=device)


# In[42]:


x = cv2.imread('엄가.jpg')
x = cv2.resize(x, (224, 224))
x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)/255
x = transforms.ToTensor()(x).type(torch.Tensor)


# In[43]:


y = cv2.imread('지옥.jpg')
y = cv2.resize(y, (224, 224))
y = cv2.cvtColor(y, cv2.COLOR_BGR2RGB)/255
y = transforms.ToTensor()(y).type(torch.Tensor)


# In[44]:


x = x.view(1, 3, 224, 224)
y = y.view(1, 3, 224, 224)


# In[45]:


transforms.ToPILImage()(x[0])


# In[46]:


transforms.ToPILImage()(y[0])


# In[35]:


# setup optimizer
lr = 0.0002
beta1 = 0.5
lr_policy='lambda'
epoch_count=1
niter=100
niter_decay=100
lr_decay_iters=50

optimizer_g = optim.Adam(net_g.parameters(),
                         lr=lr,
                         betas=(beta1, 0.999))

optimizer_d = optim.Adam(net_d.parameters(),
                             lr=lr,
                             betas=(beta1, 0.999))

criterionGAN = GANLoss().to(device)
criterionL1 = nn.L1Loss().to(device)
criterionMSE = nn.MSELoss().to(device)
net_g_scheduler = get_scheduler(optimizer_g, lr_policy, epoch_count,
                                    niter, niter_decay, lr_decay_iters)
net_d_scheduler = get_scheduler(optimizer_d, lr_policy, epoch_count,
                                    niter, niter_decay, lr_decay_iters)

rgb_from_hed = np.array([[0.65, 0.70, 0.29],
                             [0.07, 0.99, 0.11],
                             [0.27, 0.57, 0.78]])
hed_from_rgb = linalg.inv(rgb_from_hed)
hed_from_rgb = torch.Tensor(hed_from_rgb).cpu()


# # Pix2Pix  Training Code

# In[95]:


x2 = x


# In[ ]:


lamb_hed=0.9
hed_normalize=False
lamb=10
img_per_epoch =10000

d_loss = []
g_loss = []
g_loss_test = []

for epoch in range(1000):
    print('epoch {}'.format(epoch+1))

    net_d.train()
    net_g.train()
    
    # generate fake image
    G_x = net_g(x)
    optimizer_d.zero_grad()
        
    d_loss1 = -1*torch.log(nn.Sigmoid()(net_d(torch.cat((x, y), 1)))).mean()
    d_loss2 = -1*torch.log(1 - nn.Sigmoid()(net_d(torch.cat((x, G_x), 1)))).mean()
                              
    d_loss_total = (d_loss1 + d_loss2)
    d_loss_total.backward()
        
    d_loss += [d_loss_total.tolist()]
    d_loss_ = sum(d_loss) / len(d_loss)
    optimizer_d.step()
    
    
    
    G_x = net_g(x)
    
    optimizer_g.zero_grad()
    loss_g1 = -torch.log(nn.Sigmoid()(net_d(torch.cat((x, G_x), 1)))).mean()
    loss_g2 = ((y - G_x).abs()).mean()
    
    g_loss_total = (loss_g1 + loss_g2).mean()
    g_loss_total.backward()
    
    g_loss += [g_loss_total.tolist()]
    g_loss_ = sum(g_loss) / len(g_loss)
    
    optimizer_g.step()
    
    a = transforms.ToPILImage()(net_g(x).detach()[0])
    plt.imshow(a)
    plt.show()
    
    
    if not os.path.exists('result'):
        os.mkdir('result')
        
    plt.savefig('result/{} epoch.jpg'.format(epoch))    
    
    
    
    net_d.eval()
    net_g.eval()
    G_x = net_g(x2)
    
    loss_g1_test = -torch.log(nn.Sigmoid()(net_d(torch.cat((x2, G_x), 1)))).mean()
    loss_g2_test = ((y - G_x).abs()).mean()
    g_loss_total_test = (loss_g1_test + loss_g2_test).mean()
    g_loss_test += [g_loss_total_test.tolist()]
    g_loss_test_ = sum(g_loss_test) / len(g_loss_test)
    
    
    a = transforms.ToPILImage()(net_g(x).detach()[0])
    plt.imshow(a)
    plt.show()   
    
        
    if not os.path.exists('result_test'):
        os.mkdir('result_test')   
        
    plt.savefig('result_test/{} epoch.jpg'.format(epoch))           
        
        
        
    if not os.path.exists('checkpoint'):
        os.mkdir('checkpoint')

    torch.save(net_g.state_dict(), 'checkpoint/d_loss {} g_loss {} g_loss_test {} epoch {}'.format(d_loss_, g_loss_, 
                                                                                                   g_loss_test_, epoch+1))
    
    print("d_loss : {}, g_loss : {}, g_loss_test : {}".format(d_loss_, g_loss_, g_loss_test_))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[293]:


x2 = cv2.imread('x2.jpg')
x2 = cv2.resize(x2, (224, 224))
x2 = cv2.cvtColor(x2, cv2.COLOR_BGR2RGB)/255
x2 = transforms.ToTensor()(x2).type(torch.Tensor)


# In[ ]:





# In[ ]:





# In[ ]:





# # CycleGAN Training Code

# In[13]:


import functools

import torch.nn as nn


def conv_norm_act(in_dim, out_dim, kernel_size, stride, padding=0,
                  norm=nn.BatchNorm2d, relu=nn.ReLU):
    return nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size, stride, padding, bias=False),
        norm(out_dim),
        relu())


def dconv_norm_act(in_dim, out_dim, kernel_size, stride, padding=0,
                   output_padding=0, norm=nn.BatchNorm2d, relu=nn.ReLU):
    return nn.Sequential(
        nn.ConvTranspose2d(in_dim, out_dim, kernel_size, stride,
                           padding, output_padding, bias=False),
        norm(out_dim),
        relu())


class Discriminator(nn.Module):

    def __init__(self, dim=64):
        super(Discriminator, self).__init__()

        lrelu = functools.partial(nn.LeakyReLU, negative_slope=0.2)
        conv_bn_lrelu = functools.partial(conv_norm_act, relu=lrelu)

        self.ls = nn.Sequential(nn.Conv2d(3, dim, 4, 2, 1), nn.LeakyReLU(0.2),
                                conv_bn_lrelu(dim * 1, dim * 2, 4, 2, 1),
                                conv_bn_lrelu(dim * 2, dim * 4, 4, 2, 1),
                                conv_bn_lrelu(dim * 4, dim * 8, 4, 1, (1, 2)),
                                nn.Conv2d(dim * 8, 1, 4, 1, (2, 1)))

    def forward(self, x):
        return nn.Sigmoid()(self.ls(x))


class ResiduleBlock(nn.Module):

    def __init__(self, in_dim, out_dim):
        super(ResiduleBlock, self).__init__()

        conv_bn_relu = conv_norm_act

        self.ls = nn.Sequential(nn.ReflectionPad2d(1),
                                conv_bn_relu(in_dim, out_dim, 3, 1),
                                nn.ReflectionPad2d(1),
                                nn.Conv2d(out_dim, out_dim, 3, 1),
                                nn.BatchNorm2d(out_dim))

    def forward(self, x):
        return x + self.ls(x)


class Generator(nn.Module):

    def __init__(self, dim=64):
        super(Generator, self).__init__()

        conv_bn_relu = conv_norm_act
        dconv_bn_relu = dconv_norm_act

        self.ls = nn.Sequential(nn.ReflectionPad2d(3),
                                conv_bn_relu(3, dim * 1, 7, 1),
                                conv_bn_relu(dim * 1, dim * 2, 3, 2, 1),
                                conv_bn_relu(dim * 2, dim * 4, 3, 2, 1),
                                ResiduleBlock(dim * 4, dim * 4),
                                ResiduleBlock(dim * 4, dim * 4),
                                ResiduleBlock(dim * 4, dim * 4),
                                ResiduleBlock(dim * 4, dim * 4),
                                ResiduleBlock(dim * 4, dim * 4),
                                ResiduleBlock(dim * 4, dim * 4),
                                ResiduleBlock(dim * 4, dim * 4),
                                ResiduleBlock(dim * 4, dim * 4),
                                ResiduleBlock(dim * 4, dim * 4),
                                dconv_bn_relu(dim * 4, dim * 2, 3, 2, 1, 1),
                                dconv_bn_relu(dim * 2, dim * 1, 3, 2, 1, 1),
                                nn.ReflectionPad2d(3),
                                nn.Conv2d(dim, 3, 7, 1),
                                nn.Tanh())

    def forward(self, x):
        return self.ls(x)


# # CycleGAN (MSE loss)

# In[36]:


device='cpu'


# In[47]:


#G = Generator()
#F = Generator()
G = define_G(model, input_nc, output_nc, gpu_id=device)
F = define_G(model, input_nc, output_nc, gpu_id=device)


# In[48]:


D1 = Discriminator()
D2 = Discriminator()
#D1 = define_D(input_nc + output_nc, ndf, netD='basic', gpu_id=device)
#D2 = define_D(input_nc + output_nc, ndf, netD='basic', gpu_id=device)


# In[49]:


optim_G = torch.optim.Adam(G.parameters(), lr=0.0002, betas=(0.9, 0.999))
optim_F = torch.optim.Adam(F.parameters(), lr=0.0002, betas=(0.9, 0.999))

optim_D1 = torch.optim.Adam(D1.parameters(), lr=0.0002, betas=(0.9, 0.999))
optim_D2 = torch.optim.Adam(D2.parameters(), lr=0.0002, betas=(0.9, 0.999))


# In[50]:


MSE = nn.MSELoss()


# In[ ]:





# In[ ]:


num_epochs=1000

G_loss = []
D_loss = []

for epoch in range(num_epochs):
    print('epoch {}'.format(epoch+1))
    
    G.train()
    F.train()
    
    x = x
    y = y
    
    fake_x = G(x)
    fake_y = F(y)
    
    G_loss_GAN = MSE(D1(G(x)), torch.tensor(0.0).expand_as(D1(G(x)))) + MSE(D2(F(y)), torch.tensor(0.0).expand_as(D2(F(y))))
    G_loss_L1 = (F(G(x)) - x).abs().mean() + (G(F(y)) - y).abs().mean()
    
    G_total_loss = G_loss_GAN + G_loss_L1
    
    optim_G.zero_grad()
    optim_F.zero_grad()
    G_total_loss.backward()
    optim_G.step()
    optim_F.step()
    
    
    
    D1.train()
    D2.train()
    
    D_loss_GAN1 = MSE(D1(y), torch.tensor(1.0).expand_as(D1(y))) + MSE(D1(G(x)), torch.tensor(0.0).expand_as(D1(G(x))))
    D_loss_GAN2 = MSE(D2(x), torch.tensor(0.0).expand_as(D2(x))) + MSE(D2(F(y)), torch.tensor(1.0).expand_as(D2(F(y))))
    
    D_total_loss = D_loss_GAN1 + D_loss_GAN2
    
    optim_D1.zero_grad()
    optim_D2.zero_grad()
    D_total_loss.backward()
    optim_D1.step()
    optim_D2.step()
    
    print("G_loss : {}, D_loss : {}".format(G_total_loss, D_total_loss))
    
    a = transforms.ToPILImage()(G(x).detach()[0])
    plt.imshow(a)
    plt.show()
    
    b = transforms.ToPILImage()(F(y).detach()[0])
    plt.imshow(b)
    plt.show()
    
    if epoch % 30 == 0:
        torchvision.utils.save_image(G(x)[0], 'CycleGAN_result/epoch {}.jpg'.format(epoch+1))
    
        torch.save(G.state_dict(), 'checkpoint/{}.pth'.format(epoch+1))


# In[54]:


torchvision.utils.save_image(F(y)[0], 'CycleGAN_result/epoch {}.jpg'.format(epoch+10000))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# # CycleGAN( original GAN loss )

# In[71]:


num_epochs=1000

G_loss = []
D_loss = []

for epoch in range(num_epochs):
    print('epoch {}'.format(epoch+1))
    
    G.train()
    F.train()
    
    x = x
    y = y
    
    fake_x = G(x)
    fake_y = F(y)
    
    #G_loss_GAN = MSE(D1(G(x)), torch.tensor(0.0).expand_as(D1(G(x)))) + MSE(D2(F(y)), torch.tensor(0.0).expand_as(D2(F(y))))
    #G_loss_L1 = (F(G(x)) - x).abs().mean() + (G(F(y)) - y).abs().mean()
   
    G_loss_GAN = -1*torch.log(D1(G(x))).mean() -1*torch.log(D2(F(y))).mean()
    G_loss_L1 = (F(G(x)) - x).abs().mean() + (G(F(y)) - y).abs().mean()

    G_total_loss = G_loss_GAN + G_loss_L1
    
    optim_G.zero_grad()
    optim_F.zero_grad()
    G_total_loss.backward()
    optim_G.step()
    optim_F.step()
    
    
    
    D1.train()
    D2.train()
    
    #D_loss_GAN1 = MSE(D1(y), torch.tensor(1.0).expand_as(D1(y))) + MSE(D1(G(x)), torch.tensor(0.0).expand_as(D1(G(x))))
    #D_loss_GAN2 = MSE(D2(x), torch.tensor(0.0).expand_as(D2(x))) + MSE(D2(F(y)), torch.tensor(1.0).expand_as(D2(F(y))))
    
    D_loss_GAN1 = -1*torch.log(D1(y)).mean() -1*torch.log(1 - D1(G(x))).mean()
    D_loss_GAN2 = -1*torch.log(D2(x)).mean() -1*torch.log(1 - D2(F(y))).mean()
    
    D_total_loss = D_loss_GAN1 + D_loss_GAN2
    
    optim_D1.zero_grad()
    optim_D2.zero_grad()
    D_total_loss.backward()
    optim_D1.step()
    optim_D2.step()
    
    print("G_loss : {}, D_loss : {}".format(G_total_loss, D_total_loss))
    
    a = transforms.ToPILImage()(G(x).detach()[0])
    plt.imshow(a)
    plt.show()
    
    plt.savefig("CycleGAN_result/ {}.jpg".format(epoch+1))
    
    torch.save(G.state_dict(), 'checkpoint/{}.pth'.format(epoch+1))


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


a_real_test = Variable(iter(a_test_loader).next()[0], volatile=True)
b_real_test = Variable(iter(b_test_loader).next()[0], volatile=True)
a_real_test, b_real_test = utils.cuda([a_real_test, b_real_test])
for epoch in range(start_epoch, epochs):
    for i, (a_real, b_real) in enumerate(itertools.izip(a_loader, b_loader)):
        # step
        step = epoch * min(len(a_loader), len(b_loader)) + i + 1

        # set train
        Ga.train()
        Gb.train()

        # leaves
        a_real = Variable(a_real[0])
        b_real = Variable(b_real[0])
        a_real, b_real = utils.cuda([a_real, b_real])

        # train G
        a_fake = Ga(b_real)
        b_fake = Gb(a_real)

        a_rec = Ga(b_fake)
        b_rec = Gb(a_fake)

        # gen losses
        a_f_dis = Da(a_fake)
        b_f_dis = Db(b_fake)
        r_label = utils.cuda(Variable(torch.ones(a_f_dis.size())))
        a_gen_loss = MSE(a_f_dis, r_label)
        b_gen_loss = MSE(b_f_dis, r_label)

        # rec losses
        a_rec_loss = L1(a_rec, a_real)
        b_rec_loss = L1(b_rec, b_real)

        # g loss
        g_loss = a_gen_loss + b_gen_loss + a_rec_loss * 10.0 + b_rec_loss * 10.0

        # backward
        Ga.zero_grad()
        Gb.zero_grad()
        g_loss.backward()
        ga_optimizer.step()
        gb_optimizer.step()

        # leaves
        a_fake = Variable(torch.Tensor(a_fake_pool([a_fake.cpu().data.numpy()])[0]))
        b_fake = Variable(torch.Tensor(b_fake_pool([b_fake.cpu().data.numpy()])[0]))
        a_fake, b_fake = utils.cuda([a_fake, b_fake])

        # train D
        a_r_dis = Da(a_real)
        a_f_dis = Da(a_fake)
        b_r_dis = Db(b_real)
        b_f_dis = Db(b_fake)
        r_label = utils.cuda(Variable(torch.ones(a_f_dis.size())))
        f_label = utils.cuda(Variable(torch.zeros(a_f_dis.size())))

        # d loss
        a_d_r_loss = MSE(a_r_dis, r_label)
        a_d_f_loss = MSE(a_f_dis, f_label)
        b_d_r_loss = MSE(b_r_dis, r_label)
        b_d_f_loss = MSE(b_f_dis, f_label)

        a_d_loss = a_d_r_loss + a_d_f_loss
        b_d_loss = b_d_r_loss + b_d_f_loss

        # backward
        Da.zero_grad()
        Db.zero_grad()
        a_d_loss.backward()
        b_d_loss.backward()
        da_optimizer.step()
        db_optimizer.step()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[70]:


lamb_hed=0.9
hed_normalize=False
lamb=10
img_per_epoch =10000

for epoch in range(1, 201):

    loss_d_list = []
    loss_g_list = []
    loss_g_gan_list = []
    loss_g_l1_list = []
    loss_g_hed_l1_list = []

    for iteration, batch in enumerate(train_loader):
        real_a = batch['HE_image'].to(device).type(torch.float32)
        real_b = batch['CK_image'].to(device).type(torch.float32)
        real_b_hed = batch['CK_bin_image'].to(device).type(torch.float32) / 255.

        # generate fake image
        fake_b = net_g(real_a)
        #        real_a.shape
        ######################
        # (1) Update D network
        ######################
        optimizer_d.zero_grad()

        # predict with fake on Discriminator and calculate loss
        fake_ab = torch.cat((real_a, fake_b), 1)
        pred_fake = net_d.forward(fake_ab.detach())
        loss_d_fake = criterionGAN(pred_fake, False)  # if true = False

        # predict with real on Discriminator and calculate loss
        real_ab = torch.cat((real_a, real_b), 1)
        pred_real = net_d.forward(real_ab)
        loss_d_real = criterionGAN(pred_real, True)  # if true = True

        # Combined D loss
        loss_d = (loss_d_fake + loss_d_real) * 0.5  # average Discriminator losse

        loss_d.backward()
        optimizer_d.step()

        ######################
        # (2) Update G network
        ######################

        optimizer_g.zero_grad()

        # First, G(A) should fake the discriminator
        fake_ab = torch.cat((real_a, fake_b), 1)
        pred_fake = net_d.forward(fake_ab)
        loss_g_gan = criterionGAN(pred_fake, True)

        # Second, G(A) = B
        loss_g_l1 = criterionL1(fake_b, real_b)
        real_b_hed = real_b_hed.squeeze()
        fake_hed = TorchRgb2hed(fake_b, hed_from_rgb)

        if hed_normalize:
            fake_hed -= fake_hed.min(1, keepdim=True)[0]
            fake_hed /= fake_hed.max(1, keepdim=True)[0]
            real_b_hed -= real_b_hed.min(1, keepdim=True)[0]
            real_b_hed /= real_b_hed.max(1, keepdim=True)[0]

        loss_hed_l1 = criterionL1(fake_hed[:, :, :], real_b_hed[:, :, :])
        loss_g = loss_g_gan + loss_g_l1 * lamb + loss_hed_l1 * lamb_hed
        loss_g.backward()
        optimizer_g.step()

        loss_d_list.append(loss_d.item())
        loss_g_list.append(loss_g.item())
        loss_g_gan_list.append(loss_g_gan.item())
        loss_g_l1_list.append(loss_g_l1.item())
        loss_g_hed_l1_list.append(loss_hed_l1)

        if iteration % 1000 == 0:
            log.info(
                    'Epoch[{}]({}/{}): Loss_D: {:.4f} Loss_G: {:.4f}- GAN: {:.4f}, L1Loss: {:.4f}, hed_Loss: {:.4f} '.format(
                        epoch,
                        iteration,
                        len(train_loader),
                        sum(loss_d_list) / len(loss_d_list),
                        sum(loss_g_list) / len(loss_g_list),
                        sum(loss_g_gan_list) / len(loss_g_gan_list),
                        sum(loss_g_l1_list) / len(loss_g_l1_list),
                        sum(loss_g_hed_l1_list) / len(loss_g_hed_l1_list)))

        if iteration == img_per_epoch:
            break

    update_learning_rate(net_g_scheduler, optimizer_g)
    update_learning_rate(net_d_scheduler, optimizer_d)

    # checkpoint
    if epoch % 1 == 0:
        net_g_model_out_path = "./data/checkpoints/{}/netG_{}_epoch_{}.pth".format(
                run_info, run_info, epoch)
        net_d_model_out_path = "./data/checkpoints/{}/netD_{}_epoch_{}.pth".format(
                run_info, run_info, epoch)
        torch.save(net_g.state_dict(), net_g_model_out_path)
        torch.save(net_d.state_dict(), net_d_model_out_path)

        model_out_path = "./data/checkpoints/{}/model_epoch_{}.pth".format(run_info,
                                                                        epoch)
        torch.save({'epoch': epoch,
                        'Generator': net_g.state_dict(),
                        'Discriminator': net_d.state_dict(),
                        'optimizer_g': optimizer_g.state_dict(),
                        'optimizer_d': optimizer_d.state_dict(),
                        'scheduler_g': net_g_scheduler.state_dict(),
                        'scheduler_d': net_d_scheduler.state_dict()
                        }, model_out_path)

        log.info('Checkpoint saved to {}'.format(run_info))
    if epoch % 200 == 0:
        today = datetime.date.today()
        net_g_model_out_path = "./data/checkpoints/{}/{}_{}.pth".format(run_info, run_info, today.strftime("%Y%m%d"))
        torch.save(net_g.state_dict(), net_g_model_out_path)


# In[ ]:





# In[ ]:




