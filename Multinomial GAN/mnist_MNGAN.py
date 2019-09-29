import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
from torchvision import transforms
from torchvision.utils import save_image
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os,shutil,sys
import torchvision.utils as vutils
from spectral_normalization import SpectralNorm
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root_dir',
                        type=str,
                        default='/home/Data/')
    parser.add_argument('--bit', type=int, default=1)
    parser.add_argument('--log_dir', type=str, default='/home/becky/data/mnist_2/')
    args = parser.parse_args()

    return args


def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)

def to_cuda(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return x

def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)

def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        if (m.bias is not None):
            m.bias.data.zero_()


def data_preapre(args):
    # Image processing
    transform = transforms.Compose([
        transforms.ToTensor()])

    mnist = datasets.MNIST(root=args.data_root_dir,
                           train=True,
                           transform=transform,
                           download=True)
    # Data loader
    data_loader = torch.utils.data.DataLoader(dataset=mnist,
                                              batch_size=100,
                                              shuffle=True)
    return data_loader

class DisOrFuncf(nn.Module):

    def __init__(self,bit):
        super(DisOrFuncf, self).__init__()
        self.fc1 = SpectralNorm(nn.Linear(784, 512))
        self.fc2 = SpectralNorm(nn.Linear(512, 256))
        self.fc3 = SpectralNorm(nn.Linear(256, 1))
        self.relu = nn.LeakyReLU(0.2)
        self.sigm = nn.Sigmoid()
        nc = 2**bit
        self.a = self._build_mask(nc)
        self.nc = nc


    def _build_mask(self,nc):
        """
        Need to be expanded when encountering the data
        :param nc: number of channels
        :return:
        """
        ma = to_cuda(torch.zeros([nc,1,784]))
        for i in range(nc):
            ma[i] = i*torch.ones([1,784])
        return to_cuda(ma)

    def forward(self, x, is_train_g):
        if is_train_g:
            sample_x = x[:, 0, :]
            mask = self.a.expand(-1, x.shape[0], -1)
            GOGradX = to_cuda(torch.zeros_like(x))
            fnew = to_cuda(torch.zeros(self.nc, x.shape[0], 784))
            dout = self.fc1(sample_x)
            for i in range(self.nc):
                a = to_var(to_cuda(mask[i]) - sample_x.data)
                fnew_tmp = self.relu(dout.unsqueeze(1) + a.unsqueeze(-1) * self.fc1.module.weight.t())
                fnew_tmp = self.relu(fnew_tmp.matmul(self.fc2.module.weight.t()) + self.fc2.module.bias)
                fnew_tmp = fnew_tmp.matmul(self.fc3.module.weight.t()) + self.fc3.module.bias
                fnew[i] = self.sigm(fnew_tmp.squeeze(-1)).data
        else:
            dout = self.fc1(x)

        dout = self.relu(dout)
        dout = self.relu(self.fc2(dout))
        dout = self.fc3(dout)
        if is_train_g:
            tmp = dout
            fout = torch.sigmoid(tmp)
            for i in range(self.nc):
                GOGradX[:,i,:] = - fnew[-1] + fnew[i]
            GOGradX = GOGradX.detach()
            out =  torch.sum(torch.sum(x * GOGradX, -1),-1, keepdim=True) + (fout - torch.sum(torch.sum(x * GOGradX, -1),-1,keepdim=True)).detach()
        else:
            dout = self.sigm(dout)
            out = dout
        return out



class Generator(nn.Module):
    # initializers
    def __init__(self, bit, d=64):
        super(Generator, self).__init__()
        nz = 100
        ngf = d
        nc = 2**bit 
        self.nc = nc
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 4, 4, 1, 0, bias=False),
            nn.ReLU(True),
            nn.BatchNorm2d(ngf * 4),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.ReLU(True),
            nn.BatchNorm2d(ngf * 2),

            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 2, bias=False),
            nn.ReLU(True),
            nn.BatchNorm2d(ngf),

            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Softmax(dim=1),
        )

    def weight_init(self, mean, std):
        for i in enumerate(self._modules):
            normal_init(self._modules[i], mean, std)

    # forward method
    def forward(self, input):
        Qp = self.main(input)
        gout,Qp = self.GOMultinomial(Qp)
        return gout, Qp

    def GOMultinomial(self, Qp):
        Qp_sample = Qp.transpose(1, 3)
        Qp_shape = Qp.shape

        Qp_sample = Qp_sample.contiguous().view(-1, self.nc)
        Qp_sample_shape = Qp_sample.shape

        if(torch.cuda.is_available()):
            zsamp = torch.multinomial(Qp_sample,1).type(torch.cuda.FloatTensor)
        else:
            zsamp = torch.multinomial(Qp_sample, 1).type(torch.FloatTensor)
        zsamp = zsamp.expand(-1, self.nc).contiguous().view(Qp_shape[0],Qp_shape[2],Qp_shape[3],self.nc).transpose(1,3)

        zout = Qp + (zsamp - Qp).detach()
        return zout, Qp




def image_filter(images,nc):
    return torch.round(images*(nc-1)).type(torch.FloatTensor)



def do_training(data_loader,args):
    D = DisOrFuncf(args.bit)
    G = Generator(args.bit)
    # G.weight_init(mean=0.0, std=0.01)
    # D.weight_init(mean=0.0, std=0.01)

    if torch.cuda.is_available():
        D.cuda()
        G.cuda()
    # Binary cross entropy loss and optimizer
    criterion = nn.BCELoss()
    d_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, D.parameters()), lr=3e-4)
    g_optimizer = torch.optim.Adam(G.parameters(), lr=3e-4)

    # Start training
    for epoch in range(200):
        for i, (images, _) in enumerate(data_loader):
            # Build mini-batch dataset
            batch_size = images.size(0)


            images = image_filter(images,2**args.bit)
            images = to_var(images.view(batch_size, -1))

            # Create the labels which are later used as input for the BCE loss
            real_labels = to_var(torch.ones(batch_size))
            fake_labels = to_var(torch.zeros(batch_size))

            # ============= Train the discriminator =============#
            # Compute BCE_Loss using real images where BCE_Loss(x, y): - y * log(D(x)) - (1-y) * log(1 - D(x))
            # Second term of the loss is always zero since real_labels == 1
            is_train_g = 0
            outputs = D(images, is_train_g=False)
            d_loss_real = criterion(outputs, real_labels)
            #assert torch.sum(outputs < 0)==0
            real_score = outputs

            # Compute BCELoss using fake images
            # First term of the loss is always zero since fake_labels == 0
            z_ = torch.randn((batch_size, 100)).view(-1, 100, 1, 1)
            z_ = to_var(z_)


            fake_images, _ = G(z_)
            fake_images = fake_images[:,0,:,:].contiguous().view(-1,784)
            outputs = D(fake_images, is_train_g=False)
            d_loss_fake = criterion(outputs, fake_labels)
            fake_score = outputs

            # Backprop + Optimize
            d_loss = d_loss_real + d_loss_fake
            D.zero_grad()
            d_loss.backward()
            d_optimizer.step()

            # =============== Train the generator ===============#
            # Compute loss with fake images
            is_train_g = 1
            z_ = torch.randn((batch_size, 100)).view(-1, 100, 1, 1)
            z_ = to_var(z_)

            fake_images, Qp = G(z_)
            fake_images = fake_images.view(-1,2**args.bit,784)
            Qp = Qp.view(-1,2**args.bit,784)
            outputs = D(fake_images, is_train_g=True)

            g_loss = criterion(outputs, real_labels)
            # Backprop + Optimize
            D.zero_grad()
            G.zero_grad()
            g_loss.backward()
            g_optimizer.step()

            if (i + 1) % 300 == 0:
                print('Epoch [%d/%d], Step[%d/%d], d_loss: %.4f, '
                      'g_loss: %.4f, D(x): %.2f, D(G(z)): %.2f'
                      % (epoch, 200, i + 1, 600, d_loss.data[0], g_loss.data[0],
                         real_score.data.mean(), fake_score.data.mean()))

        # Save real images
        fake_images = fake_images[:,0,:].contiguous().view(images.size(0), 1, 28, 28)
        vutils.save_image(1-fake_images.data,
                          args.log_dir+ 's'+str(
                epoch + 1)  + '.png',
                          normalize=True, nrow=10)

    # Qp1 = Qp[:,0,:].contiguous().view(images.size(0), 1, 28, 28)
    # vutils.save_image(Qp1.data[:100],
    #                   out_dir+'a'+ str(
    #         epoch + 1)  + '.png',
    #                   normalize=True, nrow=10)
    # values, indices = Qp.max(1)
    # Qp2 = indices.view(images.size(0), 1, 28, 28)
    # vutils.save_image(Qp2.data[:100],
    #                   out_dir+ 'b'+str(
    #         epoch + 1)  + '.png',
    #                   normalize=True, nrow=10)

    # Save sampled images
    # fake_images = fake_images.view(fake_images.size(0), 1, 28, 28)
    # save_image(denorm(fake_images.data), './data/fake_images-%d.png' % (epoch + 1))

    # Save the trained parameters
    if((epoch+1)%5==0):
        torch.save(G.state_dict(), args.log_dir+'generator_{}.pkl'.format(epoch))
        torch.save(D.state_dict(), args.log_dir+'discriminator.pkl'.format(epoch))

def main():
    args = parse_args()
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
        shutil.copyfile(sys.argv[0], args.log_dir+ '/mn{}.py'.format(str(args.bit)))

    data = data_preapre(args)
    do_training(data,args)



if __name__ == '__main__':
    main()



