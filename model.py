'''
Author: Tammie li
Description: Define model
FilePath: \DRL\model.py
'''


import torch.nn as nn
import torch
import torch.nn.functional as F


class MGIFNet(nn.Module):
    def __init__(self, n_class=2, channels=64, f1=8, d=1, f2=2, drop_out=0.8, kernel_length=3):
        super(MGIFNet, self).__init__()
        self.F1 = f1
        self.F2 = f2
        self.drop_out = drop_out
        self.kernel_length = kernel_length
        self.D = d
        self.channel = channels

        #  Temporal representation learning
        self.temp_block1 = nn.Sequential(
            nn.ZeroPad2d((self.kernel_length//2, self.kernel_length//2, 0, 0)),
            nn.Conv2d(1, self.F1, (1, self.kernel_length), bias=False),
            nn.BatchNorm2d(self.F1)
        )
        self.temp_block2 = nn.Sequential(
            nn.ZeroPad2d((self.kernel_length//2, self.kernel_length//2, 0, 0)),
            nn.Conv2d(1, self.F1, (1, self.kernel_length), bias=False),
            nn.BatchNorm2d(self.F1)
        )
        self.temp_block3 = nn.Sequential(
            nn.ZeroPad2d((self.kernel_length//2, self.kernel_length//2, 0, 0)),
            nn.Conv2d(1, self.F1, (1, self.kernel_length), bias=False),
            nn.BatchNorm2d(self.F1)
        )
        self.temp_block4 = nn.Sequential(
            nn.ZeroPad2d((self.kernel_length//2, self.kernel_length//2, 0, 0)),
            nn.Conv2d(1, self.F1, (1, self.kernel_length), bias=False),
            nn.BatchNorm2d(self.F1)
        )

        # Spatial representation learning
        self.spatial_block1 = nn.Sequential(
            nn.Conv2d(self.F1, self.F1*self.D, (self.channel, 1), groups=self.F1),
            nn.BatchNorm2d(self.F1*self.D),
            nn.ELU(), 
            nn.Dropout(self.drop_out)
        )

        self.spatial_block2 = nn.Sequential(
            nn.Conv2d(self.F1, self.F1*self.D, (self.channel, 1), groups=self.F1),
            nn.BatchNorm2d(self.F1*self.D),
            nn.ELU(), 
            nn.Dropout(self.drop_out)
        )
        self.spatial_block3 = nn.Sequential(
            nn.Conv2d(self.F1, self.F1*self.D, (self.channel, 1), groups=self.F1),
            nn.BatchNorm2d(self.F1*self.D),
            nn.ELU(), 
            nn.Dropout(self.drop_out)
        )
        self.spatial_block4 = nn.Sequential(
            nn.Conv2d(self.F1, self.F1*self.D, (self.channel, 1), groups=self.F1),
            nn.BatchNorm2d(self.F1*self.D),
            nn.ELU(), 
            nn.Dropout(self.drop_out)
        )

        self.ts_conv1 = nn.Conv2d(self.F1*self.D, self.F2, (1, 1))
        self.ts_conv2 = nn.Conv2d(self.F1*self.D, self.F2, (1, 1))
        self.ts_conv3 = nn.Conv2d(self.F1*self.D, self.F2, (1, 1))
        self.ts_conv4 = nn.Conv2d(self.F1*self.D, self.F2, (1, 1))

        self.classifier = nn.Linear(128*F2, n_class)

    def forward(self, x):
        # divide into four levels of downsampling
        x1 = x
        x2 = x[:, :, :, range(0, x.shape[-1], 2)]
        x3 = x[:, :, :, range(0, x.shape[-1], 4)]
        x4 = x[:, :, :, range(0, x.shape[-1], 8)]

        x1 = self.temp_block1(x1)
        x2 = self.temp_block2(x2)
        x3 = self.temp_block3(x3)
        x4 = self.temp_block4(x4)

        x1 = self.spatial_block1(x1)
        x2 = self.spatial_block2(x2)
        x3 = self.spatial_block3(x3)
        x4 = self.spatial_block4(x4)

        x1 = self.ts_conv1(x1)
        x2 = self.ts_conv2(x2)
        x3 = self.ts_conv3(x3)
        x4 = self.ts_conv4(x4)


        x1 = x1[:, :, :, range(0, x.shape[-1], 2)]
        x = x1 + x2
        x = x[:, :, :, range(0, x.shape[-1], 2)]
        x = x + x3
        x = x[:, :, :, range(0, x.shape[-1], 2)]
        x = x + x4

        x = x.view(x.size(0), -1)

        x = self.classifier(x)

        probas = F.softmax(x, dim=1)

        return probas


class DRL(nn.Module):
    def __init__(self, n_class=2, channels=64, f1=8, d=1, f2=2, drop_out=0.6, kernel_length=3):
        super(DRL, self).__init__()

        self.F1 = f1
        self.F2 = f2
        self.drop_out = drop_out
        self.kernel_length = kernel_length
        self.D = d
        self.channel = channels

        #  Temporal representation learning
        self.temp_block1 = nn.Sequential(
            nn.ZeroPad2d((self.kernel_length//2, self.kernel_length//2, 0, 0)),
            nn.Conv2d(1, self.F1, (1, self.kernel_length), bias=False),
            nn.BatchNorm2d(self.F1)
        )
        self.temp_block2 = nn.Sequential(
            nn.ZeroPad2d((self.kernel_length//2, self.kernel_length//2, 0, 0)),
            nn.Conv2d(1, self.F1, (1, self.kernel_length), bias=False),
            nn.BatchNorm2d(self.F1)
        )
        self.temp_block3 = nn.Sequential(
            nn.ZeroPad2d((self.kernel_length//2, self.kernel_length//2, 0, 0)),
            nn.Conv2d(1, self.F1, (1, self.kernel_length), bias=False),
            nn.BatchNorm2d(self.F1)
        )
        self.temp_block4 = nn.Sequential(
            nn.ZeroPad2d((self.kernel_length//2, self.kernel_length//2, 0, 0)),
            nn.Conv2d(1, self.F1, (1, self.kernel_length), bias=False),
            nn.BatchNorm2d(self.F1)
        )

        # Spatial representation learning
        self.spatial_block1 = nn.Sequential(
            nn.Conv2d(self.F1, self.F1*self.D, (self.channel, 1), groups=self.F1),
            nn.BatchNorm2d(self.F1*self.D),
            nn.ELU(), 
            nn.Dropout(self.drop_out)
        )

        self.spatial_block2 = nn.Sequential(
            nn.Conv2d(self.F1, self.F1*self.D, (self.channel, 1), groups=self.F1),
            nn.BatchNorm2d(self.F1*self.D),
            nn.ELU(), 
            nn.Dropout(self.drop_out)
        )
        self.spatial_block3 = nn.Sequential(
            nn.Conv2d(self.F1, self.F1*self.D, (self.channel, 1), groups=self.F1),
            nn.BatchNorm2d(self.F1*self.D),
            nn.ELU(), 
            nn.Dropout(self.drop_out)
        )
        self.spatial_block4 = nn.Sequential(
            nn.Conv2d(self.F1, self.F1*self.D, (self.channel, 1), groups=self.F1),
            nn.BatchNorm2d(self.F1*self.D),
            nn.ELU(), 
            nn.Dropout(self.drop_out)
        )

        # dimensionality reduction
        self.ts_conv1 = nn.Conv2d(self.F1*self.D, self.F2, (1, 1))
        self.ts_conv2 = nn.Conv2d(self.F1*self.D, self.F2, (1, 1))
        self.ts_conv3 = nn.Conv2d(self.F1*self.D, self.F2, (1, 1))
        self.ts_conv4 = nn.Conv2d(self.F1*self.D, self.F2, (1, 1))


        # projection head
        self.proj_head_block = nn.Sequential()
        self.proj_head_block.add_module('fully connect layer 1', nn.Linear(128*self.F2, 64))
        self.proj_head_block.add_module('activate function 1', nn.ELU())
        self.proj_head_block.add_module('fully connect layer 2', nn.Linear(64, 16))

        # classification
        self.classifier = nn.Linear(128*self.F2, n_class)

    def cal_loss(self, result, label):
        """
        Contrastive loss function 
        """
        sum_positive_pair = 0
        sum_negative_pair = 0
        for i in range(result.shape[0]):
            if label[i] == 0:
                sum_negative_pair += torch.exp(result[i])
            else:
                sum_positive_pair += torch.exp(result[i])
        sum_positive_pair += torch.exp(torch.tensor(-10))
        sum_negative_pair += torch.exp(torch.tensor(-10))
        loss = - torch.log(sum_negative_pair / sum_positive_pair)

        return loss
    
    def _forward_one_sample_feature(self, x):
        """
        Calculate the representation of a sample
        """
        x1 = x
        x2 = x[:, :, :, range(0, x.shape[-1], 2)]
        x3 = x[:, :, :, range(0, x.shape[-1], 4)]
        x4 = x[:, :, :, range(0, x.shape[-1], 8)]

        x1 = self.temp_block1(x1)
        x2 = self.temp_block2(x2)
        x3 = self.temp_block3(x3)
        x4 = self.temp_block4(x4)

        x1 = self.spatial_block1(x1)
        x2 = self.spatial_block2(x2)
        x3 = self.spatial_block3(x3)
        x4 = self.spatial_block4(x4)



        x1 = x1[:, :, :, range(0, x.shape[-1], 2)]
        x = x1 + x2
        x = x[:, :, :, range(0, x.shape[-1], 2)]
        x = x + x3
        x = x[:, :, :, range(0, x.shape[-1], 2)]
        x = x + x4

        x = x.view(x.size(0), -1)

        return x
    
    def class_mode(self, x):
        x1 = x
        x2 = x[:, :, :, range(0, x.shape[-1], 2)]
        x3 = x[:, :, :, range(0, x.shape[-1], 4)]
        x4 = x[:, :, :, range(0, x.shape[-1], 8)]

        x1 = self.block1(x1)
        x2 = self.block2(x2)
        x3 = self.block3(x3)
        x4 = self.block4(x4)

        x1 = self.block5(x1)
        x2 = self.block6(x2)
        x3 = self.block7(x3)
        x4 = self.block8(x4)

        return x1, x2, x3, x4


        
    def forward(self, x, mode=0):
        """
        There are three mode for forward
        if mode == 0: the stage of learning representation
        if mode == 1: the stage of learning classifier
        if mode == 2: the stage of test
        """

        x = torch.tensor(x).to(torch.float32)

        if mode == 0:
            result = torch.zeros((x.shape[0]))
            x1, x2 = x[:, 0, ...], x[:, 1, ...]

            x1 = x1.reshape(x1.shape[0], 1, x1.shape[1], x1.shape[2])       
            x2 = x2.reshape(x2.shape[0], 1, x2.shape[1], x2.shape[2])

            x1 = self._forward_one_sample_feature(x1)
            x2 = self._forward_one_sample_feature(x2)

            x1 = self.proj_head_block(x1)
            x2 = self.proj_head_block(x2)

            for i in range(result.shape[0]):
                similar = torch.cosine_similarity(x1[i, ...], x2[i, ...], dim=0)
                result[i] = similar
            return result
        
        elif mode == 1:
            x = x.reshape(x.shape[0], 1, x.shape[1], x.shape[2])

            # frozen MGI
            with torch.no_grad():
                x = self._forward_one_sample_feature(x)

            x = self.classifier(x)
            x = F.softmax(x, dim=1)

            return x
        
        elif mode == 2:
            x = x.reshape(x.shape[0], 1, x.shape[1], x.shape[2])

            with torch.no_grad():
                x = self._forward_one_sample_feature(x)
                x = self.classifier(x)
                x = F.softmax(x, dim=1)

            return x
        
        else:
            print("Please choose a true mode from the following value:   \n\
                    if mode == 0: the stage of learning representation;  \n\
                    if mode == 1: the stage of learning classifier;      \n\
                    if mode == 2: the stage of test."                      )


if __name__ == "__main__":
    test_data = torch.ones(4, 2, 256, 256)
    test_label = torch.tensor([0, 1, 0, 1])
    model = MGIFNet(channels=256)
    model = DRL(channels=256)

    y = model(test_data)

    loss = model.cal_loss(y, test_label)
    