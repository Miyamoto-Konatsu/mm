import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import init
from torchvision import models
import torch.nn.functional as F
import torchvision.transforms as transforms
import pretrainedmodels

# Defines the new fc layer and classification layer
# |--Linear--|--bn--|--relu--|--Linear--|
class Ours(nn.Module):
    def __init__(self, input_dim, class_num, droprate, linear=512, return_f=False):
        super(Ours, self).__init__()
        self.return_f = return_f

        fc = [nn.Linear(input_dim + 100, linear)]
        fc +=  [nn.BatchNorm1d(linear)]
        fc += [nn.Dropout(p=droprate)]
        fc = nn.Sequential(*fc)
        fc.apply(self.init_fc)

        classifier = []
        classifier += [nn.Linear(linear, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(self.init_classifier)

        self.fc = fc
        self.classifier = classifier

    def forward(self, x, y):
        x = torch.cat([x, y], 1)
        x = self.fc(x)
        if self.return_f:
            f = x
            x = self.classifier(x)
            return [x, f]
        else:
            x = self.classifier(x)
            return x

    @staticmethod
    def init_fc(m):
        classname = m.__class__.__name__
        # print(classname)
        if classname.find('Conv') != -1:
            init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')  # For old pytorch, you may use kaiming_normal.
        elif classname.find('Linear') != -1:
            init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        elif classname.find('BatchNorm1d') != -1:
            init.normal_(m.weight.data, 1.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            init.constant_(m.bias.data, 0.0)

    @staticmethod
    def init_classifier(m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            init.normal_(m.weight.data, std=0.001)
            init.constant_(m.bias.data, 0.0)

class Model(nn.Module):

    def __init__(self, class_num ,droprate=0.5, stride=2,circle=False,ibn=False, linear_num=512,text_dim = 100):
        super(Model, self).__init__()
        self.model = models.resnet50(pretrained=True)
        self.model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.circle = circle
        self.classifier = Ours(2048, class_num, droprate, linear=linear_num, return_f=circle)
        self.bn = nn.BatchNorm1d(text_dim, affine= False)
    def forward(self, x, y):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        x = x.view(x.size(0), x.size(1))
        y = y.view(-1, 100)
        y /= 20
        # y = self.bn(y)
        x = self.classifier(x, y)
        return x


'''
# debug model structure
# Run this code with:
python model.py
'''
if __name__ == '__main__':
    # Here I left a simple forward function.
    # Test the model, before you train it.

    net = Model(5000)
    # print(net)
    # net = model_swin(751, stride=1)
    # net.classifier = nn.Sequential()
    # print(net)
    input1 = Variable(torch.FloatTensor(8, 3, 224, 224))
    input2 = Variable(torch.FloatTensor(8, 100))
    print(input2)
    input2 = F.normalize(input2,p=2, dim=1)
    print(input2)

    output = net(input1, input2)
    # print('net output size:')
    # print(output.shape)
