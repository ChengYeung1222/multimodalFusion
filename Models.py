import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch
import logging
from torch.autograd import Variable
from objectives import cca_loss

# from torchvision import models

__all__ = ['AlexNet', 'alexnet']

model_urls = {
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}


class new_Net(nn.Module):
    def __init__(self, num_classes=2):
        super(new_Net, self).__init__()
        self.features = nn.Sequential(
            nn.Linear(6, 256),  # todo
            nn.ReLU(inplace=True),
            # nn.Dropout(0.5),#todo

            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            # nn.Linear(8, num_classes),
            # nn.Dropout(0.5),#todo
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Sequential(torch.nn.Linear(256, 2))

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class AlexNet(nn.Module):

    def __init__(self, num_classes=2, blending=True):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(12, 64, kernel_size=11, stride=4, padding=2),  # todo:20
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # CrossMapLRN(5, 0.0001, 0.75),#todo
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # CrossMapLRN(5, 0.0001, 0.75),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        logging.debug('self.training=%d' % self.training)
        if self.training or blending:
            self.classifier = nn.Sequential(

                nn.Dropout(0.5),  # todo
                nn.Linear(256 * 6 * 6, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),  # todo:0.5,0.7
                nn.Linear(4096, 4096),  # todo:4096
                nn.ReLU(inplace=True),
                nn.Linear(4096, num_classes),  # todo:2048

                nn.Linear(4096, 256),  # todo:parallel [7]
                nn.ReLU(inplace=True),  # todo:parallel [8]
                nn.Linear(256, num_classes),
                nn.Linear(512, 512),
                nn.ReLU(inplace=True),
                nn.Linear(512, num_classes)

            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(256 * 6 * 6, 4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, 256),
                nn.ReLU(inplace=True),
                nn.Linear(256, num_classes),
            )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x


class AlexNetFc(nn.Module):
    def __init__(self):
        super(AlexNetFc, self).__init__()
        model_alexnet = AlexNet()
        self.features = model_alexnet.features
        # self.x_view = 0  # todo:
        self.classifier = nn.Sequential()
        for i in range(6):
            self.classifier.add_module("classifier" + str(i), model_alexnet.classifier[i])
        self.__in_features = model_alexnet.classifier[6].in_features  # nn.Linear(in_features=,)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        # self.x_view = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x

    def output_num(self):
        return self.__in_features


class DAN_with_Alex(nn.Module):

    def __init__(self, num_classes=2, branch_fixed=False):
        super(DAN_with_Alex, self).__init__()
        # self.conv1=alexnet().features[0]
        self.features = alexnet(branch_fixed=branch_fixed).features
        # for i in range(1,13):
        #     exec('self.features{} = alexnet().features[{}]'.format(i, i))
        # self.classifier=alexnet().classifier
        self.l6 = alexnet(branch_fixed=branch_fixed).classifier[0]
        self.cls1 = alexnet(branch_fixed=branch_fixed).classifier[1]
        self.cls2 = alexnet(branch_fixed=branch_fixed).classifier[2]
        self.l7 = alexnet(branch_fixed=branch_fixed).classifier[3]
        self.cls4 = alexnet(branch_fixed=branch_fixed).classifier[4]
        self.l8 = alexnet(branch_fixed=branch_fixed).classifier[5]

        self.L7 = alexnet(branch_fixed=branch_fixed).classifier[7]
        self.L9 = alexnet(branch_fixed=branch_fixed).classifier[9]
        self.L10 = alexnet(branch_fixed=branch_fixed).classifier[10]
        self.L12 = alexnet(branch_fixed=branch_fixed).classifier[12]
        # ++++++++++

        self.cls_fc = nn.Linear(4096, num_classes)  # todo:test

    def forward(self, source, fluid_source,
                blending=False, parallel=False, fluid_feature=0.,
                correlation=False, outdim_size=20, device=torch.device('cpu')):
        loss = .0
        source = self.features(source)  # todo:nmdwsm
        source = source.view(source.size(0), 256 * 6 * 6)

        if blending:
            # new_features = Variable(torch.rand(256, 8), requires_grad=True)  # todo: parallel
            new_features = fluid_source
            new_features = new_net().cuda().features[0](new_features)  # todo:parallel
            new_features = new_net().cuda().features[1](new_features)
        else:
            new_features = 0.

        source = self.l6(source)
        self.cls1.cuda()
        self.cls2.cuda()
        source = self.cls1(source)
        source = self.cls2(source)
        source = self.l7(source)
        if blending:
            new_features = new_net().cuda().features[2](new_features)  # todo:parallel
            new_features = new_net().cuda().features[3](new_features)
        self.cls4.cuda()
        source = self.cls4(source)
        source = self.l8(source)
        if blending:
            source = alexnet().cuda().classifier[7](source)
            source = alexnet().cuda().classifier[8](source)

        if blending:
            new_features = new_net().cuda().features[4](new_features)  # todo:parallel
            new_features = new_net().cuda().features[5](new_features)

        if parallel:
            # source = self.L7(source) todo
            # source = alexnet().cuda().classifier[8](source)
            if correlation:
                negative_corr = cca_loss(outdim_size=outdim_size, device=device).loss(H1=source, H2=new_features)
            new_cat = torch.cat((source, new_features), dim=1)
            new_cat = self.L10(new_cat)
            new_cat = alexnet().cuda().classifier[11](new_cat)
            new_cat = self.L12(new_cat)
        else:
            source = self.cls_fc(source)
            new_cat = 0.

        if correlation:
            return source, loss, new_cat, negative_corr
        else:
            return source, loss, new_cat


def new_net():
    model = new_Net()
    for name, params in model.named_parameters():
        if name.find('weight') != -1:
            torch.nn.init.kaiming_normal(params)
        if name.find('bias') != -1:
            torch.nn.init.constant(params, val=0.)
    return model


def alexnet(pretrained=False, frozen=True, branch_fixed=True, **kwargs):  # todo
    model = AlexNet()
    for name, params in model.named_parameters():
        if branch_fixed:
            pass
            # if name.find('features') != -1:
            #     params.requires_grad = False
            # else:
            #     params.requires_grad = False
            #     if name.find('7') != -1:
            #         break
            # count = 0
            # while (count < 7):
            #     params.requires_grad = False
            #     count += 1
        if frozen:  # todo:
            if name.find('3') != -1:
                params.requires_grad = False
            if name.find('6') != -1:
                params.requires_grad = False
        if name.find('bias') == -1:
            if name.find('features') != -1:
                torch.nn.init.kaiming_normal(params)
            # if name.find('conv2')!=-1:

            elif name.find('classifier') != -1:
                torch.nn.init.kaiming_normal(params)
        else:
            torch.nn.init.constant(params, val=0.01)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['alexnet']), strict=False)
    return model
