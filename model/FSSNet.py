import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import model.resnet as models
import model.vgg as vgg_models
from model.widgets.ASPP import ASPP
from model.g_GCN import g_GCN
from model.similarity import similarity
from model.Tripletencoder import Tripletencoder101
import torch.nn.init as initer

def embedding_distance(feature_1,feature_2):
    feature_1=  feature_1.cpu().detach().numpy()
    feature_2 = feature_2.cpu().detach().numpy()
    dist =np.linalg.norm(feature_1-feature_2)
    return dist

def get_vgg16_layer(model):
    layer0_idx = range(0,7)
    layer1_idx = range(7,14)
    layer2_idx = range(14,24)
    layer3_idx = range(24,34)
    layer4_idx = range(34,43)
    layers_0 = []
    layers_1 = []
    layers_2 = []
    layers_3 = []
    layers_4 = []
    for idx in layer0_idx:
        layers_0 += [model.features[idx]]
    for idx in layer1_idx:
        layers_1 += [model.features[idx]]
    for idx in layer2_idx:
        layers_2 += [model.features[idx]]
    for idx in layer3_idx:
        layers_3 += [model.features[idx]]
    for idx in layer4_idx:
        layers_4 += [model.features[idx]]
    layer0 = nn.Sequential(*layers_0)
    layer1 = nn.Sequential(*layers_1)
    layer2 = nn.Sequential(*layers_2)
    layer3 = nn.Sequential(*layers_3)
    layer4 = nn.Sequential(*layers_4)
    return layer0,layer1,layer2,layer3,layer4

def init_weights(model, conv='kaiming', batchnorm='normal', linear='kaiming', lstm='kaiming'):

    for m in model.modules():
        if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            if conv == 'kaiming':
                initer.kaiming_normal_(m.weight)
            elif conv == 'xavier':
                initer.xavier_normal_(m.weight)
            else:
                raise ValueError("init type of conv error.\n")
            if m.bias is not None:
                initer.constant_(m.bias, 0)

        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):#, BatchNorm1d, BatchNorm2d, BatchNorm3d)):
            if batchnorm == 'normal':
                initer.normal_(m.weight, 1.0, 0.02)
            elif batchnorm == 'constant':
                initer.constant_(m.weight, 1.0)
            else:
                raise ValueError("init type of batchnorm error.\n")
            initer.constant_(m.bias, 0.0)

        elif isinstance(m, nn.Linear):
            if linear == 'kaiming':
                initer.kaiming_normal_(m.weight)
            elif linear == 'xavier':
                initer.xavier_normal_(m.weight)
            else:
                raise ValueError("init type of linear error.\n")
            if m.bias is not None:
                initer.constant_(m.bias, 0)

        elif isinstance(m, nn.LSTM):
            for name, param in m.named_parameters():
                if 'weight' in name:
                    if lstm == 'kaiming':
                        initer.kaiming_normal_(param)
                    elif lstm == 'xavier':
                        initer.xavier_normal_(param)
                    else:
                        raise ValueError("init type of lstm error.\n")
                elif 'bias' in name:
                    initer.constant_(param, 0)

class FSSNet(nn.Module):
    def __init__(self, layers=50, classes=2, criterion=nn.CrossEntropyLoss(ignore_index=255),
                 pretrained=True, shot=1, ppm_scales=[60, 30, 15, 8], vgg=False, FPN=True):
        super(FSSNet, self).__init__()
        assert layers in [50, 101, 152]
        print(ppm_scales)
        assert classes > 1
        from torch.nn import BatchNorm2d as BatchNorm
        # 参数
        self.criterion = criterion
        self.shot = shot
        self.vgg = vgg
        self.trip = nn.Parameter(torch.zeros(1))

        models.BatchNorm = BatchNorm

        # Backbone Related
        if self.vgg:
            print('INFO: Using VGG_16 bn')
            vgg_models.BatchNorm = BatchNorm
            vgg16 = vgg_models.vgg16_bn(pretrained=pretrained)
            print(vgg16)
            self.layer0, self.layer1, self.layer2, \
                self.layer3, self.layer4 = get_vgg16_layer(vgg16)

        else:
            print('INFO: Using ResNet {}'.format(layers))
            if layers == 50:
                resnet = models.resnet50(pretrained=pretrained)
            elif layers == 101:
                resnet = models.resnet101(pretrained=pretrained)
            else:
                resnet = models.resnet152(pretrained=pretrained)
            self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu1, resnet.conv2, resnet.bn2, resnet.relu2,
                                        resnet.conv3, resnet.bn3, resnet.relu3, resnet.maxpool)
            self.layer1, self.layer2, self.layer3, self.layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4


        # feature dimension
        reduce_dim = 256
        class_num =2
        # if self.vgg:
        #     fea_dim = 512 + 256
        # else:
        #     fea_dim = 1024 + 512

        # query  下采样
        self.down = nn.Sequential(
            nn.Conv2d(512, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5)
        )
        self.down_s = nn.Sequential(
            nn.Conv2d(1024, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5)
        )
        #
        # # support 下采样
        self.Tripletencoder = Tripletencoder101(FPN).cuda()
        # 256 -> 256
        self.res1 = nn.Sequential(
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
        )
        # 256 -> 256
        self.res2 = nn.Sequential(
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
        )
        # 256 -> 256
        self.res3 = nn.Sequential(
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
        )
        # ASPP
        self.ASPP = ASPP().cuda()
        self.GCN = g_GCN(reduce_dim, int(reduce_dim / 2)).cuda()


        # 512 -> 256


        self.cls = nn.Sequential(
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(reduce_dim, classes, kernel_size=1)
        )

        self.int1 = nn.Sequential(
            nn.Conv2d(reduce_dim+1, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True))

        self.int2 = nn.Sequential(
            nn.Conv2d(reduce_dim+1, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True))
        self.int3 = nn.Sequential(
            nn.Conv2d(reduce_dim+1, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True))


    def forward(self, x, s_x, nom, s_y, y=None):
        """ x : query   image  [b,3,473,473]            y: query mask [b,473,473]
           s_x: support image  [b,shot,3, 473,473]  s_y: support mask [b, shot, 473,473]
        """
        x_size = x.size()
        h, w = x_size[2], x_size[3]

        """ Get Query feature"""

        query_feat_0,query_feat_1,query_feat_2,query_feat_3 = self.query_encoder(x)
        nom_feat_0, nom_feat_1, nom_feat_2, nom_feat_3 = self.query_encoder(nom)

        """ Get Support feature"""
        foreground_feat_list_0,foreground_feat_list_1,foreground_feat_list_2,foreground_feat_list_3 ,mask_1= self.support_encoder(s_x, s_y)

        # n-shot,逐元素相加求均值

        supp_feat_0 = foreground_feat_list_0[0]
        if self.shot > 1:
            for i in range(1, len(foreground_feat_list_0)):
                supp_feat_0 += foreground_feat_list_0[i]
            supp_feat_0 /= len(foreground_feat_list_0)
        supp_feat_1 = foreground_feat_list_1[0]
        if self.shot > 1:
            for i in range(1, len(foreground_feat_list_1)):
                supp_feat_1 += foreground_feat_list_1[i]
            supp_feat_1 /= len(foreground_feat_list_1)
        supp_feat_2 = foreground_feat_list_2[0]
        if self.shot > 1:
            for i in range(1, len(foreground_feat_list_2)):
                supp_feat_2 += foreground_feat_list_2[i]
            supp_feat_2 /= len(foreground_feat_list_2)
        supp_feat_3 = foreground_feat_list_3[0]
        if self.shot > 1:
            for i in range(1, len(foreground_feat_list_3)):
                supp_feat_3 += foreground_feat_list_3[i]
            supp_feat_3 /= len(foreground_feat_list_3)
        # baseline
        # out = torch.cat((query_feat_1,supp_feat_1),dim=1)
        # out = self.down(out)
        supp_feat_1_fpn = self.Tripletencoder(supp_feat_1,supp_feat_2,supp_feat_3)
        query_feat_1_fpn = self.Tripletencoder(query_feat_1, query_feat_2, query_feat_3)
        nom_feat_1_fpn = self.Tripletencoder(nom_feat_1, nom_feat_2, nom_feat_3)

        feat_1 = self.GCN(supp_feat_1_fpn, query_feat_1_fpn)
        feat_2 = self.GCN(nom_feat_1_fpn, query_feat_1_fpn)

        corr_a1 = similarity(supp_feat_1_fpn, query_feat_1_fpn)
        corr_a2 = similarity(nom_feat_1_fpn, query_feat_1_fpn)
        #
        out_1= self.cls_f(feat_1,w,h)
        out_2 = self.cls_f(feat_2, w, h)

        out =out_1-out_2
        out1=torch.cat((corr_a1,corr_a2),1)
        out =out + (self.trip * out1)
        out_mask2 = out.max(1)[1]
        out_mask2=out_mask2.unsqueeze(1)
        out_mask3=1-out_mask2
        out = F.interpolate(out, size=(h, w), mode='bilinear', align_corners=True)
        out_mask =out.max(1)[1]
        #
        # #####################
        h, w = query_feat_1.shape[-2:][0], query_feat_1.shape[-2:][1]

        area_s = F.avg_pool2d(mask_1, supp_feat_1.shape[-2:]) * h * w + 0.0005
        z_s = F.avg_pool2d(input=supp_feat_1,
                         kernel_size=supp_feat_1.shape[-2:]) * h * w / area_s

        out_mask_flt = out_mask3.float()
        out_mask_flt_qj = out_mask2.float()

        area = F.avg_pool2d(out_mask_flt, query_feat_1.shape[-2:]) * h * w + 0.0005
        area_qj = F.avg_pool2d(out_mask_flt_qj, query_feat_1.shape[-2:]) * h * w + 0.0005

        z = out_mask_flt * query_feat_1
        z_qj = out_mask_flt_qj * query_feat_1


        z = F.avg_pool2d(input=z,
                         kernel_size=query_feat_1.shape[-2:]) * h * w / area
        z_qj = F.avg_pool2d(input=z_qj,
                         kernel_size=query_feat_1.shape[-2:]) * h * w / area_qj
        z_nom = F.avg_pool2d(input=nom_feat_1,
                         kernel_size=nom_feat_1.shape[-2:])

        if self.training:
            aux_loss = embedding_distance(z, z_nom)
            aux_loss2 = embedding_distance(z_qj, z_s)
            main_loss = self.criterion(out, y.long())
            loss = main_loss +0.2*(aux_loss+aux_loss2)
            return out_mask, loss
        else:
            return out  # [b,2,473,473]

    def cls_f(self,query_feat_x,w,h):
        # query_feat_x = self.res4(query_feat_x)
        query_feat_x = self.res1(query_feat_x) + query_feat_x # [b,256,60,60]
        query_feat_x = self.res2(query_feat_x) + query_feat_x  # [b,256,60,60]
        query_feat_x = self.res3(query_feat_x) + query_feat_x  # [b,256,60,60]
        query_feat_x = self.ASPP(query_feat_x) # [b,256,60,60]
        out = self.cls(query_feat_x)#[b,256,60,60]

        """Output Part"""
        return  out

    def query_encoder(self, x):

        query_feat_0 = self.layer0(x)  # [64,100,100]
        query_feat_1 = self.layer1(query_feat_0)  # [128,50,50]
        query_feat_2 = self.layer2(query_feat_1)  # [256,13,13]
        query_feat_3 = self.layer3(query_feat_2)  # [512,13,13]

        if self.vgg:
            query_feat_3 = F.interpolate(query_feat_3, size=(query_feat_2.size(2), query_feat_2.size(3)),
                                         mode='bilinear', align_corners=True)

        return query_feat_0,query_feat_1,query_feat_2,query_feat_3

    def support_encoder(self, s_x, s_y):
        mask_list = []  # [shot,b,1,473,473]
        foreground_feat_list_0 = []
        foreground_feat_list_1 = []
        foreground_feat_list_2 = []
        foreground_feat_list_3 = []

        # background_feat_list = []
        for i in range(self.shot):
            mask = (s_y[:, i, :, :] == 1).float().unsqueeze(1)
            mask_list.append(mask)

            supp_feat_0 = self.layer0(s_x[:, i, :, :, :])
            supp_feat_1 = self.layer1(supp_feat_0)
            supp_feat_2 = self.layer2(supp_feat_1)
            supp_feat_3 = self.layer3(supp_feat_2)
            mask_0 = F.interpolate(mask, size=(supp_feat_0.size(2), supp_feat_0.size(3)), mode='bilinear',
                                 align_corners=True)
            mask_1 = F.interpolate(mask, size=(supp_feat_1.size(2), supp_feat_1.size(3)), mode='bilinear',
                                 align_corners=True)
            mask_2 = F.interpolate(mask, size=(supp_feat_2.size(2), supp_feat_2.size(3)), mode='bilinear',
                                 align_corners=True)
            mask_3 = F.interpolate(mask, size=(supp_feat_3.size(2), supp_feat_3.size(3)), mode='bilinear',
                                 align_corners=True)

            if self.vgg:
                supp_feat_3 = F.interpolate(supp_feat_3, size=(supp_feat_2.size(2), supp_feat_2.size(3)),
                                            mode='bilinear', align_corners=True)

            # supp_feat = torch.cat([supp_feat_2, supp_feat_3], dim=1)
            # supp_feat = self.down_supp(supp_feat)
            supp_feat_0 = supp_feat_0 * mask_0
            supp_feat_1 = supp_feat_1 * mask_1
            supp_feat_2 = supp_feat_2 * mask_2
            supp_feat_3 = supp_feat_3 * mask_3
            # supp_feat_1 = torch.cat((supp_feat_1,supp_feat_0),dim=1)
            # supp_feat_1 = self.down_supp(supp_feat_1)
            foreground_feat_list_0.append(supp_feat_0)
            foreground_feat_list_1.append(supp_feat_1)
            foreground_feat_list_2.append(supp_feat_2)
            foreground_feat_list_3.append(supp_feat_3)

            # background_feat_list.append(back_feat)

        return foreground_feat_list_0,foreground_feat_list_1,foreground_feat_list_2,foreground_feat_list_3, mask_1

