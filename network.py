import torch
import torch.nn as nn
from torchvision import models

class HashEmbNet(nn.Module):
    def __init__(self, hash_bit, pretrained=True):
        super(HashEmbNet, self).__init__()

        model_alexnet = models.alexnet(pretrained=pretrained)
        # self.features = model_alexnet.features
        cl1 = nn.Linear(2048, 1024)
        cl1.weight = nn.Parameter(model_alexnet.classifier[1].weight[0:1024,0:2048])
        cl1.bias = nn.Parameter(model_alexnet.classifier[1].bias[0:1024])

        cl2 = nn.Linear(1024, 1024)
        cl2.weight = nn.Parameter(model_alexnet.classifier[4].weight[0:1024,0:1024])
        cl2.bias = nn.Parameter(model_alexnet.classifier[4].bias[0:1024])

        self.hash_layer = nn.Sequential(
            nn.Dropout(),
            cl1,
            nn.ReLU(inplace=True),
            nn.Dropout(),
            cl2,
            nn.ReLU(inplace=True),
            nn.Linear(1024, hash_bit),
        )

    def forward(self, x):
        # x = self.features(x)
        x = x.view(x.size(0), 2048)
        x = self.hash_layer(x)
        return x

class HashEmbNet_Scratch(nn.Module):
    def __init__(self, hash_bit, pretrained=True):
        super(HashEmbNet_Scratch, self).__init__()

        # model_alexnet = models.alexnet(pretrained=pretrained)
        # self.features = model_alexnet.features
        conv1 = nn.Conv2d(in_channels=2048,out_channels= 2048,kernel_size=1,stride=1,padding=0)
        nn.init.xavier_normal_(conv1.weight.data,gain=1.0)
        conv1.bias.data.fill_(0.0)
        cl1 = nn.Linear(2048, 1024)
        nn.init.xavier_normal_(cl1.weight.data,gain=1.0) 
        cl1.bias.data.fill_(0.0)
        # cl1.weight = nn.Parameter(model_alexnet.classifier[1].weight[0:1024,0:2048])
        # cl1.bias = nn.Parameter(model_alexnet.classifier[1].bias[0:1024])

        cl2 = nn.Linear(1024, 1024)
        nn.init.xavier_normal_(cl2.weight.data,gain=1.0) 
        cl2.bias.data.fill_(0.0)
        # cl2.weight = nn.Parameter(model_alexnet.classifier[4].weight[0:1024,0:1024])
        # cl2.bias = nn.Parameter(model_alexnet.classifier[4].bias[0:1024])
        self.conv_layer = nn.Sequential(
            conv1,
            nn.BatchNorm2d(2048),
            nn.ReLU(inplace=True),
        )

        self.hash_layer = nn.Sequential(
            # conv1,
            # nn.BatchNorm1d(2048),
            # nn.ReLU(inplace=True),
            nn.Dropout(),
            cl1,
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            cl2,
            nn.ReLU(inplace=True),
            nn.Linear(1024, hash_bit),
        )

    def forward(self, x):
        # x = self.features(x)
        x = x.view(x.size(0), 2048,1,1)
        x = self.conv_layer(x)
        x = x.view(x.size(0), 2048)
        x = self.hash_layer(x)
        # x=nn.functional.normalize(x)
        return x

class HashTransNet(nn.Module):
    def __init__(self, hash_bit, pretrained=True):
        super(HashTransNet, self).__init__()

        # model_alexnet = models.alexnet(pretrained=pretrained)
        # self.features = model_alexnet.features
        net_HashEmb = torch.load("save/DSH/GLDv2-0-test0.024-model_net.pt")
        # conv1.weight = net_HashEmb.conv_layer[0].weight
        # conv1.bias = net_HashEmb.conv_layer[0].bias

        conv1 = nn.Conv2d(in_channels=2048,out_channels= 2048,kernel_size=1,stride=1,padding=0)
        cl1 = nn.Linear(2048, 2048)
        # nn.init.xavier_normal_(cl1.weight.data,gain=1.0) 
        # cl1.bias.data.fill_(0.0)
        # cl1.weight = nn.Parameter(model_alexnet.classifier[1].weight[0:1024,0:2048])
        # cl1.bias = nn.Parameter(model_alexnet.classifier[1].bias[0:1024])
        cl2 = nn.Linear(2048, 4096)
        # nn.init.xavier_normal_(cl2.weight.data,gain=1.0) 
        # cl2.bias.data.fill_(0.0)
        # cl2.weight = nn.Parameter(model_alexnet.classifier[4].weight[0:1024,0:1024])
        # cl2.bias = nn.Parameter(model_alexnet.classifier[4].bias[0:1024])
        self.conv_layer = nn.Sequential(
            conv1,
            nn.BatchNorm2d(2048),
            nn.ReLU(inplace=True),
        )
        self.conv_layer[0].weight = net_HashEmb.conv_layer[0].weight
        self.conv_layer[0].bias = net_HashEmb.conv_layer[0].bias
        self.conv_layer[1].weight = net_HashEmb.conv_layer[1].weight
        self.conv_layer[1].bias = net_HashEmb.conv_layer[1].bias

        self.hash_layer = nn.Sequential(
            nn.Dropout(),
            cl1,
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            cl2,
            nn.ReLU(inplace=True),
            nn.Linear(4096, hash_bit),
        )

        self.hash_layer[1].weight = nn.Parameter(net_HashEmb.hash_layer[1].weight.data.repeat(2,1)) #cl1
        self.hash_layer[1].bias = nn.Parameter(net_HashEmb.hash_layer[1].bias.data.repeat(2))
        self.hash_layer[2].weight = nn.Parameter(net_HashEmb.hash_layer[2].weight.data.repeat(2)) #BN(2048)
        self.hash_layer[2].bias = nn.Parameter(net_HashEmb.hash_layer[2].bias.data.repeat(2))
        self.hash_layer[5].weight = nn.Parameter(net_HashEmb.hash_layer[5].weight.data.repeat(4,2)) #cl2(2048)
        self.hash_layer[5].bias = nn.Parameter(net_HashEmb.hash_layer[5].bias.data.repeat(4))


    def forward(self, x):
        # x = self.features(x)
        x = x.view(x.size(0), 2048,1,1)
        x = self.conv_layer(x)
        x = x.view(x.size(0), 2048)
        x = self.hash_layer(x)
        # x=nn.functional.normalize(x)
        return x

class HashTransNet1(nn.Module):
    def __init__(self, hash_bit, pretrained=True):
        super(HashTransNet1, self).__init__()

        # model_alexnet = models.alexnet(pretrained=pretrained)
        # self.features = model_alexnet.features
        # net_HashEmb = torch.load("save/DSH/GLDv2-0-test0.024-model_net.pt")
        model_path = "save/DSH/GLDv2-0-test0.149-end-model.pt"
        net_HashDTH = torch.load(model_path)
        # conv1.weight = net_HashEmb.conv_layer[0].weight
        # conv1.bias = net_HashEmb.conv_layer[0].bias

        conv1 = nn.Conv2d(in_channels=2048,out_channels= 2048,kernel_size=1,stride=1,padding=0)
        cl1 = nn.Linear(2048, 2048)
        # nn.init.xavier_normal_(cl1.weight.data,gain=1.0) 
        # cl1.bias.data.fill_(0.0)
        # cl1.weight = nn.Parameter(model_alexnet.classifier[1].weight[0:1024,0:2048])
        # cl1.bias = nn.Parameter(model_alexnet.classifier[1].bias[0:1024])
        cl2 = nn.Linear(2048, 4096)
        self.conv_layer = nn.Sequential(
            conv1,
            nn.BatchNorm2d(2048),
            nn.Tanh(),
        )
        # self.conv_layer[0].weight = net_HashEmb.conv_layer[0].weight
        # self.conv_layer[0].bias = net_HashEmb.conv_layer[0].bias
        # self.conv_layer[1].weight = net_HashEmb.conv_layer[1].weight
        # self.conv_layer[1].bias = net_HashEmb.conv_layer[1].bias

        self.hash_layer = nn.Sequential(
            cl1,
            nn.BatchNorm1d(2048),
            nn.Tanh(),
            cl2,
            nn.Tanh(),
            nn.Linear(4096, hash_bit),
        )
        self.conv_layer[0].weight = nn.Parameter(net_HashDTH["conv_layer.0.weight"])
        self.conv_layer[0].bias = nn.Parameter(net_HashDTH["conv_layer.0.bias"])
        self.conv_layer[1].weight = nn.Parameter(net_HashDTH["conv_layer.1.weight"])
        self.conv_layer[1].bias = nn.Parameter(net_HashDTH["conv_layer.1.bias"])

        self.hash_layer[0].weight = nn.Parameter(net_HashDTH["hash_layer.1.weight"])#fc1
        self.hash_layer[0].bias = nn.Parameter(net_HashDTH["hash_layer.1.bias"])
        self.hash_layer[1].weight = nn.Parameter(net_HashDTH["hash_layer.2.weight"])#bn1
        self.hash_layer[1].bias = nn.Parameter(net_HashDTH["hash_layer.2.bias"])
        self.hash_layer[3].weight = nn.Parameter(net_HashDTH["hash_layer.5.weight"])#fc2
        self.hash_layer[3].bias = nn.Parameter(net_HashDTH["hash_layer.5.bias"])
        self.hash_layer[5].weight = nn.Parameter(net_HashDTH["hash_layer.7.weight"])#fc3
        self.hash_layer[5].bias = nn.Parameter(net_HashDTH["hash_layer.7.bias"])


    def forward(self, x):
        # x = self.features(x)
        x = x.view(x.size(0), 2048,1,1)
        x = self.conv_layer(x)
        x = x.view(x.size(0), 2048)
        x = self.hash_layer(x)
        # x=nn.functional.normalize(x)
        return x

class HashTransNet2(nn.Module):
    def __init__(self, hash_bit, pretrained=True):
        super(HashTransNet2, self).__init__()
        model_path = "save/DSH/GLDv2-0-test0.161-net1-model.pt"
        net_HashDTH = torch.load(model_path)

        conv1 = nn.Conv2d(in_channels=2048,out_channels= 2048,kernel_size=1,stride=1,padding=0)
        cl1 = nn.Linear(2048, 2048)
        # nn.init.xavier_normal_(cl1.weight.data,gain=1.0) 
        # cl1.bias.data.fill_(0.0)
        # cl1.weight = nn.Parameter(model_alexnet.classifier[1].weight[0:1024,0:2048])
        # cl1.bias = nn.Parameter(model_alexnet.classifier[1].bias[0:1024])
        cl2 = nn.Linear(2048, 4096)
        cl3 = nn.Linear(4096, 4096)
        self.conv_layer = nn.Sequential(
            conv1,
            nn.BatchNorm2d(2048),
            nn.Tanh(),
        )

        self.hash_layer = nn.Sequential(
            cl1,
            nn.BatchNorm1d(2048),
            nn.Tanh(),
            cl2,
            nn.Tanh(),
            cl3,
            nn.Tanh(),
            nn.Linear(4096, hash_bit),  
        )
        self.conv_layer[0].weight = nn.Parameter(net_HashDTH["conv_layer.0.weight"])
        self.conv_layer[0].bias = nn.Parameter(net_HashDTH["conv_layer.0.bias"])
        self.conv_layer[1].weight = nn.Parameter(net_HashDTH["conv_layer.1.weight"])
        self.conv_layer[1].bias = nn.Parameter(net_HashDTH["conv_layer.1.bias"])

        self.hash_layer[0].weight = nn.Parameter(net_HashDTH["hash_layer.0.weight"])#fc1
        self.hash_layer[0].bias = nn.Parameter(net_HashDTH["hash_layer.0.bias"])
        self.hash_layer[1].weight = nn.Parameter(net_HashDTH["hash_layer.1.weight"])#bn1
        self.hash_layer[1].bias = nn.Parameter(net_HashDTH["hash_layer.1.bias"])
        self.hash_layer[3].weight = nn.Parameter(net_HashDTH["hash_layer.3.weight"])#fc2
        self.hash_layer[3].bias = nn.Parameter(net_HashDTH["hash_layer.3.bias"])
        self.hash_layer[5].weight = nn.Parameter(net_HashDTH["hash_layer.3.weight"].repeat(1,2))
        self.hash_layer[5].bias = nn.Parameter(net_HashDTH["hash_layer.3.bias"])
        # nn.init.xavier_normal_(self.hash_layer[5].weight.data,gain=1.0)
        # self.hash_layer[5].bias.data.fill_(0.0)
        self.hash_layer[7].weight = nn.Parameter(net_HashDTH["hash_layer.5.weight"])#fc3
        self.hash_layer[7].bias = nn.Parameter(net_HashDTH["hash_layer.5.bias"])


    def forward(self, x):
        x = x.view(x.size(0), 2048,1,1)
        x = self.conv_layer(x)
        x = x.view(x.size(0), 2048)
        x = self.hash_layer(x)
        return x

class HashTransNet3(nn.Module):
    def __init__(self, hash_bit, pretrained=True):
        super(HashTransNet3, self).__init__()
        model_path = "save/DSH/GLDv2-0-test0.161-net1-model.pt"
        net_HashDTH = torch.load(model_path)

        conv1 = nn.Conv2d(in_channels=2048,out_channels= 2048,kernel_size=1,stride=1,padding=0)
        cl1 = nn.Linear(2048, 2048)
        # nn.init.xavier_normal_(cl1.weight.data,gain=1.0) 
        # cl1.bias.data.fill_(0.0)
        # cl1.weight = nn.Parameter(model_alexnet.classifier[1].weight[0:1024,0:2048])
        # cl1.bias = nn.Parameter(model_alexnet.classifier[1].bias[0:1024])
        cl2 = nn.Linear(2048, 4096)
        cl3 = nn.Linear(4096, 4096)
        self.conv_layer = nn.Sequential(
            conv1,
            nn.BatchNorm2d(2048),
            nn.Tanh(),
        )

        self.hash_layer = nn.Sequential(
            cl1,
            nn.BatchNorm1d(2048),
            nn.Tanh(),
            cl2,
            nn.Tanh(),
            # nn.Dropout(p=0.2),
            cl3,
            nn.Tanh(),
            nn.Linear(4096, hash_bit),  
            nn.Tanh(),
        )
        self.conv_layer[0].weight = nn.Parameter(net_HashDTH["conv_layer.0.weight"])
        self.conv_layer[0].bias = nn.Parameter(net_HashDTH["conv_layer.0.bias"])
        self.conv_layer[1].weight = nn.Parameter(net_HashDTH["conv_layer.1.weight"])
        self.conv_layer[1].bias = nn.Parameter(net_HashDTH["conv_layer.1.bias"])

        self.hash_layer[0].weight = nn.Parameter(net_HashDTH["hash_layer.0.weight"])#fc1
        self.hash_layer[0].bias = nn.Parameter(net_HashDTH["hash_layer.0.bias"])
        self.hash_layer[1].weight = nn.Parameter(net_HashDTH["hash_layer.1.weight"])#bn1
        self.hash_layer[1].bias = nn.Parameter(net_HashDTH["hash_layer.1.bias"])
        self.hash_layer[3].weight = nn.Parameter(net_HashDTH["hash_layer.3.weight"])#fc2
        self.hash_layer[3].bias = nn.Parameter(net_HashDTH["hash_layer.3.bias"])
        # self.hash_layer[6].weight = nn.Parameter(net_HashDTH["hash_layer.3.weight"].repeat(1,2))
        # self.hash_layer[6].bias = nn.Parameter(net_HashDTH["hash_layer.3.bias"])
        nn.init.xavier_normal_(self.hash_layer[5].weight.data,gain=1.0)
        self.hash_layer[5].bias.data.fill_(0.0)
        self.hash_layer[7].weight = nn.Parameter(net_HashDTH["hash_layer.5.weight"])#fc3
        self.hash_layer[7].bias = nn.Parameter(net_HashDTH["hash_layer.5.bias"])


    def forward(self, x):
        x = x.view(x.size(0), 2048,1,1)
        x = self.conv_layer(x)
        x = x.view(x.size(0), 2048)
        x = self.hash_layer(x)
        return x

class HashTransNet4(nn.Module):
    def __init__(self, hash_bit, pretrained=True):
        super(HashTransNet4, self).__init__()
        model_path = "save/DSH/GLDv2-0-test0.1667-net3-model.pt"
        net_HashDTH = torch.load(model_path)

        conv1 = nn.Conv2d(in_channels=2048,out_channels= 2048,kernel_size=1,stride=1,padding=0)
        cl1 = nn.Linear(2048, 2048)
        cl2 = nn.Linear(2048, 4096)
        cl3 = nn.Linear(4096, 4096)
        cl4 = nn.Linear(4096, 4096)
        self.conv_layer = nn.Sequential(
            conv1,
            nn.BatchNorm2d(2048),
            nn.Tanh(),
        )

        self.hash_layer = nn.Sequential(
            cl1,
            nn.BatchNorm1d(2048),
            nn.Tanh(),
            cl2,
            nn.Tanh(),
            # nn.Dropout(p=0.2),
            cl3,
            nn.BatchNorm1d(4096),
            # nn.Sigmoid(),
            nn.Tanh(),
            cl4,
            nn.Tanh(),
            nn.Linear(4096, hash_bit),  
            nn.Tanh(),
        )
        self.conv_layer[0].weight = nn.Parameter(net_HashDTH["conv_layer.0.weight"])
        self.conv_layer[0].bias = nn.Parameter(net_HashDTH["conv_layer.0.bias"])
        self.conv_layer[1].weight = nn.Parameter(net_HashDTH["conv_layer.1.weight"])
        self.conv_layer[1].bias = nn.Parameter(net_HashDTH["conv_layer.1.bias"])

        self.hash_layer[0].weight = nn.Parameter(net_HashDTH["hash_layer.0.weight"])#fc1
        self.hash_layer[0].bias = nn.Parameter(net_HashDTH["hash_layer.0.bias"])
        self.hash_layer[1].weight = nn.Parameter(net_HashDTH["hash_layer.1.weight"])#bn1
        self.hash_layer[1].bias = nn.Parameter(net_HashDTH["hash_layer.1.bias"])
        self.hash_layer[3].weight = nn.Parameter(net_HashDTH["hash_layer.3.weight"])#fc2
        self.hash_layer[3].bias = nn.Parameter(net_HashDTH["hash_layer.3.bias"])
        self.hash_layer[5].weight = nn.Parameter(net_HashDTH["hash_layer.5.weight"])
        self.hash_layer[5].bias = nn.Parameter(net_HashDTH["hash_layer.5.bias"])
        nn.init.xavier_normal_(self.hash_layer[8].weight.data,gain=1.0)
        self.hash_layer[8].bias.data.fill_(0.0)
        self.hash_layer[10].weight = nn.Parameter(net_HashDTH["hash_layer.7.weight"])#fc3
        self.hash_layer[10].bias = nn.Parameter(net_HashDTH["hash_layer.7.bias"])


    def forward(self, x):
        x = x.view(x.size(0), 2048,1,1)
        x = self.conv_layer(x)
        x = x.view(x.size(0), 2048)
        x = self.hash_layer(x)
        return x


class AlexNet(nn.Module):
    def __init__(self, hash_bit, pretrained=True):
        super(AlexNet, self).__init__()

        model_alexnet = models.alexnet(pretrained=pretrained)
        self.features = model_alexnet.features
        cl1 = nn.Linear(256 * 6 * 6, 4096)
        cl1.weight = model_alexnet.classifier[1].weight
        cl1.bias = model_alexnet.classifier[1].bias

        cl2 = nn.Linear(4096, 4096)
        cl2.weight = model_alexnet.classifier[4].weight
        cl2.bias = model_alexnet.classifier[4].bias

        self.hash_layer = nn.Sequential(
            nn.Dropout(),
            cl1,
            nn.ReLU(inplace=True),
            nn.Dropout(),
            cl2,
            nn.ReLU(inplace=True),
            nn.Linear(4096, hash_bit),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.hash_layer(x)
        return x


resnet_dict = {"ResNet18": models.resnet18, "ResNet34": models.resnet34, "ResNet50": models.resnet50,
               "ResNet101": models.resnet101, "ResNet152": models.resnet152}


class ResNet(nn.Module):
    def __init__(self, hash_bit, res_model="ResNet50"):
        super(ResNet, self).__init__()
        model_resnet = resnet_dict[res_model](pretrained=True)
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool
        self.feature_layers = nn.Sequential(self.conv1, self.bn1, self.relu, self.maxpool, \
                                            self.layer1, self.layer2, self.layer3, self.layer4, self.avgpool)

        self.hash_layer = nn.Linear(model_resnet.fc.in_features, hash_bit)
        self.hash_layer.weight.data.normal_(0, 0.01)
        self.hash_layer.bias.data.fill_(0.0)

    def forward(self, x):
        x = self.feature_layers(x)
        x = x.view(x.size(0), -1)
        y = self.hash_layer(x)
        return y
