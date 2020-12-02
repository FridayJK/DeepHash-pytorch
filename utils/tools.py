import numpy as np
import torch.utils.data as util_data
from torchvision import transforms
import torch
from PIL import Image
from tqdm import tqdm
import torchvision.datasets as dsets
import os

def config_dataset(config):
    if "cifar" in config["dataset"]:
        config["topK"] = -1
        config["n_class"] = 10
    elif "GLDv2" in config["dataset"]:
        config["topK"] = 4
        config["n_class"] = 101302 #0-101301
    elif config["dataset"] in ["nuswide_21", "nuswide_21_m"]:
        config["topK"] = 5000
        config["n_class"] = 21
    elif config["dataset"] == "nuswide_81_m":
        config["topK"] = 5000
        config["n_class"] = 81
    elif config["dataset"] == "coco":
        config["topK"] = 5000
        config["n_class"] = 80
    elif config["dataset"] == "imagenet":
        config["topK"] = 1000
        config["n_class"] = 100
    elif config["dataset"] == "mirflickr":
        config["topK"] = -1
        config["n_class"] = 38
    elif config["dataset"] == "voc2012":
        config["topK"] = -1
        config["n_class"] = 20

    config["data_path"] = "/dataset/" + config["dataset"] + "/"
    if config["dataset"] == "nuswide_21":
        config["data_path"] = "/dataset/NUS-WIDE/"
    if config["dataset"] in ["nuswide_21_m", "nuswide_81_m"]:
        config["data_path"] = "/dataset/nus_wide_m/"
    if config["dataset"] == "coco":
        config["data_path"] = "/dataset/COCO_2014/"
    if config["dataset"] == "voc2012":
        config["data_path"] = "/dataset/"
    if config["dataset"] == "GLDv2-0":
        config["data_path"] = "/home/data/GLDv2-0/"
    config["data"] = {
        "train_set": {"list_path": "./data/" + config["dataset"] + "/train.txt", "batch_size": config["batch_size"]},
        "database": {"list_path": "./data/" + config["dataset"] + "/database.txt", "batch_size": config["batch_size"]},
        "test": {"list_path": "./data/" + config["dataset"] + "/test.txt", "batch_size": config["batch_size"]}}
    return config


class ImageList(object):

    def __init__(self, data_path, image_list, transform):
        self.imgs = [(data_path + val.split()[0], np.array([int(la) for la in val.split()[1:]])) for val in image_list]
        self.transform = transform

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = Image.open(path).convert('RGB')
        img = self.transform(img)
        return img, target, index

    def __len__(self):
        return len(self.imgs)


def image_transform(resize_size, crop_size, data_set):
    if data_set == "train_set":
        step = [transforms.RandomHorizontalFlip(), transforms.RandomCrop(crop_size)]
    else:
        step = [transforms.CenterCrop(crop_size)]
    return transforms.Compose([transforms.Resize(resize_size)]
                              + step +
                              [transforms.ToTensor(),
                               transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225])
                               ])


class MyCIFAR10(dsets.CIFAR10):
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        img = self.transform(img)
        target = np.eye(10, dtype=np.int8)[np.array(target)]
        return img, target, index

class MyGLDv2(object):
    def __init__(self, data, targets, transform,class_num=0,train_flag=0):
        self.data = data
        self.targets = targets
        self.transform = transform
        self.class_num = class_num
        self.train_flag = train_flag
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        # img = Image.fromarray(img)
        # img = self.transform(img)
        if(self.train_flag):
            # target = np.eye(self.class_num, dtype=np.int8)[np.array(target)]   #?
            target = np.array(target)
            return img, target, index
        else:
            target = np.array(target)
            return img, target, index

    def __len__(self):
        return self.data.shape[0]


def cifar_dataset(config):
    batch_size = config["batch_size"]

    train_size = 500
    test_size = 100

    if config["dataset"] == "cifar10-2":
        train_size = 5000
        test_size = 1000

    transform = transforms.Compose([
        transforms.Resize(config["crop_size"]),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Dataset
    train_dataset = MyCIFAR10(root='/dataset/cifar/',
                              train=True,
                              transform=transform,
                              download=True)

    test_dataset = MyCIFAR10(root='/dataset/cifar/',
                             train=False,
                             transform=transform)

    database_dataset = MyCIFAR10(root='/dataset/cifar/',
                                 train=False,
                                 transform=transform)

    X = np.concatenate((train_dataset.data, test_dataset.data))
    L = np.concatenate((np.array(train_dataset.targets), np.array(test_dataset.targets)))

    first = True
    for label in range(10):
        index = np.where(L == label)[0]

        N = index.shape[0]
        perm = np.random.permutation(N)
        index = index[perm]

        if first:
            test_index = index[:test_size]
            train_index = index[test_size: train_size + test_size]
            database_index = index[train_size + test_size:]
        else:
            test_index = np.concatenate((test_index, index[:test_size]))
            train_index = np.concatenate((train_index, index[test_size: train_size + test_size]))
            database_index = np.concatenate((database_index, index[train_size + test_size:]))
        first = False

    if config["dataset"] == "cifar10":
        # test:1000, train:5000, database:54000
        pass
    elif config["dataset"] == "cifar10-1":
        # test:1000, train:5000, database:59000
        database_index = np.concatenate((train_index, database_index))
    elif config["dataset"] == "cifar10-2":
        # test:10000, train:50000, database:50000
        database_index = train_index

    train_dataset.data = X[train_index]
    train_dataset.targets = L[train_index]
    test_dataset.data = X[test_index]
    test_dataset.targets = L[test_index]
    database_dataset.data = X[database_index]
    database_dataset.targets = L[database_index]

    print("train_dataset", train_dataset.data.shape[0])
    print("test_dataset", test_dataset.data.shape[0])
    print("database_dataset", database_dataset.data.shape[0])

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=0)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              num_workers=0)

    database_loader = torch.utils.data.DataLoader(dataset=database_dataset,
                                                  batch_size=batch_size,
                                                  shuffle=False,
                                                  num_workers=0)

    return train_loader, test_loader, database_loader, \
           train_index.shape[0], test_index.shape[0], database_index.shape[0]

def gldv2_dataset(config):
    batch_size = config["batch_size"]

    # train_size = 500
    # test_size = 100

    # if config["dataset"] == "cifar10-2":
    #     train_size = 5000
    #     test_size = 1000

    transform = transforms.Compose([
        # transforms.Resize(config["crop_size"]),
        transforms.ToTensor(),
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_data = np.load(config["data_path"]+"GLDv2_index_r-mac_feat2048.npy")
    train_labels = np.load(config["data_path"]+"index_land_ids_labels.npy")
    # train_data = 
    test_data = np.load(config["data_path"]+"GLDv2_test_r-mac_feat2048_clean.npy")
    test_labels = np.load(config["data_path"]+"test_land_ids_label.npy")


    #排除掉 < thre 个数的类别
    hole_test = np.zeros((train_labels.max()+1),dtype=np.int)
    for i in range(train_labels.shape[0]):
        hole_test[train_labels[i]]+=1

    thre = 5
    first = True
    land_id_filter = np.where(hole_test >= thre)[0]
    if(os.path.exists(config["data_path"]+str(thre)+"_thre_data.npy")):
        X = np.load(config["data_path"]+str(thre)+"_thre_data.npy")
        L = np.load(config["data_path"]+str(thre)+"_thre_label.npy")
        print("X shape",land_id_filter.shape[0],X.shape[0],X.shape[1])
    else:
        filter_num = 0
        for i,land_id in enumerate(land_id_filter):
            index = np.where(train_labels==land_id)[0]
            filter_num += index.shape[0]
        L = np.zeros((filter_num), dtype=np.int)
        X = np.zeros((filter_num, train_data.shape[1]), dtype=np.float32)
        filter_num = 0
        for i,land_id in enumerate(land_id_filter):
            index = np.where(train_labels==land_id)[0]
            train_labels[index] = i #修改训练数据id label
            L[filter_num:(filter_num+index.shape[0])] = train_labels[index]
            X[filter_num:(filter_num+index.shape[0]),:] = train_data[index]
            filter_num += index.shape[0]
            # if first:
            #     train_labels[index] = i
            #     L = train_labels[index]
            #     X = train_data[index]
            #     first = False
            # else:
            #     train_labels[index] = i
            #     L = np.concatenate((L,train_labels[index]))
            #     X = np.concatenate((X,train_data[index]))
            print("X shape",land_id_filter.shape[0],i,X.shape[0],filter_num)

        np.save(config["data_path"]+str(thre)+"_thre_data.npy",X)
        np.save(config["data_path"]+str(thre)+"_thre_label.npy",L)


        


    # Dataset
    train_dataset = MyGLDv2(data=X,
                              targets=L,
                              transform=transform,
                              class_num = land_id_filter.shape[0],
                              train_flag = 1)

    test_dataset = MyGLDv2(data=test_data,
                             targets=test_labels,
                             transform=transform,
                             class_num = hole_test.shape[0])

    database_dataset = MyGLDv2(data=train_data,
                              targets=train_labels,
                              transform=transform,
                              class_num = hole_test.shape[0])

    # database_dataset = MyGLDv2(data=train_data[0:10000],
    #                           targets=train_labels[0:10000],
    #                           transform=transform,
    #                           class_num = hole_test.shape[0])

    print("train_dataset", train_dataset.data.shape[0])
    print("test_dataset", test_dataset.data.shape[0])
    print("database_dataset", database_dataset.data.shape[0])   

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=0)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              num_workers=0)

    database_loader = torch.utils.data.DataLoader(dataset=database_dataset,
                                                  batch_size=batch_size,
                                                  shuffle=False,
                                                  num_workers=0)

    config["n_class"] = land_id_filter.shape[0]

    return train_loader, test_loader, database_loader, \
           train_dataset.data.shape[0], test_dataset.data.shape[0], database_dataset.data.shape[0]

def get_data(config):
    if "cifar" in config["dataset"]:
        return cifar_dataset(config)

    if "GLDv2" in config["dataset"]:
        return gldv2_dataset(config)

    dsets = {}
    dset_loaders = {}
    data_config = config["data"]

    for data_set in ["train_set", "test", "database"]:
        dsets[data_set] = ImageList(config["data_path"],
                                    open(data_config[data_set]["list_path"]).readlines(),
                                    transform=image_transform(config["resize_size"], config["crop_size"], data_set))
        print(data_set, len(dsets[data_set]))
        dset_loaders[data_set] = util_data.DataLoader(dsets[data_set],
                                                      batch_size=data_config[data_set]["batch_size"],
                                                      shuffle=True, num_workers=4)

    return dset_loaders["train_set"], dset_loaders["test"], dset_loaders["database"], \
           len(dsets["train_set"]), len(dsets["test"]), len(dsets["database"])

def compute_result(dataloader, net, device):
    bs, clses = [], []
    net.eval()
    for img, cls, _ in tqdm(dataloader):
        clses.append(cls)
        bs.append((net(img.to(device))).data.cpu())
    return torch.cat(bs).sign(), torch.cat(clses)

def compute_result_gldv2(dataloader, net, device):
    bs = torch.zeros(761757,48)
    clses = torch.zeros((761757,101302),dtype=torch.int8)
    net.eval()
    i=0
    index = 0
    for img, cls, _ in tqdm(dataloader):
        index += img.size()[0]
        clses[i*64:index,:] = cls
        bs[i*64:index,:] = (net(img.to(device))).data.cpu()
        i+=1
    return bs.sign(), clses

def compute_result_gldv2_only_feat(dataloader, net, config, flag, device):
    clses = []
    bs = torch.zeros(dataloader.dataset.__len__(),config['bit_list'][0])
    net.eval()
    i=0
    index = 0
    for img, cls, _ in tqdm(dataloader):
        index += img.size()[0]
        clses.append(cls)
        bs[i*config["batch_size"]:index,:] = (net(img.to(device))).data.cpu()
        i+=1
    return bs.sign(), torch.cat(clses)


def CalcHammingDist(B1, B2):
    q = B2.shape[1]
    distH = 0.5 * (q - np.dot(B1, B2.transpose()))
    return distH

def CalcTopMAP_ByIndex(rB, qB, retrievalL, queryL, topk):
    num_query = queryL.shape[0]
    topkmap = 0
    P = []
    for iter in tqdm(range(num_query)):
        # gnd = (np.dot(queryL[iter, :], retrievalL.transpose()) > 0).astype(np.float32)
        hamm = CalcHammingDist(qB[iter, :], rB)
        ind = np.argsort(hamm)
        gnd = retrievalL[ind]
        # gnd = gnd[ind]
        tgnd = gnd[0:topk]

        r_num = (tgnd==queryL[iter]).sum()
        P.append(r_num/topk)

    topkmap = sum(P)/len(P)
 
    return topkmap

def CalcTopMap(rB, qB, retrievalL, queryL, topk):
    num_query = queryL.shape[0]
    topkmap = 0
    for iter in tqdm(range(num_query)):
        gnd = (np.dot(queryL[iter, :], retrievalL.transpose()) > 0).astype(np.float32)
        hamm = CalcHammingDist(qB[iter, :], rB)
        ind = np.argsort(hamm)
        gnd = gnd[ind]

        tgnd = gnd[0:topk]
        tsum = np.sum(tgnd).astype(int)
        if tsum == 0:
            continue
        count = np.linspace(1, tsum, tsum)

        tindex = np.asarray(np.where(tgnd == 1)) + 1.0
        topkmap_ = np.mean(count / (tindex))
        topkmap = topkmap + topkmap_
    topkmap = topkmap / num_query
    return topkmap
