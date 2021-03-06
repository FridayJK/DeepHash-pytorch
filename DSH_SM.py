from utils.tools import *
from network import *

import math
import os
import torch
import torch.optim as optim
import time
import numpy as np
from tqdm import tqdm

# os.environ['CUDA_VISIBLE_DEVICES'] = '3'

torch.multiprocessing.set_sharing_strategy('file_system')

# DSH(CVPR2016)
# paper [Deep Supervised Hashing for Fast Image Retrieval](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Liu_Deep_Supervised_Hashing_CVPR_2016_paper.pdf)
# code [DSH-pytorch](https://github.com/weixu000/DSH-pytorch)

def set_learning_rate(optimizer, epoch, iter_size, iter_num):
    current_iter = epoch * iter_size + iter_num
    if current_iter < 1000:
        current_lr = 1e-5 * math.pow(current_iter / 1000, 4)
    else:
        current_lr = 1e-5 * (1 + math.cos(epoch * math.pi / 100)) / 2
    for param_group in optimizer.param_groups:
        param_group['lr'] = current_lr
    return current_lr

def get_config():
    config = {
        "alpha": 0.1,
        # "optimizer":{"type":  optim.SGD, "optim_params": {"lr": 0.05, "weight_decay": 10 ** -5}},
        "optimizer": {"type": optim.RMSprop, "optim_params": {"lr": 1e-5, "weight_decay": 10 ** -5}},
        "info": "[DSH]",
        # "resize_size": 256,
        # "crop_size": 224,
        "resize_size": 32,
        "crop_size": 224,
        "batch_size": 128,
        "net": HashEmbNet_Scratch,
        # "net":ResNet,
        "dataset": "GLDv2-0",
        # "dataset": "cifar10",
        # "dataset": "cifar10-1",
        # "dataset": "cifar10-2",
        # "dataset": "coco",
        # "dataset": "mirflickr",
        # "dataset": "voc2012",
        # "dataset": "imagenet",
        # "dataset": "nuswide_21",
        # "dataset": "nuswide_21_m",
        # "dataset": "nuswide_81_m",
        "epoch": 100,
        "test_map": 1,
        "save_path": "save/DSH",
        # "device":torch.device("cpu"),
        "device": torch.device("cuda:3"),
        "bit_list": [128],
    }
    config = config_dataset(config)
    return config



class DSHLoss(torch.nn.Module):
    def __init__(self, config, bit):
        super(DSHLoss, self).__init__()
        self.m = 2 * bit
        self.U = torch.zeros(config["num_train"], bit).float().to(config["device"])
        # self.Y = torch.zeros(config["num_train"], config["n_class"]).float()
        # self.U = torch.zeros(config["num_train"], bit).float().to(config["device"])
        self.Y = torch.zeros(config["num_train"], config["n_class"]).float().to(config["device"])


    def forward(self, u, y, ind, config):
        self.U[ind, :] = u.data
        self.Y[ind, :] = y.float()

        dist = (u.unsqueeze(1) - self.U.unsqueeze(0)).pow(2).sum(dim=2)
        y = (y @ self.Y.t() == 0).float()

        loss = (1 - y) / 2 * dist + y / 2 * (self.m - dist).clamp(min=0)
        loss1 = loss.mean()
        loss2 = config["alpha"] * (1 - u.sign()).abs().mean()

        return loss1 + loss2

class DSHLoss_CPU(torch.nn.Module):
    def __init__(self, config, bit):
        super(DSHLoss_CPU, self).__init__()
        self.m = 2 * bit
        self.U = torch.zeros(config["num_train"], bit).float().to(config["device"])
        # self.Y = torch.zeros(config["num_train"], config["n_class"]).float()
        # self.U = torch.zeros(config["num_train"], bit).float().to(config["device"])
        self.Y = torch.zeros(config["num_train"], config["n_class"]).float()


    def forward(self, u, y, ind, config):
        self.U[ind, :] = u.data
        self.Y[ind, :] = y.float()

        dist = (u.unsqueeze(1) - self.U.unsqueeze(0)).pow(2).sum(dim=2)
        # dist = dist.cpu()
        y = (y @ self.Y.t() == 0).float()
        y = y.to(config["device"])
        loss = (1 - y) / 2 * dist + y / 2 * (self.m - dist).clamp(min=0)
        loss1 = loss.mean()
        loss2 = config["alpha"] * (1 - u.sign()).abs().mean()

        return loss1 + loss2

class DSHLoss_PartSample(torch.nn.Module):
    def __init__(self, config, bit):
        super(DSHLoss_PartSample, self).__init__()
        self.m = 2 * bit
        self.U = torch.zeros(config["num_train"], bit).float().to(config["device"])
        # self.Y = torch.zeros(config["num_train"], config["n_class"]).float()
        # self.U = torch.zeros(config["num_train"], bit).float().to(config["device"])
        self.Y = torch.zeros(config["num_train"]).float()
        # self.Y = -1.0


    def forward(self, u, y, ind, config):
        self.U[ind, :] = u.data
        self.Y[ind] = y.float()

        max_SampleNum = 30

        #make part sample
        # randm_indx = torch.randperm(self.Y.shape[0])
        randm_indx = torch.randperm(max_SampleNum*y.shape[0])
        step = 0
        for i , indx in enumerate(y):
            tmp_idx = torch.where(self.Y==indx)
            if(tmp_idx[0].shape[0]>max_SampleNum):
                randm_indx[step:(step+max_SampleNum)] = tmp_idx[0][0:max_SampleNum]
                step += max_SampleNum
            else:
                randm_indx[step:(step+tmp_idx[0].shape[0])] = tmp_idx[0]
                step += tmp_idx[0].shape[0]
        #new smple pool
        Y_POOL = self.Y[randm_indx[0:step]]
        U_POLL = self.U[randm_indx[0:step], :]
            


        dist = (u.unsqueeze(1) - U_POLL.unsqueeze(0)).pow(2).sum(dim=2)
        # dist = dist.cpu()
        y = (torch.repeat_interleave(y,step).reshape(y.shape[0],step) - Y_POOL.repeat(y.shape[0]).reshape(y.shape[0],step))
        # y = (y.t() - Y_POOL.repeat(y.shape[0]).reshape(y.shape[0],step))
        y = (y!=0).float()

        # y = (y @ Y_POOL.t() == 0).float()
        y = y.to(config["device"])
        loss = (1 - y) / 2 * dist + y / 2 * (self.m - dist).clamp(min=0)
        loss1 = loss.mean()
        loss2 = config["alpha"] * (u.abs() - 1).abs().mean()
        # loss2 = config["alpha"] * (1 - u.sign()).abs().mean()

        return loss1 + loss2

def train_val(config, bit):

    device = config["device"]
    train_loader, test_loader, dataset_loader, num_train, num_test, num_dataset = get_data(config)
    config["num_train"] = num_train
    config["num_test"] = num_test
    config["num_dataset"] = num_dataset

    net = config["net"](bit).to(device)
    if(os.path.exists("save/DSH/GLDv2-0-val0.197-model.pt")):
        net.load_state_dict(torch.load("save/DSH/GLDv2-0-val0.197-model.pt"))
        print("train from","save/DSH/GLDv2-0-val0.197-model.pt")

    optimizer = config["optimizer"]["type"](net.parameters(), **(config["optimizer"]["optim_params"]))

    criterion = DSHLoss_PartSample(config, bit)

    Best_mAP = 0

    for epoch in range(config["epoch"]):

        current_time = time.strftime('%H:%M:%S', time.localtime(time.time()))

        print("%s[%2d/%2d][%s] bit:%d, dataset:%s, training...." % (
            config["info"], epoch + 1, config["epoch"], current_time, bit, config["dataset"]), end="")

        net.train()

        train_loss = 0
        i=0
        for image, label, ind in tqdm(train_loader):
            # set_learning_rate(optimizer, epoch, len(train_loader), i)
            image = image.to(device)
            # label = label.to(device)

            optimizer.zero_grad()
            u = net(image)

            loss = criterion(u, label.float(), ind, config)
            train_loss += loss.item()

            loss.backward()
            optimizer.step()
            i+=1
            # if(i%100 == 0):
            #     print("\b\b\b\b\b\b\b ", len(train_loader), i)

        train_loss = train_loss / len(train_loader)

        print("\b\b\b\b\b\b\b loss:%.3f" % (train_loss))

        if (epoch + 1) % config["test_map"] == 0:

            val_binary, val_label = compute_result_gldv2_only_feat(train_loader, net, config, 1, device=device)
            val_map = CalcTopMAP_ByIndex(val_binary.numpy()[1000:,:], val_binary.numpy()[0:1000,:], val_label.numpy()[1000:], val_label.numpy()[0:1000], config["topK"])
            print("%s epoch:%d, bit:%d, dataset:%s,val_MAP:%.3f" % (
                config["info"], epoch + 1, bit, config["dataset"], val_map))

            tst_binary, tst_label = compute_result_gldv2_only_feat(test_loader, net, config, 0, device=device)
            trn_binary, trn_label = compute_result_gldv2_only_feat(dataset_loader, net, config, 1, device=device)
            mAP = CalcTopMAP_ByIndex(trn_binary.numpy(), tst_binary.numpy(), trn_label.numpy(), tst_label.numpy(),
                             config["topK"])

            if mAP > Best_mAP:
                Best_mAP = mAP

                if "save_path" in config:
                    if not os.path.exists(config["save_path"]):
                        os.makedirs(config["save_path"])
                    print("save in ", config["save_path"])
                    np.save(os.path.join(config["save_path"], config["dataset"] + str(mAP) + "-" + "trn_binary.npy"),
                            trn_binary.numpy())
                    torch.save(net.state_dict(),
                               os.path.join(config["save_path"], config["dataset"] + "-" + str(mAP) + "-model.pt"))
            print("%s epoch:%d, bit:%d, dataset:%s, MAP:%.3f, Best MAP: %.3f" % (
                config["info"], epoch + 1, bit, config["dataset"], mAP, Best_mAP))
            print(config)


if __name__ == "__main__":
    config = get_config()
    print(config)
    for bit in config["bit_list"]:
        train_val(config, bit)
