import os
import random
from argparse import ArgumentParser

from scipy import stats
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.optim import Adam, AdamW
from torch.utils.data import Dataset, DataLoader
import numpy as np

from models.QAmodel import GatedMoM
from datasets.DataSource import ODVSource, VRVQWSource, JVQDSource


class TrainSet(Dataset):
    def __init__(self, images, dmos, database):
        self.images = []
        self.dmos = []
        for j in range(len(images)):
            vname = images[j]
            for i in range(0, 5):
                if os.path.exists(os.path.join("./cache/features/%s/" % database, vname[:-4] + "_VP" + str(i) + ".npy")):
                    self.images.append(vname[:-4] + "_VP" + str(i) + ".npy")
                    self.dmos.append(dmos[j])

        self.database = database

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        vname = self.images[index]
        feat = np.load(os.path.join("./cache/features/%s/" % self.database, vname))
        refs = np.load(os.path.join("./cache/ref_features/%s/" % self.database, vname))
        # video = torch.permute(video,[0,3,1,2])
        dmos = self.dmos[index]
        sample = {"feat": feat, "refs": refs, "label": dmos}
        return sample


class TestSet(Dataset):
    def __init__(self, images, dmos, database):
        self.images = images
        self.dmos = dmos
        self.database = database

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        vname = self.images[index]
        feat = []
        refs = []
        for i in range(0, 5):
            if os.path.exists("./cache/features/%s/" % self.database + vname[:-4] + "_VP" + str(i) + ".npy"):
                feat.append(np.load("./cache/features/%s/" % self.database + vname[:-4] + "_VP" + str(i) + ".npy")[np.newaxis, :])
                refs.append(np.load("./cache/ref_features/%s/" % self.database + vname[:-4] + "_VP" + str(i) + ".npy")[np.newaxis, :])
        feat = np.concatenate(feat, axis=0)
        refs = np.concatenate(refs, axis=0)
        dmos = self.dmos[index]
        sample = {"feat": feat, "refs": refs, "label": dmos}
        return sample


def five_fold_eval(database: str):
    # parameters
    DATASources = {
        "VQA-ODV": ODVSource,
        "VRVQW": VRVQWSource,
        "JVQD": JVQDSource
    }

    Videosource = DATASources[database]
    device = torch.device("cuda")
    videosource = Videosource()  # generate tran-test splits
    metrics = np.zeros((5, 3))

    for r in range(0, 5):
        checkpoint = videosource.fiveFolds[r]  # 5-folds eval
        train_images = checkpoint["train_images"]
        train_dmos = checkpoint["train_dmos"]
        test_images = checkpoint["test_images"]
        test_dmos = checkpoint["test_dmos"]

        train_set = TrainSet(images=train_images, dmos=train_dmos, database=database)
        test_set = TestSet(images=test_images, dmos=test_dmos, database=database)
        dataloader = DataLoader(train_set, batch_size=16, shuffle=True, num_workers=6,
                                pin_memory=True, persistent_workers=True, prefetch_factor=3)
        testloader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=2,
                                pin_memory=True, persistent_workers=True)

        model = GatedMoM().to(device)
        optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        criterion = nn.MSELoss()  # LossGroupQP()
        sroccbest = 0

        for epoch in range(300):
            # ------train-------------
            model.train()

            for idx, data in enumerate(dataloader):
                features = data["feat"].float().to(device)
                refs = data["refs"].float().to(device)
                label = data["label"].float().to(device)
                pre, pre1, pre2, a = model(features, refs)
                l1 = criterion(pre1, label.to(device))
                l2 = criterion(pre2, label.to(device))
                loss = criterion(pre, label.to(device)) + l1 + l2 + 0.01 * criterion(a[:, :, 0], a[:, :, 1])
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            model.eval()
            pre = np.array([0])
            tar = np.array([0])
            pre1 = np.array([0])
            pre2 = np.array([0])
            with torch.inference_mode():
                for idx, data in enumerate(testloader):
                    features = data["feat"].float().to(device).squeeze(0)
                    refs = data["refs"].float().to(device).squeeze(0)
                    label = data["label"].view(-1).numpy()
                    pred, pred1, pred2, _ = model(features, refs)
                    pred = pred.mean().view(-1).to("cpu").numpy()
                    pred1 = pred1.mean().view(-1).to("cpu").numpy()
                    pred2 = pred2.mean().view(-1).to("cpu").numpy()
                    pre = np.hstack((pre, pred))
                    pre1 = np.hstack((pre1, pred1))
                    pre2 = np.hstack((pre2, pred2))
                    tar = np.hstack((tar, label))

                srocc1, _ = stats.spearmanr(pre[1:], tar[1:])
                plcc1, _ = stats.pearsonr(pre[1:], tar[1:])
                rmse1 = np.sqrt(np.mean(np.square(pre[1:] - tar[1:])))
                if srocc1 > sroccbest:
                    sroccbest = srocc1
                    metrics[r, 0] = srocc1
                    metrics[r, 0] = plcc1
                    metrics[r, 0] = rmse1

    return metrics.mean(axis=0)


if __name__ == "__main__":
    parser = ArgumentParser(description="Trian and Test")
    parser.add_argument("--database", type=str, help="Database name: VQA-ODV/VRVQW/JVQD")
    args = parser.parse_args()

    SEED = 42
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    metrics = five_fold_eval(args.database)
    print("SRCC  | PLCC  | RMSE ")
    print("%.4f | %.4f | %.4f" % (metrics[0], metrics[1], metrics[2]))
