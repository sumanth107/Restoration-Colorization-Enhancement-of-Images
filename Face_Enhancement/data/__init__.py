

import importlib
import torch.utils.data
from Face_Enhancement.data.base_dataset import BaseDataset
from Face_Enhancement.data.face_dataset import FaceTestDataset


def create_dataloader(opt):

    instance = FaceTestDataset()
    instance.initialize(opt)
    dataloader = torch.utils.data.DataLoader(
        instance,
        batch_size=opt.batchSize,
        shuffle=not opt.serial_batches,
        num_workers=int(opt.nThreads),
        drop_last=opt.isTrain,
    )
    return dataloader
