import os
import argparse
import torch.optim as optim

import time
from torch.autograd import Variable
from torch.utils.data import DataLoader

from MRFNETcolor.model import MRFNET
from MRFNETcolor.dataset import prepare_data, Dataset
from MRFNETcolor.utils import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description="ADNet")
parser.add_argument("--preprocess", type=bool, default=True, help='run prepare_data or not')
parser.add_argument("--resume", type=bool, default=False, help='resume from a ckpt or not')
parser.add_argument("--batchSize", type=int, default=512, help="Training batch size")
parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
parser.add_argument("--milestone", type=int, default=30, help="When to decay learning rate; should be less than epochs")
parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
parser.add_argument("--outf", type=str, default="logs", help='path of log files')
parser.add_argument("--mode", type=str, default="S", help='with known noise level (S) or blind training (B)')
parser.add_argument("--noiseL", type=float, default=25, help='noise level; ignored when mode=B')
parser.add_argument("--val_noiseL", type=float, default=25, help='noise level used on validation set')
parser.add_argument("--half", type=bool, default=True, help='resume from a ckpt or not')
opt = parser.parse_args()
use_amp = opt.half
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)


def main():
    # Load dataset
    save_dir = opt.outf + 'sigma' + str(opt.noiseL) + '_' + time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()) + '/'

    if not os.path.exists(save_dir) and opt.resume is False:
        os.mkdir(save_dir)

    # Load dataset
    print('Loading dataset ...\n')
    dataset_train = Dataset(train=True)
    # dataset_val = Dataset(train=False)
    loader_train = DataLoader(dataset=dataset_train, num_workers=4, batch_size=opt.batchSize, shuffle=True)
    print("# of training samples: %d\n" % int(len(dataset_train)))
    # Build model
    net = MRFNET(channels=3)

    criterion = nn.MSELoss(reduction='sum')

    # Move to GPU
    device_ids = [0]
    model = nn.DataParallel(net, device_ids=device_ids).cuda()

    if opt.resume is True:
        resume_path_dir = 'logssigma50_2021-03-14-16-37-02/model_21.pth'
        save_dir = 'logssigma50_2021-03-14-16-37-02/'
        model.load_state_dict(torch.load(resume_path_dir))
        epoch_s = 21
    else:
        epoch_s = 0

    criterion.cuda()
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    noiseL_B = [0, 55]  # ingnored when opt.mode=='S'
    # psnr_list = []

    for epoch in range(epoch_s, opt.epochs):

        if epoch <= opt.milestone:
            current_lr = opt.lr
        elif 30 < epoch <= 60:
            current_lr = opt.lr / 10.
        elif 60 < epoch <= 90:
            current_lr = opt.lr / 100.
        elif 90 < epoch <= 120:
            current_lr = opt.lr / 1000.
        else:
            current_lr = opt.lr / 10000.
        # set learning rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr
        print('learning rate %f' % current_lr)

        # train
        start_time = time.time()
        loss_sum = 0
        for i, data in enumerate(loader_train, 0):
            # training step
            model.train()
            img_train = data
            if opt.mode == 'S':
                noise = torch.FloatTensor(img_train.size()).normal_(mean=0, std=opt.noiseL / 255.)
            elif opt.mode == 'B':
                noise = torch.zeros(img_train.size())
                stdN = np.random.uniform(noiseL_B[0], noiseL_B[1], size=noise.size()[0])
                for n in range(noise.size()[0]):
                    sizeN = noise[0, :, :, :].size()
                    noise[n, :, :, :] = torch.FloatTensor(sizeN).normal_(mean=0, std=stdN[n] / 255.)
            else:
                noise = 0  # only to avoid warning

            imgn_train = img_train + noise
            img_train, imgn_train = Variable(img_train.cuda()), Variable(imgn_train.cuda())
            # noise = Variable(noise.cuda())
            if opt.half:
                with torch.cuda.amp.autocast(enabled=use_amp):
                    out_train = model(imgn_train)
                    loss = criterion(out_train, img_train) / (imgn_train.size()[0] * 2)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                # optimizer.step()
                loss_sum += loss.item()
                optimizer.zero_grad()
            else:
                out_train = model(imgn_train)
                loss = criterion(out_train, img_train) / (imgn_train.size()[0] * 2)
                loss_sum += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if i % 100 == 0:
                elapse_time = time.time() - start_time
                start_time = time.time()
                print('epoch:{},batch:{}/{},loss:{}, time:{}'.format(epoch + 1, i + 1, len(loader_train), loss.item(),
                                                                     elapse_time))
                print('epoch:' + str(epoch + 1) + ',current average loss:' + str(loss_sum / (i + 1)))

        info = 'epoch:' + str(epoch + 1) + ',loss:' + str(loss_sum / len(loader_train)) + ' \n'
        print(info)
        with open('./epoch_loss' + '.txt', "a+") as f:
            f.write(info)

        model_name = 'model' + '_' + str(epoch + 1) + '.pth'
        torch.save(model.state_dict(), os.path.join(save_dir, model_name))


if __name__ == "__main__":
    if opt.preprocess:
        if opt.mode == 'S':
            prepare_data(data_path='../path', patch_size=50, stride=40, aug_times=1)
        if opt.mode == 'B':
            prepare_data(data_path='../path', patch_size=50, stride=10, aug_times=2)
    main()
