import time

import cv2
import os
import argparse
import glob
from MRFNETgray.model import MRFNET
# import numpy as np
import torch
# import torch.nn as nn
from torchsummary import summary
# from torchviz import make_dot
from torch.autograd import Variable

from MRFNETgray.utils import *

from skimage.io import imsave

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description="MRFNET_Test")
parser.add_argument("--logdir", type=str, default="logs25", help='path of log files')
parser.add_argument("--test_data", type=str, default='Set12', help='test on Set12 or Set68')
parser.add_argument("--test_model", type=str, default='model.pth', help='test on model_xxx')
parser.add_argument("--test_noiseL", type=float, default=25, help='noise level used on test set')
opt = parser.parse_args()

# s25 s12 30.573952
# s25 model_39 set12 30.656465
# s25 model_58 set12 30.657936
# s25 model_58 set12 30.665434

# s25 model_58 set68 29.303171 adnet(29.25) dncnn(29.23)
# s25 model_90 set68 29.310399


def time_synchronized():
    # pytorch-accurate time
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


def normalize(data):
    return data / 255.


def main():
    # Build model
    print('Loading model ...\n')
    net = MRFNET(channels=1)
    device_ids = [0]
    model = nn.DataParallel(net, device_ids=device_ids).cuda()
    model.load_state_dict(torch.load(os.path.join(opt.logdir, opt.test_model)))
    model.eval()
    # load data info
    print('Loading data info ...\n')
    files_source = glob.glob(os.path.join('./data', opt.test_data, '*.png'))
    files_source.sort()
    # process data
    psnr_test = 0
    ssim_test = 0

    psnr_me = []
    ssim_me = []
    for f in files_source:
        # image
        Img = cv2.imread(f)
        Img = normalize(np.float32(Img[:, :, 0]))
        Img = np.expand_dims(Img, 0)
        Img = np.expand_dims(Img, 1)
        ISource = torch.Tensor(Img)
        # noise
        torch.manual_seed(0)  # set the seed
        noise = torch.FloatTensor(ISource.size()).normal_(mean=0, std=opt.test_noiseL / 255.)
        # noisy image
        INoisy = ISource + noise
        ISource = Variable(ISource)
        INoisy = Variable(INoisy)
        ISource = ISource.cuda()
        INoisy = INoisy.cuda()
        with torch.no_grad():  # this can save much memory
            summary(model, (1, 225, 225), batch_size=-1, device='cuda')
            # make_dot(model(torch.rand(1, 1, 32, 32).cuda()), params=dict(model.named_parameters()))
            # print(INoisy.shape)
            # Out = torch.clamp(model(INoisy), 0., 1.)
            start_time = time_synchronized()
            Out = model(INoisy)
            time_now = time_synchronized()
            eliptime = time_now - start_time

        saved_img = Out.cpu()
        saved_img = torch.reshape(saved_img, (saved_img.shape[2], saved_img.shape[3]))
        saved_img = np.uint8(np.rint(saved_img * 255))

        if not os.path.exists('./results/'):
            os.mkdir('./results/')
        if not os.path.exists('./results/' + opt.test_data):
            os.mkdir('./results/' + opt.test_data)

        imsave('./results/' + opt.test_data + '/' + f[len(f) - 6:len(f)], saved_img)

        psnr = batch_PSNR(Out, ISource, 1.)
        ssim = batch_ssim(Out.reshape((Out.shape[0], Out.shape[2],
                                      Out.shape[3])),
                          ISource.reshape((ISource.shape[0], ISource.shape[2],
                                          ISource.shape[3])))

        psnr_test += psnr
        ssim_test += ssim
        print("%s PSNR %f ssim %f time %2.5f" % (f, psnr, ssim, eliptime))
        # print("%s ssim %f" % (f, ssim))

        psnr_me.append(psnr)
        ssim_me.append(ssim)

    psnr_test /= len(files_source)
    print("\nPSNR on test data %f" % psnr_test)

    ssim_test /= len(files_source)
    print("\nssim on test data %f" % ssim_test)

    # write ssim and psnr to text
    infoi = np.hstack((psnr_me, ssim_me, psnr_test, ssim_test))
    with open('./results/' + opt.test_data + '/results_b.txt', "w+") as f:
        f.write(str(infoi))

    return 0


if __name__ == "__main__":
    main()
