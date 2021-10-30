import cv2
import os
import argparse
import glob

from torch.autograd import Variable

from MRFNETcolor.model import MRFNET
import matplotlib.pyplot as plt
from MRFNETcolor.utils import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description="MRFNet_Test")
parser.add_argument("--logdir", type=str, default="logs50", help='path of log files')
parser.add_argument("--test_data", type=str, default='CBSD68', help='test on Set12 or Set68')
parser.add_argument("--test_noiseL", type=float, default=50, help='noise level used on test set')
parser.add_argument("--half", type=bool, default=False, help='True to use flost16')
opt = parser.parse_args()


def normalize(data):
    return data / 255.


def display_batch_image(Out):
    Img = Out.data.cpu().numpy().astype(np.float32)
    for i in range(Img.shape[0]):
        display_data = np.reshape(Img[i], (Img.shape[2], Img.shape[3], Img.shape[1]))
        plt.imshow(display_data, interpolation='nearest')
        plt.show()


def main():
    # Build model
    print('Loading model ...\n')
    net = MRFNET(channels=3)
    device_ids = [0]
    model = nn.DataParallel(net, device_ids=device_ids).cuda()
    if opt.half:
        model = model.half()
    model.load_state_dict(torch.load(os.path.join(opt.logdir, 'model.pth')))

    model.eval()
    # summary(model, (3, 256, 256), batch_size=8, device='cuda')

    # load data info
    print('Loading data info ...\n')
    files_source = glob.glob(os.path.join('./data', opt.test_data, '*'))
    files_source.sort()
    # process data
    psnr_test, ssim_test = 0, 0
    for f in files_source:
        # image
        Img = cv2.imread(f)
        Img = torch.tensor(Img)
        # print Img.shape
        Img = Img.permute(2, 0, 1)
        Img = Img.numpy()
        a1, a2, a3 = Img.shape
        Img = np.tile(Img, (3, 1, 1, 1))  # expand the dimensional
        Img = np.float32(normalize(Img))
        ISource = torch.Tensor(Img)
        # noise
        torch.manual_seed(12)
        noise = torch.FloatTensor(ISource.size()).normal_(mean=0, std=opt.test_noiseL / 255.)
        # noisy image
        INoisy = ISource + noise
        ISource = Variable(ISource)
        INoisy = Variable(INoisy)
        ISource = ISource.cuda()
        INoisy = INoisy.cuda()
        if opt.half:
            INoisy = INoisy.half()
        with torch.no_grad():  # this can save much memory

            start_time = time_synchronized()
            Out = model(INoisy)
            time_now = time_synchronized()
            eliptime = time_now - start_time
            Out = torch.clamp(Out, 0., 1.)

        psnr = batch_PSNR(Out, ISource, 1.)
        ssim = batch_ssim(Out.reshape((Out.shape[0], Out.shape[2],
                                       Out.shape[3], Out.shape[1])),
                          ISource.reshape((ISource.shape[0], ISource.shape[2],
                                           ISource.shape[3], ISource.shape[1])))

        saved_img = Out.cpu()
        saved_img = torch.mean(saved_img, 0)
        saved_img = saved_img.permute(1, 2, 0)
        saved_img = np.uint8(np.rint(saved_img * 255))

        if not os.path.exists('./results/'):
            os.mkdir('./results/')
        if not os.path.exists('./results/' + opt.test_data):
            os.mkdir('./results/' + opt.test_data)

        cv2.imwrite('./results/' + opt.test_data + '/' + f[len(f) - 6:len(f)], saved_img)

        psnr_test += psnr
        ssim_test += ssim
        print("%s PSNR %f ssim %f time %2.5f" % (f, psnr, ssim, eliptime))

    psnr_test /= len(files_source)
    print("\nPSNR on test data %f" % psnr_test)

    ssim_test /= len(files_source)
    print("\nssim on test data %f" % ssim_test)


if __name__ == "__main__":
    main()
