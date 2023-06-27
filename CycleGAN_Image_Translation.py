import argparse
import itertools
from tqdm import tqdm
import os
from torch.autograd import Variable
import time
import shutil
import torch
import torchvision.transforms as transforms
import numpy as np
import torchvision
import random
from PIL import Image
import torch.distributed
import matplotlib
import matplotlib.pyplot as plt
import torch.utils.data


matplotlib.use('TkAgg')
plt.rcParams["figure.figsize"] = (7, 15)
plt.style.use('dark_background')


parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('-run_path', default='100', type=str)

parser.add_argument('-num_epochs', default=1000, type=int)
parser.add_argument('-batch_size', default=32, type=int)
parser.add_argument('-samples_per_class', default=1000, type=int)

parser.add_argument('-learning_rate_g', default=3e-4, type=float)
parser.add_argument('-learning_rate_d', default=1e-4, type=float)

parser.add_argument('-z_size', default=128, type=int)

parser.add_argument('-coef_c', default=10, type=float)
parser.add_argument('-coef_i', default=5.0, type=float)

parser.add_argument('-is_debug', default=False, type=lambda x: (str(x).lower() == 'true'))

args, _ = parser.parse_known_args()

RUN_PATH = args.run_path
BATCH_SIZE = args.batch_size
EPOCHS = args.num_epochs
Z_SIZE = args.z_size
DEVICE = 'cuda'
MAX_LEN = args.samples_per_class
IS_DEBUG = args.is_debug
INPUT_SIZE = 64

if not torch.cuda.is_available() or IS_DEBUG:
    IS_DEBUG = True
    MAX_LEN = 300 # per class for debugging
    DEVICE = 'cpu'
    BATCH_SIZE = 16

if len(RUN_PATH):
    RUN_PATH = f'{int(time.time())}_{RUN_PATH}'
    if os.path.exists(RUN_PATH):
        shutil.rmtree(RUN_PATH)
    os.makedirs(RUN_PATH)

print(DEVICE)

class DatasetHorsesZebras(torch.utils.data.Dataset):
    def __init__(self, unaligned=True, is_train=True):
        super().__init__()
        self.unaligned = unaligned

        data_root = '../data'
        train_dir = os.path.join(data_root, 'trainA' if is_train else 'testA')
        test_dir = os.path.join(data_root, 'trainB' if is_train else 'testB')

        self.dataA = self.load_images_from_folder(train_dir)
        self.dataB = self.load_images_from_folder(test_dir)

        self.transform = transforms.Compose([
            transforms.Resize(INPUT_SIZE),
            transforms.ToTensor(),
        ])

    def __len__(self):
        if IS_DEBUG:
            return MAX_LEN
        return max(len(self.dataA), len(self.dataB))

    def __getitem__(self, idx):
        imageA = self.dataA[idx % len(self.dataA)]
        if self.unaligned:
            imageB = self.dataB[random.randint(0, len(self.dataB) - 1)]
        else:
            imageB = self.dataB[idx % len(self.dataB)]

        imageA = self.transform(imageA)
        imageB = self.transform(imageB)
        return imageA, imageB

    def load_images_from_folder(self, folder):
        images = []
        for filename in os.listdir(folder):
            img = Image.open(os.path.join(folder, filename))
            if img is not None:
                images.append(img)
        return images


dataset_train = DatasetHorsesZebras()
dataset_test = DatasetHorsesZebras(is_train=False)

print(f'dataset_train: {len(dataset_train)} dataset_test: {len(dataset_test)}')

data_loader_train = torch.utils.data.DataLoader(
    dataset=dataset_train,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True,
    num_workers=0
)


class ModelD(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.output_shape = (1, INPUT_SIZE // 2 ** 4, INPUT_SIZE // 2 ** 4)

        def discriminator_block(in_filters, out_filters, normalize=True):
            layers = [torch.nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(torch.nn.InstanceNorm2d(out_filters))
            layers.append(torch.nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = torch.nn.Sequential(
            *discriminator_block(3, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            torch.nn.ZeroPad2d((1, 0, 1, 0)),
            torch.nn.Conv2d(512, 1, 4, padding=1)
        )

    def forward(self, img):
        return self.model(img)


class ResidualBlock(torch.nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        self.block = torch.nn.Sequential(
            torch.nn.ReflectionPad2d(1),
            torch.nn.Conv2d(in_features, in_features, 3),
            torch.nn.InstanceNorm2d(in_features),
            torch.nn.ReLU(inplace=True),
            torch.nn.ReflectionPad2d(1),
            torch.nn.Conv2d(in_features, in_features, 3),
            torch.nn.InstanceNorm2d(in_features),
        )

    def forward(self, x):
        return x + self.block(x)


class ModelG(torch.nn.Module):
    def __init__(self):
        super().__init__()

        out_features = 64
        num_residual_blocks = 6
        model = [
            torch.nn.ReflectionPad2d(3),
            torch.nn.Conv2d(3, out_features, 7),
            torch.nn.InstanceNorm2d(out_features),
            torch.nn.ReLU(inplace=True),
        ]
        in_features = out_features

        for _ in range(2):
            out_features *= 2
            model += [
                torch.nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                torch.nn.InstanceNorm2d(out_features),
                torch.nn.ReLU(inplace=True),
            ]
            in_features = out_features

        for _ in range(num_residual_blocks):
            model += [ResidualBlock(out_features)]

        for _ in range(2):
            out_features //= 2
            model += [
                torch.nn.Upsample(scale_factor=2),
                torch.nn.Conv2d(in_features, out_features, 3, stride=1, padding=1),
                torch.nn.InstanceNorm2d(out_features),
                torch.nn.ReLU(inplace=True),
            ]
            in_features = out_features

        model += [torch.nn.ReflectionPad2d(3), torch.nn.Conv2d(out_features, 3, 7), torch.nn.Sigmoid()]

        self.model = torch.nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        if m.weight is not None:
            torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, "bias") and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find("InstanceNorm2d") != -1:
        if m.weight is not None:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)




def get_param_count(model):
    params = list(model.parameters())
    result = 0
    for param in params:
        count_param = np.prod(param.size()) # size[0] * size[1] ...
        result += count_param
    return result


model_D_A = ModelD().to(DEVICE)
model_D_B = ModelD().to(DEVICE)
model_G_AB = ModelG().to(DEVICE)
model_G_BA = ModelG().to(DEVICE)

model_D_A.apply(weights_init_normal)
model_D_B.apply(weights_init_normal)
model_G_AB.apply(weights_init_normal)
model_G_BA.apply(weights_init_normal)

params_D = itertools.chain(model_D_A.parameters(), model_D_B.parameters())
params_G = itertools.chain(model_G_AB.parameters(), model_G_BA.parameters())

optimizer_D = torch.optim.Adam(params_D, lr=args.learning_rate_d, betas=(0.5, 0.999))
optimizer_G = torch.optim.Adam(params_G, lr=args.learning_rate_g, betas=(0.5, 0.999))

metrics = {}
for stage in ['train']:
    for metric in [
        'loss_gan',
        'loss_identity',
        'loss_d',
        'loss_cycle'
    ]:
        metrics[f'{stage}_{metric}'] = []

for epoch in range(1, EPOCHS):
    metrics_epoch = {key: [] for key in metrics.keys()}

    stage = 'train'
    for real_A, real_B in tqdm(data_loader_train, desc=stage):

        real_A = real_A.to(DEVICE)
        real_B = real_B.to(DEVICE)

        valid = Variable(torch.Tensor(np.ones((real_A.size(0), *model_D_A.output_shape))), requires_grad=False).to(DEVICE)

        model_G_AB.train()
        model_G_BA.train()
        optimizer_G.zero_grad()

        loss_i_A = torch.mean(torch.abs(model_G_BA.forward(real_A) - real_A))
        loss_i_B = torch.mean(torch.abs(model_G_AB.forward(real_B) - real_B))
        loss_identity = (loss_i_A + loss_i_B) / 2

        fake_B = model_G_AB.forward(real_A)
        fake_A = model_G_BA.forward(real_B)
        loss_GAN_AB = torch.mean((model_D_B.forward(fake_B) - valid) ** 2)
        loss_GAN_BA = torch.mean((model_D_A.forward(fake_A) - valid) ** 2)
        loss_GAN = (loss_GAN_AB + loss_GAN_BA) / 2

        recover_A = model_G_BA.forward(fake_B)
        recover_B = model_G_AB.forward(fake_A)
        loss_cycle_A = torch.mean(torch.abs(recover_A - real_A))
        loss_cycle_B = torch.mean(torch.abs(recover_B - real_B))
        loss_cycle = (loss_cycle_A + loss_cycle_B) / 2

        loss = loss_GAN + loss_identity * args.coef_i + loss_cycle * args.coef_c
        loss.backward()
        optimizer_G.step()

        metrics_epoch[f'{stage}_loss_gan'].append(loss_GAN.cpu().item())
        metrics_epoch[f'{stage}_loss_identity'].append(loss_identity.cpu().item())
        metrics_epoch[f'{stage}_loss_cycle'].append(loss_cycle.cpu().item())

        optimizer_D.zero_grad()
        optimizer_G.zero_grad()
        model_D_A.train()
        model_D_B.train()

        loss_A_real = torch.mean((model_D_A.forward(real_A) - valid) ** 2)
        loss_A_fake = torch.mean(model_D_A.forward(real_A) ** 2)
        loss_B_real = torch.mean((model_D_B.forward(real_B) - valid) ** 2)
        loss_B_fake = torch.mean(model_D_B.forward(real_B) ** 2)
        loss_d = (loss_A_real + loss_A_fake + loss_B_real + loss_B_fake) / 4

        metrics_epoch[f'{stage}_loss_d'].append(loss_d.cpu().item())

    metrics_strs = []
    for key in metrics_epoch.keys():
        value = 0
        if len(metrics_epoch[key]):
            value = np.mean(metrics_epoch[key])
        metrics[key].append(value)
        metrics_strs.append(f'{key}: {round(value, 2)}')

    print(f'epoch: {epoch} {" ".join(metrics_strs)}')

    plt.figure()
    plt.subplot(7, 1, 1)  # row col idx
    plts = []
    c = 0
    for key, value in metrics.items():
        plts += plt.plot(value, f'C{c}', label=key)
        ax = plt.twinx()
        c += 1
    plt.legend(plts, [it.get_label() for it in plts])

    plt.subplot(7, 1, 2)  # row col idx
    plt.title('Real winter scenes')
    grid_img = torchvision.utils.make_grid(
        real_B[:8].detach().cpu(),
        padding=5,
        scale_each=True,
        nrow=8
    )
    plt.imshow(grid_img.permute(1, 2, 0).data.numpy())

    plt.subplot(7, 1, 3)  # row col idx
    plt.title('Transformed summer scenes')
    grid_img = torchvision.utils.make_grid(
        fake_B[:8].detach().cpu(),
        padding=5,
        scale_each=True,
        nrow=8
    )
    plt.imshow(grid_img.permute(1, 2, 0).data.numpy())

    plt.subplot(7, 1, 4)  # row col idx
    plt.title('Recovered winter scenes')
    grid_img = torchvision.utils.make_grid(
        recover_B[:8].detach().cpu(),
        padding=5,
        scale_each=True,
        nrow=8
    )
    plt.imshow(grid_img.permute(1, 2, 0).data.numpy())

    plt.subplot(7, 1, 5)  # row col idx
    plt.title('Real summer scenes')
    grid_img = torchvision.utils.make_grid(
        real_A[:8].detach().cpu(),
        padding=5,
        scale_each=True,
        nrow=8
    )
    plt.imshow(grid_img.permute(1, 2, 0).data.numpy())

    plt.subplot(7, 1, 6)  # row col idx
    plt.title('Transformed winter scenes')
    grid_img = torchvision.utils.make_grid(
        fake_A[:8].detach().cpu(),
        padding=5,
        scale_each=True,
        nrow=8
    )
    plt.imshow(grid_img.permute(1, 2, 0).data.numpy())

    plt.subplot(7, 1, 7)  # row col idx
    plt.title('Recovered summer scenes')
    grid_img = torchvision.utils.make_grid(
        recover_A[:8].detach().cpu(),
        padding=5,
        scale_each=True,
        nrow=8
    )
    plt.imshow(grid_img.permute(1, 2, 0).data.numpy())

    plt.tight_layout(pad=0.5)

    if len(RUN_PATH) == 0:
        plt.show()
    else:
        if np.isnan(metrics[f'train_loss_gan'][-1]) or np.isnan(metrics[f'train_loss_d'][-1]):
            exit()
        plt.savefig(f'{RUN_PATH}/plt-{epoch}.png')