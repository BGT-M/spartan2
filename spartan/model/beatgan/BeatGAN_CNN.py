
import torch
import os
import numpy as np
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from .metric import evaluate
from .preprocess import preprocess_data

from .._model import MLmodel
from . import param_default

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

param_default_dict = {
    'network': 'CNN',
    'net_type': 'gru',
    'layers': 1,
    'seq_len': 256,
    'input_size': 1,
    'hidden_size': 100,
    'rep_size': 20,
    'batch_size': 64,
    'max_epoch': 5,
    'lr': 0.01,
    'lambda': 1,
    'out_path': './output'
}


class Generator(nn.Module):
    def __init__(self, nc=25, nz=100, seq_len=128, device=None):
        super(Generator, self).__init__()
        self.device = device
        self.nc = nc
        self.nz = nz
        self.ndf = 32

        encoder_modules = []
        encoder_modules.append(nn.Conv1d(self.nc, self.ndf, 4, 2, 1, bias=False))
        encoder_modules.append(nn.LeakyReLU(0.2, inplace=True))

        cur_len = seq_len//2
        i = 1
        while cur_len > 5:
            encoder_modules.append(nn.Conv1d(self.ndf*i, self.ndf * i*2, 4, 2, 1, bias=False))
            encoder_modules.append(nn.BatchNorm1d(self.ndf * i*2))
            encoder_modules.append(nn.LeakyReLU(0.2, inplace=True))
            i = i*2
            cur_len = cur_len//2

        encoder_modules.append(nn.Conv1d(self.ndf * i, self.nz, cur_len, 1, 0, bias=False))

        self.encoder = nn.Sequential(*encoder_modules)

        decoder_modules = []
        decoder_modules.append(nn.ConvTranspose1d(self.nz, self.ndf * i, 4, 1, 0, bias=False))
        decoder_modules.append(nn.BatchNorm1d(self.ndf * i))
        decoder_modules.append(nn.ReLU(True))
        i = i//2

        while i >= 1:
            decoder_modules.append(nn.ConvTranspose1d(self.ndf*(i*2), self.ndf * i, 4, 2, 1, bias=False))
            decoder_modules.append(nn.BatchNorm1d(self.ndf * i))
            decoder_modules.append(nn.ReLU(True))
            i = i//2
        decoder_modules.append(nn.ConvTranspose1d(self.ndf, self.nc, 4, 2, 1, bias=False))
        decoder_modules.append(nn.Tanh())

        self.decoder = nn.Sequential(*decoder_modules)

        print(encoder_modules)
        print(decoder_modules)

    def forward(self, input):
        input = torch.transpose(input, 1, 2)
        z = self.encoder(input)
        out = self.decoder(z)
        out = torch.transpose(out, 1, 2)
        return out, z


class Discriminator(nn.Module):
    def __init__(self, nc=25, seq_len=128, device=None):
        super(Discriminator, self).__init__()
        self.device = device
        self.nc = nc
        self.ndf = 32

        modules = []
        modules.append(nn.Conv1d(self.nc, self.ndf, 4, 2, 1, bias=False))
        modules.append(nn.LeakyReLU(0.2, inplace=True))

        cur_len = seq_len // 2
        i = 1
        while cur_len > 5:
            modules.append(nn.Conv1d(self.ndf * i, self.ndf * i * 2, 4, 2, 1, bias=False))
            modules.append(nn.BatchNorm1d(self.ndf * i*2))
            modules.append(nn.LeakyReLU(0.2, inplace=True))
            i = i * 2
            cur_len = cur_len // 2

        self.features = nn.Sequential(*modules)

        self.classifer = nn.Sequential(
            nn.Conv1d(self.ndf * i, 1, cur_len, 1, 0, bias=False),
            # state size. 1
            nn.Sigmoid()
        )

    def forward(self, input):
        input = torch.transpose(input, 1, 2)

        features = self.features(input)
        out = self.classifer(features).view(-1, 1).squeeze(1)

        return out, features


def weights_init(mod):
    """
    Custom weights initialization called on netG, netD and netE
    :param m:
    :return:
    """
    classname = mod.__class__.__name__
    if classname.find('Conv') != -1:
        # mod.weight.data.normal_(0.0, 0.02)
        nn.init.xavier_normal_(mod.weight.data)
        # nn.init.kaiming_uniform_(mod.weight.data)

    elif classname.find('BatchNorm') != -1:
        mod.weight.data.normal_(1.0, 0.02)
        mod.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        torch.nn.init.xavier_uniform(mod.weight)
        mod.bias.data.fill_(0.01)


class BeatGAN(MLmodel):
    def __init__(self, dataloader):
        super(BeatGAN, self).__init__(dataloader)
        self.dataloader = None
        self.device = None
        self.lamda_value = None

        self.generator = None
        self.discriminator = None

        self.mse = None
        self.bce = None

        self.optimizerG = None
        self.optimizerD = None

        self.out_dir = None
        self.iteration = None

        self.fix_input = None

    def train(self, data=None):
        self.generator.apply(weights_init)
        self.discriminator.apply(weights_init)
        self.generator.train()
        self.discriminator.train()

        best_auc = 0
        best_auc_epo = 0
        best_f1 = 0
        best_f1_epo = 0

        for epoch in tqdm(range(self.max_epoch)):
            self.train_epoch()
            # val_auc = self.val_epoch()
            # test_auc, test_th, test_f1 = self.test(intrain=True)

            self.save_cur_model(self.out_dir, "cur_w.pth")

            # if val_auc > best_val_auc:
            #     best_val_auc = val_auc
            #     best_val_auc_epo = epoch
            #     self.save_cur_model(self.out_dir, "beat_val_auc_w.pth")

            # if test_auc > best_auc:
            #     best_auc = test_auc
            #     best_auc_epo = epoch
            #     self.save_cur_model(self.out_dir, "beat_auc_w.pth")
            #
            # if test_f1 > best_f1:
            #     best_f1 = test_f1
            #     best_f1_epo = epoch
            #     self.save_cur_model(self.out_dir, "beat_f1_w.pth")

            # print(
            #     "[{}]test auc:{:.4f} th:{:.4f} f1:{:.4f} \n".format(epoch, test_auc, test_th, test_f1))
            # print("best auc:{} in [{}]    best f1:{}  in [{}]".format(best_auc, best_auc_epo, best_f1, best_f1_epo))
            # print("best val auc:{} in [{}]".format(best_val_auc, best_val_auc_epo))

    def train_epoch(self):
        self.generator.train()
        self.discriminator.train()

        for i, data in enumerate(self.dataloader):
            self.iteration += 1

            data_X, y = data

            data_X = data_X.to(self.device)

            if self.fix_input is None:
                self.fix_input = data_X

            # train discriminator
            self.discriminator.zero_grad()

            # train with real

            d_real_prob, _ = self.discriminator(data_X)

            # train with fake
            fake_data, enc_z = self.generator(data_X)
            d_fake_prob, _ = self.discriminator(fake_data)

            loss_d_real = self.bce(d_real_prob, torch.full((d_real_prob.size(0),), 1, dtype=torch.float, device=self.device))
            loss_d_fake = self.bce(d_fake_prob, torch.full((d_fake_prob.size(0),), 0, dtype=torch.float, device=self.device))

            loss_d = loss_d_real + loss_d_fake
            loss_d.backward()
            self.optimizerD.step()

            # train generator
            self.generator.zero_grad()

            fake_data, _ = self.generator(data_X)
            d_real_prob, d_real_feat = self.discriminator(data_X)
            d_fake_prob, d_fake_feat = self.discriminator(fake_data)

            loss_g_rec = self.mse(data_X, fake_data)
            loss_g_adv = self.mse(d_real_feat, d_fake_feat)

            loss_g = loss_g_rec + loss_g_adv * self.lamda_value
            loss_g.backward()

            self.optimizerG.step()

            if (i % 100) == 0:
                print("{}:loss_g(rec/adv):{}/{},loss_d(real/fake):{}/{}".format(i, loss_g_rec,
                                                                                loss_g_adv * self.lamda_value,
                                                                                loss_d_real, loss_d_fake))
                # fake_data, z = self.generator(self.fix_input)
                # show_num = 8
                # img_tensor = multi_ts2image(fake_data.detach().cpu().numpy()[:show_num],
                #                             self.fix_input.cpu().numpy()[:show_num])
                # for img_idx in range(show_num):
                #     self.writer.add_image("image/rec" + str(img_idx), img_tensor[img_idx], global_step=self.iteration,
                #                           dataformats='HWC')

            if loss_d.item() < 5e-6:
                self.discriminator.apply(weights_init)
                print('Reloading dis net')

    def test(self, data=None, intrain=False, scale=True):
        # if not intrain:
        #     self.load_model(self.out_dir, "cur_w.pth")
        self.generator.eval()
        self.discriminator.eval()

        if data is None:
            dataloader = self.dataloader
        else:
            dataloader = preprocess_data(data, labels=None, param={}, is_train=False)
        
        rec_diff = self.get_diff(dataloader)
        # print(rec_diff)
        return rec_diff

    def save_cur_model(self, outdir, filename):
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        state = {"generator": self.generator.state_dict(),
                 "discriminator": self.discriminator.state_dict(),
                 }
        torch.save(state, os.path.join(outdir, filename))

    def save_model_to(self, path):
        state = {"generator": self.generator.state_dict(),
                 "discriminator": self.discriminator.state_dict(),
                 }
        torch.save(state, path)

    def load_model(self, outdir, filename):
        state = torch.load(os.path.join(outdir, filename), map_location=self.device)
        self.generator.load_state_dict(state["generator"])
        self.discriminator.load_state_dict(state["discriminator"])

    def load_model_from(self, path):
        state = torch.load(path, map_location=self.device)
        self.generator.load_state_dict(state["generator"])
        self.discriminator.load_state_dict(state["discriminator"])

    def get_diff(self, dataloader):

        rec_diff = []
        ori_ts = []
        rec_ts = []
        rec_err = []
        with torch.no_grad():
            for i, data in enumerate(dataloader):
                data_X, y = data

                data_X = data_X.to(self.device)
                # data_cond = data_cond.to(self.device)

                rec_x, z = self.generator(data_X)

                mse_metric = torch.mean(torch.sum(torch.pow(data_X - rec_x, 2), dim=2),
                                        dim=1).detach().cpu().numpy()

                rec_diff.append(mse_metric)
                ori_ts.append(data_X.detach().cpu().numpy())
                rec_ts.append(rec_x.detach().cpu().numpy())
                rec_err.append(torch.sum(torch.pow(data_X - rec_x, 2), dim=2).detach().cpu().numpy())

        rec_diff = np.concatenate(rec_diff)
        ori_ts = np.concatenate(ori_ts)
        rec_ts = np.concatenate(rec_ts)
        rec_err = np.concatenate(rec_err)

        ori_ts = np.transpose(ori_ts, axes=(0, 2, 1))
        rec_ts = np.transpose(rec_ts, axes=(0, 2, 1))

        return rec_diff, ori_ts, rec_ts, rec_err


class BeatGAN_CNN(BeatGAN):
    def __init__(self, tensor, *args, **kwargs):
        super(BeatGAN_CNN, self).__init__(tensor)
        self.dataloader = preprocess_data(tensor, labels=None, param=kwargs, is_train=True)
        self.device = device
        self.lamda_value = param_default(kwargs, "lambda", param_default_dict)

        self.max_epoch = param_default(kwargs, "max_epoch", param_default_dict)

        self.generator = Generator(nc=param_default(kwargs, "input_size", param_default_dict),
                                   nz=param_default(kwargs, "rep_size", param_default_dict),
                                   seq_len=param_default(kwargs, "seq_len", param_default_dict), device=device).to(device)
        self.discriminator = Discriminator(nc=param_default(kwargs, "input_size", param_default_dict),
                                           seq_len=param_default(kwargs, "seq_len", param_default_dict), device=device).to(device)

        self.mse = nn.MSELoss()
        self.bce = nn.BCELoss()

        self.optimizerG = optim.Adam(self.generator.parameters(), lr=param_default(kwargs, "lr", param_default_dict))

        self.optimizerD = optim.Adam(self.discriminator.parameters(), lr=param_default(kwargs, "lr", param_default_dict))

        self.out_dir = param_default(kwargs, "out_path", param_default_dict)
        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)

        self.iteration = 0

        self.fix_input = None
        self.fix_data_cond = None

        if kwargs.__contains__("model_path"):
            self.load_model_from(kwargs["model_path"])

    def fit(self, data=None):
        return self.train(data)

    def predict(self, data=None):
        return self.test(data)

    def train(self, data=None):
        super().train(data)
        return self
