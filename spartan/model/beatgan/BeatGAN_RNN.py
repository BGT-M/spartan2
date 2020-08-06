import torch
import os
import numpy as np
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from .metric import evaluate
from .preprocess import preprocess_data
from .._model import MLmodel

from .import param_default

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class Encoder(nn.Module):
    def __init__(self, layers, input_size, hidden_size, cond_size, rep_size, device, cell='gru'):
        super(Encoder, self).__init__()
        self.device = device
        self.layers = layers
        self.cell = cell

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.cond_size = cond_size
        self.rep_size = rep_size

        if cell == "rnn":
            self.rnn1 = nn.RNNCell(self.input_size+self.cond_size, self.hidden_size)
            self.rnn2 = nn.ModuleList([nn.RNNCell(self.hidden_size, self.hidden_size) for i in range(self.layers - 1)])
        elif cell == "lstm":
            self.rnn1 = nn.LSTMCell(self.input_size+self.cond_size, self.hidden_size)
            self.rnn2 = nn.ModuleList([nn.LSTMCell(self.hidden_size, self.hidden_size) for i in range(self.layers - 1)])
        elif cell == "gru":
            self.rnn1 = nn.GRUCell(self.input_size + self.cond_size, self.hidden_size)
            self.rnn2 = nn.ModuleList([nn.GRUCell(self.hidden_size, self.hidden_size) for i in range(self.layers - 1)])
        else:
            raise Exception("no this cell:{}".format(cell))

        self.linear = nn.Linear(self.hidden_size, self.rep_size)

    def forward(self, input, cond):
        '''
        :param input:  batch,seq,ts dim
        :param cond: batch,seq,cond dim
        :return:
        '''

        h_t = torch.zeros(input.size(0), self.hidden_size, device=self.device)
        c_t = torch.zeros(input.size(0), self.hidden_size, device=self.device)
        if self.cell == "lstm":
            hc_all = [(h_t, c_t)]
        else:
            hc_all = [h_t]
        for i in range(self.layers-1):
            h_t2 = torch.zeros(input.size(0), self.hidden_size, device=self.device)
            c_t2 = torch.zeros(input.size(0), self.hidden_size, device=self.device)
            if self.cell == "lstm":
                hc_all.append((h_t2, c_t2))
            else:
                hc_all.append(h_t2)
        for i in range(input.size(1)):
            input_t = input[:, i, :]
            if self.cond_size > 0:
                x = torch.cat([input_t, cond[:, i, :]], dim=1)
            else:
                x = input_t
            hc_all[0] = self.rnn1(x, hc_all[0])
            for i in range(1, self.layers):
                if self.cell == "lstm":
                    hc_all[i] = self.rnn2[i-1](hc_all[i-1][0], hc_all[i])
                else:
                    hc_all[i] = self.rnn2[i - 1](hc_all[i - 1], hc_all[i])
        if self.cell == "lstm":
            z = self.linear(hc_all[-1][0])
        else:
            z = self.linear(hc_all[-1])
        return z


class Decoder(nn.Module):
    def __init__(self, layers, input_size, hidden_size, cond_size, rep_size, device, cell="gru"):
        '''
        :param input_size:  time series dims
        :param hidden_size:
        :param cond_size:  condition dims
        '''
        super(Decoder, self).__init__()
        self.device = device
        self.layers = layers
        self.cell = cell

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.cond_size = cond_size
        self.rep_size = rep_size

        if cell == "rnn":
            self.rnn1 = nn.RNNCell(self.input_size+self.cond_size, self.hidden_size)
            self.rnn2 = nn.ModuleList([nn.RNNCell(self.hidden_size, self.hidden_size) for i in range(self.layers - 1)])
        elif cell == "lstm":

            self.rnn1 = nn.LSTMCell(self.input_size+self.cond_size, self.hidden_size)
            self.rnn2 = nn.ModuleList([nn.LSTMCell(self.hidden_size, self.hidden_size) for i in range(self.layers - 1)])
        elif cell == "gru":
            self.rnn1 = nn.GRUCell(self.input_size + self.cond_size, self.hidden_size)
            self.rnn2 = nn.ModuleList([nn.GRUCell(self.hidden_size, self.hidden_size) for i in range(self.layers - 1)])
        else:
            raise Exception("no this cell:{}".format(cell))

        self.linear_z = nn.Linear(self.rep_size, self.input_size)
        self.linear_out = nn.Linear(self.hidden_size, self.input_size)
        self.tanh = nn.Tanh()

    def forward(self, seq_len, init_hidden, cond, target=None):
        '''

        :param init_hidden:
        :param cond:
        :param target:
        :return:
        '''
        cur_input = self.linear_z(init_hidden)

        h_t = torch.zeros(init_hidden.size(0), self.hidden_size, device=self.device)
        c_t = torch.zeros(init_hidden.size(0), self.hidden_size, device=self.device)
        if self.cell == "lstm":
            hc_all = [(h_t, c_t)]
        else:
            hc_all = [h_t]

        for i in range(self.layers-1):
            h_t2 = torch.zeros(init_hidden.size(0), self.hidden_size, device=self.device)
            c_t2 = torch.zeros(init_hidden.size(0), self.hidden_size, device=self.device)
            if self.cell == "lstm":
                hc_all.append((h_t2, c_t2))
            else:
                hc_all.append(h_t2)

        outputs = []

        for i in range(seq_len):

            if self.cond_size > 0:
                cond_t = cond[:, i, :]
                x = torch.cat([cur_input, cond_t], dim=1)
            else:
                x = cur_input

            hc_all[0] = self.rnn1(x, hc_all[0])
            for i in range(1, self.layers):
                if self.cell == "lstm":
                    hc_all[i] = self.rnn2[i - 1](hc_all[i - 1][0], hc_all[i])
                else:
                    hc_all[i] = self.rnn2[i - 1](hc_all[i - 1], hc_all[i])

            if self.cell == "lstm":
                cur_input = self.tanh(self.linear_out(hc_all[-1][0]))
            else:
                cur_input = self.tanh(self.linear_out(hc_all[-1]))

            outputs += [cur_input]
        outputs = torch.stack(outputs, 1)
        return outputs


class Discriminator(nn.Module):
    def __init__(self, layers, input_size, hidden_size, cond_size, device, cell="lstm"):
        super(Discriminator, self).__init__()
        self.device = device
        self.layers = layers
        self.cell = cell

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.cond_size = cond_size

        if cell == "rnn":
            self.rnn1 = nn.RNNCell(self.input_size+self.cond_size, self.hidden_size)
            self.rnn2 = nn.ModuleList([nn.RNNCell(self.hidden_size, self.hidden_size) for i in range(self.layers - 1)])
        elif cell == "lstm":

            self.rnn1 = nn.LSTMCell(self.input_size+self.cond_size, self.hidden_size)
            self.rnn2 = nn.ModuleList([nn.LSTMCell(self.hidden_size, self.hidden_size) for i in range(self.layers - 1)])
        elif cell == "gru":
            self.rnn1 = nn.GRUCell(self.input_size + self.cond_size, self.hidden_size)
            self.rnn2 = nn.ModuleList([nn.GRUCell(self.hidden_size, self.hidden_size) for i in range(self.layers - 1)])
        else:
            raise Exception("no this cell:{}".format(cell))

        self.linear = nn.Linear(self.hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input, cond):
        '''

        :param input:  batch,seq,ts dim
        :param cond: batch,seq,cond dim
        :return:
        '''

        h_t = torch.zeros(input.size(0), self.hidden_size, device=self.device)
        c_t = torch.zeros(input.size(0), self.hidden_size, device=self.device)

        if self.cell == "lstm":
            hc_all = [(h_t, c_t)]
        else:
            hc_all = [h_t]
        for i in range(self.layers-1):
            h_t2 = torch.zeros(input.size(0), self.hidden_size, device=self.device)
            c_t2 = torch.zeros(input.size(0), self.hidden_size, device=self.device)
            if self.cell == "lstm":
                hc_all.append((h_t2, c_t2))
            else:
                hc_all.append(h_t2)

        for i in range(input.size(1)):
            input_t = input[:, i, :]
            if self.cond_size > 0:
                x = torch.cat([input_t, cond[:, i, :]], dim=1)
            else:
                x = input_t
            hc_all[0] = self.rnn1(x, hc_all[0])
            for i in range(1, self.layers):
                if self.cell == "lstm":
                    hc_all[i] = self.rnn2[i - 1](hc_all[i - 1][0], hc_all[i])
                else:
                    hc_all[i] = self.rnn2[i - 1](hc_all[i - 1], hc_all[i])
        if self.cell == "lstm":
            logit = self.linear(hc_all[-1][0])
            output = self.sigmoid(logit)
            feature = hc_all[-1][0]
        else:
            logit = self.linear(hc_all[-1])
            output = self.sigmoid(logit)
            feature = hc_all[-1]

        return output, logit, feature


class BeatGAN(MLmodel):
    def __init__(self, data, *args, **param):
        super(BeatGAN, self).__init__(data, *args, **param)
        self.dataloader = preprocess_data(data, False, param)
        self.device = device
        self.lamda_value = param_default(param, "lambda", 1)
        self.seq_len = param_default(param, "seq_len", 64)
        self.max_epoch = param_default(param, "max_epoch", 5)
        self.lr = param_default(param, "lr", 0.01)

        self.encoder = Encoder(param_default(param, "layers", 1), \
            param_default(param, "input_size", 1), param_default(param, "hidden_size", 100), \
                0, param_default(param, "rep_size", 20), device, cell=param_default(param, "net_type", "gru")).to(device)
        self.decoder = Decoder(param_default(param, "layers", 1), \
            param_default(param, "input_size", 1), param_default(param, "hidden_size", 100), \
                0, param_default(param, "rep_size", 20), device, cell=param_default(param, "net_type", "gru")).to(device)
        self.discriminator = Discriminator(param_default(param, "layers", 1), \
            param_default(param, "input_size", 1), param_default(param, "hidden_size", 100), \
                0, device, cell=param_default(param, "net_type", "gru")).to(device)

        self.mse = nn.MSELoss()
        self.bce = nn.BCELoss()
        self.l1loss = nn.L1Loss()

        params = list(self.encoder.parameters()) + list(self.decoder.parameters())
        self.optimizerG = optim.Adam(params, lr=self.lr)
        self.optimizerD = optim.Adam(self.discriminator.parameters(), lr=self.lr)

        self.out_dir = os.path.join("output", self.model_name)
        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)

        self.best_loss = float("inf")
        self.best_loss_epoch = 0
        self.early_stop = 10
        self.iteration = 0

        self.fix_input = None
        self.fix_data_cond = None

    def train(self):
        self.encoder.train()
        self.decoder.train()
        self.discriminator.train()

        best_auc = 0
        best_auc_epo = 0
        best_f1 = 0
        best_f1_epo = 0

        for epoch in tqdm(range(self.max_epoch)):
            self.train_epoch()

            self.save_cur_model(self.out_dir, "cur_w.pth")

            # print(
            #     "[{}]test auc:{:.4f} th:{:.4f} f1:{:.4f} \n".format(epoch, test_auc, test_th, test_f1))
            # if epoch-self.best_loss_epoch>self.early_stop:
            #     print("early stop!!!")
            #     break
            if epoch > 0 and epoch % 10 == 0:
                for param_group in self.optimizerG.param_groups:
                    param_group['lr'] = param_group['lr'] * 0.75
                for param_group in self.optimizerD.param_groups:
                    param_group['lr'] = param_group['lr'] * 0.75

            # print("best auc:{} in [{}]    best f1:{}  in [{}]".format(best_auc, best_auc_epo, best_f1, best_f1_epo))

    def train_epoch(self):
        self.encoder.train()
        self.decoder.train()
        self.discriminator.train()

        for i, data in enumerate(self.dataloader):
            self.iteration += 1

            data_X = data
            data_cond = None

            data_X = data_X.to(self.device)

            if self.fix_input is None:
                self.fix_input = data_X
                self.fix_data_cond = data_cond

            # train discriminator
            # self.discriminator.zero_grad()
            self.optimizerD.zero_grad()

            # train with real

            d_real_prob, d_real_logit, d_real_feat = self.discriminator(data_X, data_cond)

            # train with fake
            z = self.encoder(data_X, data_cond)
            rec_x = self.decoder(self.seq_len, z, data_cond)
            d_fake_prob, d_fake_logit, d_fake_feat = self.discriminator(rec_x, data_cond)

            loss_d_real = self.bce(d_real_prob, torch.full((d_real_prob.size(0),), 1, device=self.device))

            loss_d_fake = self.bce(d_fake_prob, torch.full((d_fake_prob.size(0),), 0, device=self.device))

            loss_d = loss_d_real+loss_d_fake
            loss_d.backward()
            self.optimizerD.step()

            # train generator
            # self.encoder.zero_grad()
            # self.decoder.zero_grad()
            self.optimizerG.zero_grad()

            z = self.encoder(data_X, data_cond)
            rec_x = self.decoder(self.seq_len, z, data_cond)
            d_real_prob, d_real_logit, d_real_feat = self.discriminator(data_X, data_cond)
            d_fake_prob, d_fake_logit, d_fake_feat = self.discriminator(rec_x, data_cond)

            loss_g_rec = self.mse(data_X, rec_x)
            loss_g_adv = self.mse(d_real_feat, d_fake_feat)
            # loss_g_adv=self.bce(d_fake_prob,torch.full((d_fake_prob.size(0),), 1,device=self.device))

            loss_g = loss_g_rec+loss_g_adv*self.lamda_value
            loss_g.backward()
            self.optimizerG.step()

            if (i % 100) == 0:
                print("{}:loss_g(rec/adv):{}/{},loss_d(real/fake):{}/{}".format(i, loss_g_rec, loss_g_adv*self.lamda_value, loss_d_real, loss_d_fake))
                # z = self.encoder(self.fix_input, self.fix_data_cond)
                # rec_x = self.decoder(self.seq_len,z, self.fix_data_cond)
                # show_num=8
                # img_tensor=multi_ts2image(rec_x.detach().cpu().numpy()[:show_num],self.fix_input.cpu().numpy()[:show_num])
                # for img_idx in range(show_num):
                #     self.writer.add_image("image/rec"+str(img_idx),img_tensor[img_idx], global_step=self.iteration, dataformats='HWC')

    def test(self, intrain=False):
        # if not intrain:
        #     self.load_model(self.out_dir,"cur_w.pth")
        self.encoder.eval()
        self.decoder.eval()
        self.discriminator.eval()

        rec_diff = self.get_diff(self.dataloader, save_pic=False)
        print(rec_diff)
        return rec_diff

    def save_cur_model(self, outdir, filename):
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        state = {"encoder": self.encoder.state_dict(), "decoder": self.decoder.state_dict(),
                 "discriminator": self.discriminator.state_dict(),
                 }
        torch.save(state, os.path.join(outdir, filename))

    def save_model_to(self, path):
        state = {"encoder": self.encoder.state_dict(), "decoder": self.decoder.state_dict(),
                 "discriminator": self.discriminator.state_dict(),
                 }
        torch.save(state, path)

    def load_model(self, outdir, filename):
        state = torch.load(os.path.join(outdir, filename))
        self.encoder.load_state_dict(state["encoder"])
        self.decoder.load_state_dict(state["decoder"])
        self.discriminator.load_state_dict(state["discriminator"])

    def load_model_from(self, path):
        state = torch.load(path)
        self.encoder.load_state_dict(state["encoder"])
        self.decoder.load_state_dict(state["decoder"])
        self.discriminator.load_state_dict(state["discriminator"])

    def get_diff(self, dataloader, save_pic=False):

        rec_diff = []
        data_cond = None
        with torch.no_grad():
            for i, data in enumerate(dataloader):
                data_X = data

                data_X = data_X.to(self.device)

                z = self.encoder(data_X, data_cond)
                rec_x = self.decoder(self.seq_len, z, data_cond)

                mse_metric = torch.mean(torch.sum(torch.pow(data_X - rec_x, 2), dim=2),
                                        dim=1).detach().cpu().numpy()
                rec_diff.append(mse_metric)

        rec_diff = np.concatenate(rec_diff)

        return rec_diff

class BeatGAN_RNN(BeatGAN):
    def __init__(self, *args, **kwargs):
        super(BeatGAN_RNN, self).__init__(*args, **kwargs)

    def fit(self):
        self.train()
    
    def predict(self):
        self.test()

    def train(self):
        super().train()
        return self
