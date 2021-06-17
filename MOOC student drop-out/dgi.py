import torch
import torch.nn as nn

# mutual information regularization

class AvgReadout(nn.Module):
    def __init__(self):
        super(AvgReadout, self).__init__()

    def forward(self, seq, msk):
        if msk is None:
            return torch.mean(seq,0)
        else:
            msk = torch.unsqueeze(msk, -1)
            return torch.sum(seq * msk, 1) / torch.sum(msk)

class Discriminator(nn.Module):
    def __init__(self, n_in, n_out):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_in, n_out, 1)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, z, h_pl, h_mi, s_bias1=None, s_bias2=None):

        z_x = torch.unsqueeze(z, 1).t()
        z_x_1 = z_x.repeat(h_pl.shape[0],1)
        z_x_2= z_x.repeat(h_mi.shape[0],1)


        sc_1 = self.f_k(h_pl, z_x_1)
        sc_2 = self.f_k(h_mi, z_x_2)

        if s_bias1 is not None:
            sc_1 += s_bias1
        if s_bias2 is not None:
            sc_2 += s_bias2

        logits = torch.cat((sc_1.t(), sc_2.t()), 1) 

        return logits

class DGI_(nn.Module):

    def __init__(self, n_in, n_out):
        super(DGI_, self).__init__()

        self.read = AvgReadout()

        self.sigm = nn.Sigmoid()

        self.disc = Discriminator(n_in,n_out)

    def forward(self, h, seq1, seq2, msk, samp_bias1=None, samp_bias2=None):

        z = self.read(h, msk)
        z = self.sigm(z)
        ret = self.disc(z, seq1, seq2, samp_bias1, samp_bias2)

        return ret

