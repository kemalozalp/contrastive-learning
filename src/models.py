import torch
from torch import nn


class Dataset(torch.utils.data.Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, id):
        feature = self.features[id]
        label = self.labels[id]
        return feature, label


class MLP(nn.Sequential):
    def __init__(self, input_size):
        super(MLP, self).__init__(
            nn.Linear(input_size, int(input_size / 2)),
            nn.ReLU(),
            nn.Linear(int(input_size / 2), 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 2),
            nn.LogSoftmax(dim=-1),
        )


class LSTM(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers):
        super(LSTM, self).__init__()
        self.in_size = input_size
        self.hid_size = hidden_size
        self.nlayers = num_layers
        self.out_size = output_size
        self.lstm = nn.LSTM(
            input_size=self.in_size,
            hidden_size=self.hid_size,
            num_layers=self.nlayers,
            batch_first=True,
        )
        self.fc = nn.Linear(self.hid_size, self.out_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.25)

    def forward(self, x):
        output, (hn, cn) = self.lstm(x)
        hn = torch.squeeze(hn, 0)
        out = self.relu(hn)
        out = self.fc(out)
        out = self.dropout(out)

        return out


class supervisedCL(nn.Module):
    def __init__(self, temp: float = 0.2, base_temp: float = 0.2):
        super().__init__()
        self.temp = temp
        self.base_temp = base_temp

    def forward(self, x, y=None, mask=None):
        device = torch.device("cuda") if x.is_cuda else torch.device("cpu")
        bsz = x.shape[0]
        y = y.contiguous().view(-1, 1)

        # mask
        mask = torch.eq(y, y.T).float().to(device)

        # logits
        contrast = x
        anchor = x
        anchor_inner_contrast = torch.div(torch.matmul(anchor, contrast.T), self.temp)
        logits_max, _ = torch.max(anchor_inner_contrast, dim=1, keepdim=True)
        logits = anchor_inner_contrast - logits_max.detach()

        # zero our diagonals
        logits_mask = torch.scatter(
            torch.ones_like(mask), 1, torch.arange(bsz).view(-1, 1).to(device), 0
        )
        mask = mask * logits_mask

        # log_proba
        exp_logits = torch.exp(logits) * logits_mask
        pos_simi = exp_logits * mask
        numerator = pos_simi.sum(1, keepdim=True)
        denominator = exp_logits.sum(1, keepdim=True)
        log_prob_pos = torch.log(numerator) - torch.log(denominator)
        log_prob_pos = log_prob_pos * y

        # loss
        loss = -(self.temp / self.base_temp) * log_prob_pos
        num_pos = y.sum(0)
        loss = torch.div(loss.sum(), num_pos)

        return loss
