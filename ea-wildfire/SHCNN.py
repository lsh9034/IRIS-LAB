from pre_import import *


class SH_Forestfire(nn.Module):
    def __init__(self, input_features, cnn_out=64):
        super(SH_Forestfire, self).__init__()
        self.data_layers = nn.Sequential(
            nn.Conv2d(input_features, 64, (3,3), padding='same'),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, (3,3), padding='same'),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, (1,1)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 128, (1,1)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, (1,1)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Conv2d(128,64,(1,1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.last_layers = nn.Sequential(
            nn.Linear(cnn_out, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )
    def forward(self, meta_info, img):
        simple_cnn = self.data_layers(img).squeeze()
        out = self.last_layers(simple_cnn)
        return out


def load_SHCNN(path, band_size, device):
    record = torch.load(path, map_location=device)
    model = SH_Forestfire(band_size)
    model.load_state_dict(record['model_state_dict'])
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters())
    optimizer.load_state_dict(record['optimizer_state_dict'])
    return model, optimizer
