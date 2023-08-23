from pre_import import *

class ResNet18_Forestfire(nn.Module):
    def __init__(self, input_features, resnet_out=64):
        super(ResNet18_Forestfire, self).__init__()
        self.resnet18 = torchvision.models.resnet18()
        self.resnet18.conv1 = nn.Conv2d(input_features, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1,1), bias=False)
        self.resnet18.fc = nn.Linear(512, resnet_out, bias=True)
        # self.img_layers = nn.Sequential(
        #     torch.nn.BatchNorm2d(input_features),
        #     self.resnet18
        # )
        # self.metainfo_layers = nn.Sequential(
        #     nn.Linear(4, 16),
        #     nn.ReLU(),
        #     nn.Linear(16, 32),
        #     nn.BatchNorm1d(32),
        #     nn.ReLU(),
        #     nn.Linear(32,16),
        #     nn.BatchNorm1d(16),
        #     nn.ReLU(),
        # )
        self.last_layers = nn.Sequential(
            nn.Linear(resnet_out, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )
    def forward(self, meta_info, img):
        resnet_out = self.resnet18(img)
        #metainfo_out = self.metainfo_layers(meta_info)
        #features = torch.cat([resnet_out, metainfo_out], dim=-1)
        out = self.last_layers(resnet_out)
        return out
    
    
    
class ResNet18_Forestfire2(nn.Module):
    def __init__(self, input_features, resnet_out=64):
        super(ResNet18_Forestfire2, self).__init__()
        self.resnet18 = torchvision.models.resnet18()
        self.resnet18.conv1 = nn.Conv2d(input_features-5, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1,1), bias=False)
        self.resnet18.fc = nn.Linear(512, resnet_out, bias=True)
        # self.img_layers = nn.Sequential(
        #     torch.nn.BatchNorm2d(input_features),
        #     self.resnet18
        # )
        self.metainfo_layers = nn.Sequential(
            nn.Linear(5, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32,16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
        )
        self.last_layers = nn.Sequential(
            nn.Linear(resnet_out+16, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )
    def forward(self, meta_info, img):
        resnet_out = self.resnet18(img)
        metainfo_out = self.metainfo_layers(meta_info)
        features = torch.cat([resnet_out, metainfo_out], dim=-1)
        out = self.last_layers(features)
        return out
    

def load_ResNet18(path, band_size, device):
    record = torch.load(path, map_location=device)
    model = ResNet18_Forestfire(band_size)
    model.load_state_dict(record['model_state_dict'])
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters())
    optimizer.load_state_dict(record['optimizer_state_dict'])
    return model, optimizer



def load_ResNet18_2(path, band_size, device):
    record = torch.load(path, map_location=device)
    model = ResNet18_Forestfire2(band_size)
    model.load_state_dict(record['model_state_dict'])
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters())
    optimizer.load_state_dict(record['optimizer_state_dict'])
    return model, optimizer