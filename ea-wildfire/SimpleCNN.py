from pre_import import *


class SimpleCNN_Forestfire(nn.Module):
    def __init__(self, input_features, cnn_out=64):
        super(SimpleCNN_Forestfire, self).__init__()
        self.data_layers = nn.Sequential(
            nn.Conv2d(input_features, 128, (3,3)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, (3,3)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 512, (3,3)),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Conv2d(512,64,(1,1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
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
        #metainfo_out = self.metainfo_layers(meta_info)
        #features = torch.cat([simple_cnn, metainfo_out], dim=-1)
        out = self.last_layers(simple_cnn)
        return out
    
class SimpleCNN_Forestfire2(nn.Module):
    def __init__(self, input_features, cnn_out=64):
        super(SimpleCNN_Forestfire2, self).__init__()
        self.data_layers = nn.Sequential(
            nn.Conv2d(input_features-5, 32, (3,3), padding='same'),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((2,2)),
            nn.Conv2d(32, 32, (3,3), padding='same'),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((2,2)),
            nn.Conv2d(32, cnn_out, (3,3), padding='same'),
            nn.BatchNorm2d(cnn_out),
            nn.ReLU(),
            nn.MaxPool2d((2,2)),
            nn.Dropout(0.5),
            nn.AdaptiveAvgPool2d((1,1))
        )
        self.metainfo_layers = nn.Sequential(
            nn.Linear(5, 16),
            nn.ReLU(),
        )
        self.last_layers = nn.Sequential(
            nn.Linear(cnn_out + 16, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
    def forward(self, meta_info, img):
        simple_cnn = self.data_layers(img).squeeze()
        metainfo_out = self.metainfo_layers(meta_info)
        features = torch.cat([simple_cnn, metainfo_out], dim=-1)
        out = self.last_layers(features)
        return out
        
# class SimpleCNN_Forestfire(nn.Module):
#     def __init__(self, input_features, cnn_out=8):
#         super(SimpleCNN_Forestfire, self).__init__()
#         self.data_layers = nn.Sequential(
#             nn.Conv2d(input_features, 16, (3,3)),
#             nn.BatchNorm2d(16),
#             nn.ReLU(),
#             nn.Conv2d(16, 16, (3,3)),
#             nn.BatchNorm2d(16),
#             nn.ReLU(),
#             nn.AdaptiveAvgPool2d((1,1)),
#             nn.Conv2d(16, cnn_out,(1,1)),
#             nn.BatchNorm2d(cnn_out),
#             nn.ReLU(),
#         )
#         # self.metainfo_layers = nn.Sequential(
#         #     nn.Linear(4, 16),
#         #     nn.ReLU(),
#         #     nn.Linear(16, 32),
#         #     nn.BatchNorm1d(32),
#         #     nn.ReLU(),
#         #     nn.Linear(32,16),
#         #     nn.BatchNorm1d(16),
#         #     nn.ReLU(),
#         # )
#         self.last_layers = nn.Sequential(
#             nn.Linear(cnn_out, 10),
#             nn.BatchNorm1d(10),
#             nn.ReLU(),
#             nn.Dropout(0.3),
#             nn.Linear(10, 1),
#         )
#     def forward(self, meta_info, img):
#         simple_cnn = self.data_layers(img).squeeze()
#         #metainfo_out = self.metainfo_layers(meta_info)
#         #features = torch.cat([simple_cnn, metainfo_out], dim=-1)
#         out = self.last_layers(simple_cnn)
#         return out

def load_SimpleCNN(path, band_size, device):
    record = torch.load(path, map_location=device)
    model = SimpleCNN_Forestfire(band_size)
    model.load_state_dict(record['model_state_dict'])
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters())
    optimizer.load_state_dict(record['optimizer_state_dict'])
    return model, optimizer



def load_SimpleCNN2(path, band_size, device):
    record = torch.load(path, map_location=device)
    model = SimpleCNN_Forestfire2(band_size)
    model.load_state_dict(record['model_state_dict'])
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters())
    optimizer.load_state_dict(record['optimizer_state_dict'])
    return model, optimizer