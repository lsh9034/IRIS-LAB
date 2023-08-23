from pre_import import *

def load_oldviersion(path, band_size, device):
    a = torch.load(path, map_location=device)
    model = ResNet18.ResNet18_Forestfire(band_size)
    model.load_state_dict(a)
    model = model.to(device)
    return model

def imputation(data):
    idx = np.where(np.isnan(data))
    data[idx]=0
    
device = torch.device('cuda:2') if torch.cuda.is_available() else torch.device('cpu')

# resnet18 = load_oldviersion('/share/wildfire-2/shlee/ResNet18-5bands-best.pkl', 5, device)
# resnet18, resnet_optimizer = ResNet18.load_ResNet18('./models/ResNet18-4bands_znorm_ver1_1_best.pkl', 4, device)
#efficientnet, efficientnet_optimizer = efficient_net.load_EfficientNet('EfficientNet-9bands_best.pkl',9, device=device)
# simplecnn, simplecnn_optimizer = SimpleCNN.load_SimpleCNN2('SimpleCNN-9bands_best.pkl', 9, device)
shcnn, shcnn_optimizer = SHCNN.load_SHCNN('./models/SHCNN-8bands_best.pkl', 8, device)

resnet_total_pred = np.array([])
efficientnet_total_pred = np.array([])
simplecnn_total_pred = np.array([])
shcnn_total_pred = np.array([])

ensemble_total_pred = np.array([])

delete_col = share.delete_col
# delete_col = [0,1,2,3,4,5,6,7,8]
aus_valset, aus_val_dataloader = AUS_dataset.load_AUS_FULL_DataLoader(delete_col=delete_col,norm='Z-Origin', mode='Filter')

# resnet_pred, resnet_prob = analyze.make_pred_prob(resnet18, aus_val_dataloader, device)
# resnet_total_pred = np.append(resnet_total_pred, resnet_pred)

# simplecnn_pred, simplecnn_prob = analyze.make_pred_prob(simplecnn, aus_val_dataloader, device)
# simplecnn_total_pred = np.append(resnet_total_pred, simplecnn_pred)

shcnn_pred, shcnn_prob = analyze.make_pred_prob(shcnn, aus_val_dataloader, device)
shcnn_total_pred = np.append(shcnn_total_pred, shcnn_pred)


# analyze.print_all_analyze(aus_valset.label.numpy(), resnet_pred, 'aus_resnet')
#analyze.print_all_analyze(aus_valset.label.numpy(), efficientnet_total_pred, 'aus_efficientnet')
#analyze.print_all_analyze(aus_valset.label.numpy(), ensemble_total_pred, 'aus_ensemble')
# analyze.print_all_analyze(aus_valset.label.numpy(), simplecnn_total_pred, 'aus_simplecnn')
analyze.print_all_analyze(aus_valset.label.numpy(), shcnn_total_pred, 'aus_shcnn')