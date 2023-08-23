from pre_import import *

def imputation(data):
    idx = np.where(np.isnan(data))
    data[idx]=0
    

device = torch.device('cuda:2') if torch.cuda.is_available() else torch.device('cpu')

resnet18, resnet_optimizer = ResNet18.load_ResNet18_2('./models/ResNet18-9bands_znorm_ver2_3_best.pkl', 9, device)
efficientnet, efficientnet_optimizer = efficient_net.load_EfficientNet2('./models/EfficientNet-9bands_znorm_ver2_1_best.pkl',9, device=device)
# simplecnn, simplecnn_optimizer = SimpleCNN.load_SimpleCNN2('./models/SimpleCNN-9bands_znorm_ver2_1_best.pkl', 9, device)

resnet_total_pred = np.array([])
efficientnet_total_pred = np.array([])
simplecnn_total_pred = np.array([])

ensemble_total_pred = np.array([])

delete_col = share.delete_col

aus_valset, aus_val_dataloader = AUS_dataset.load_AUS_FULL2_DataLoader(delete_col=delete_col, norm='Z', mode='Full')

resnet_pred, resnet_prob = analyze.make_pred_prob(resnet18, aus_val_dataloader, device)
resnet_total_pred = np.append(resnet_total_pred, resnet_pred)

efficientnet_pred, efficientnet_prob = analyze.make_pred_prob(efficientnet, aus_val_dataloader, device)
efficientnet_total_pred = np.append(efficientnet_total_pred, efficientnet_pred)

# simplecnn_pred, simplecnn_prob = analyze.make_pred_prob(simplecnn, aus_val_dataloader, device)
# simplecnn_total_pred = np.append(simplecnn_total_pred, simplecnn_pred)

imputation(resnet_total_pred)
imputation(efficientnet_total_pred)
# imputation(simplecnn_total_pred)
ensemble_total_pred = np.round(resnet_prob*0.7 + efficientnet_prob*0.3)

analyze.print_all_analyze(aus_valset.label.numpy(), resnet_pred, 'aus_resnet')
analyze.print_all_analyze(aus_valset.label.numpy(), efficientnet_total_pred, 'aus_efficientnet')
analyze.print_all_analyze(aus_valset.label.numpy(), ensemble_total_pred, 'aus_ensemble')
# analyze.print_all_analyze(aus_valset.label.numpy(), simplecnn_total_pred, 'aus_simplecnn')