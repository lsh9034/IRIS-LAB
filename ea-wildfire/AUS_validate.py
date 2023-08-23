from pre_import import *


device = torch.device('cuda:2') if torch.cuda.is_available() else torch.device('cpu')

aus_valset = AUS_dataset.load_info_and_label(delete_col=[])

resnet18, resnet_optimizer = ResNet18.load_ResNet18('ResNet18-9bands.pkl', 9, device)
#efficientnet, efficientnet_optimizer = efficient_net.load_EfficientNet('EfficientNet-9bands_best.pkl',9, device=device)
#simplecnn, simplecnn_optimizer = SimpleCNN.load_SimpleCNN('SimpleCNN-9bands_best.pkl', 9, device)

resnet_total_pred = np.array([])
efficientnet_total_pred = np.array([])
simplecnn_total_pred = np.array([])

ensemble_total_pred = np.array([])
label = np.array([])

for i in range(1,16):
    _, aus_val_dataloader = AUS_dataset.load_AUS_DataLoader(aus_valset, i)
    resnet_pred, resnet_prob = analyze.make_pred_prob(resnet18, aus_val_dataloader, device)
    resnet_total_pred = np.append(resnet_total_pred, resnet_pred)

    # efficientnet_pred, efficientnet_prob = analyze.make_pred_prob(efficientnet, aus_val_dataloader, device)
    # efficientnet_total_pred = np.append(efficientnet_total_pred, efficientnet_pred)

    # simplecnn_pred, simplecnn_prob = analyze.make_pred_prob(simplecnn, aus_val_dataloader, device)
    # simplecnn_total_pred = np.append(simplecnn_total_pred, simplecnn_pred)

    # ensemble_prob = (resnet_prob*0.5 + efficientnet_prob*0.5)
    # ensemble_pred = np.round(ensemble_prob)
    # ensemble_total_pred = np.append(ensemble_total_pred, ensemble_pred)

# Eliminating data that has nan values    
nan_idx = aus_valset.nan_idx
bool_idx = np.array([True]*len(aus_valset.label))
bool_idx[nan_idx]=False

label = aus_valset.label.numpy()[bool_idx]
resnet_total_pred = resnet_total_pred[bool_idx]
#efficientnet_total_pred = efficientnet_total_pred[bool_idx]
#simplecnn_total_pred = simplecnn_total_pred[bool_idx]
#ensemble_total_pred = ensemble_total_pred[bool_idx]

analyze.print_all_analyze(label, resnet_total_pred, 'aus_resnet')
#analyze.print_all_analyze(label, efficientnet_total_pred, 'aus_efficientnet')
#analyze.print_all_analyze(label, ensemble_total_pred, 'aus_ensemble')
#analyze.print_all_analyze(label, simplecnn_total_pred, 'aus_simplecnn')