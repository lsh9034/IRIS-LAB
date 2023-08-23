from pre_import import *


def load_EfficientNet(path, band_size, device):
    a = torch.load(path)
    model = efficient_net.EfficientNet_Forestfire(band_size)
    model.load_state_dict(a)
    model = model.to(device)
    return model


def eval_model(model, dataset, dataloader, set_name, device):
    pred = analyze.make_pred(model, dataloader, device)
    analyze.print_all_analyze(dataset.label.numpy(), pred, set_name)
    analyze.stats_per_site(set_name,dataset, pred)
    
def AUS_eval_model(model, dataset, dataloader, set_name, device):
    pred = analyze.make_pred(model, dataloader, device)
    print(np.count_nonzero(np.isnan(pred)))
    non_idx = np.where(np.isnan(pred))
    pred[non_idx]=0
    label = AUS_dataset.get_label_by_subset(dataset).numpy()
    analyze.print_all_analyze(label, pred, set_name)
    

device = torch.device('cuda:2') if torch.cuda.is_available() else torch.device('cpu')
delete_col=share.delete_col
valset, val_dataloader = EA_dataset.EA_Valid_DataLoader(delete_col=delete_col)
testset, test_dataloader = EA_dataset.EA_Test_DataLoader(delete_col=delete_col)

# resnet18, optimizer = ResNet18.load_ResNet18('./models/ResNet18-4bands_znorm_ver1_1_best.pkl', 4, device)
# eval_model(resnet18, valset, val_dataloader,'val', device)
# eval_model(resnet18, testset, test_dataloader,'test', device)


# efficientnet, optimizer = efficient_net.load_EfficientNet('EfficientNet-9bands_best.pkl',9, device)
# eval_model(efficientnet, valset, val_dataloader,'val', device)
# eval_model(efficientnet, testset, test_dataloader,'test', device)

#simple_cnn, optimizer = SimpleCNN.load_SimpleCNN2('SimpleCNN-9bands_best.pkl', 9, device)
#eval_model(simple_cnn, valset, val_dataloader, 'val', device)
#eval_model(simple_cnn, testset, test_dataloader, 'test', device)

shcnn, optimizer = SHCNN.load_SHCNN('./models/SHCNN-8bands_best.pkl', 8, device)
eval_model(shcnn, valset, val_dataloader, 'val', device)
eval_model(shcnn, testset, test_dataloader, 'test', device)


# AUS Data Validate
# aus_valset = AUS_dataset.load_info_and_label(delete_col=[])
# aus_valset, aus_dataloader = AUS_dataset.load_AUS_DataLoader(aus_valset, 1)

#AUS_eval_model(resnet18, aus_valset, aus_dataloader, 'aus_valset-1', device)
#AUS_eval_model(efficientnet, aus_valset, aus_dataloader, 'aus_valset-1', device)
# AUS_eval_model(simple_cnn, aus_valset, aus_dataloader, 'aus_valset-1', device)

