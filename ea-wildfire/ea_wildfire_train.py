from pre_import import *

device = torch.device('cuda:3') if torch.cuda.is_available() else torch.device('cpu')

def train_resnet(band_size, path=None):
    if path!=None:
        resnet18, optimizer = ResNet18.load_ResNet18(path, band_size, device)
    else:
        resnet18 = ResNet18.ResNet18_Forestfire(band_size).to(device)
    
    class_weight = [93790/26997]
    weight = torch.FloatTensor(class_weight).to(device)

    if path==None: optimizer = torch.optim.Adam(resnet18.parameters(), weight_decay=5e-2, lr=1e-5)
    resnet_manager = train.TrainManager(loss_fn=torch.nn.BCEWithLogitsLoss(pos_weight=weight),
                                        optimizer=optimizer,
                                        scheduler=torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lambda epoch: 0.999 ** epoch))
    
    print('start train')
    train.train_loop(model=resnet18, epochs=40, train_dataloader=train_dataloader, valid_dataloader=val_dataloader,
                     loss_fn=resnet_manager.loss_fn, optimizer=resnet_manager.optimizer, device=device, 
                     model_name='ResNet18-'+str(band_size)+'bands_znorm_ver1_1', save_best=True, scheduler=resnet_manager.scheduler)
    print('end train')
    
def train_efficientnet(band_size, is_load=False, path=''):
    if is_load==False:
        efficientnet = efficient_net.EfficientNet_Forestfire(band_size).to(device)
    else: 
        efficientnet, optimizer = efficient_net.load_EfficientNet('EfficientNet-8bands_best.pkl', band_size, device)

    class_weight = [93790/26997]
    weight = torch.FloatTensor(class_weight).to(device)


    optimizer = torch.optim.Adam(efficientnet.parameters(), weight_decay=1e-2, lr=1e-5)
    efficientnet_manager = train.TrainManager(loss_fn=torch.nn.BCEWithLogitsLoss(pos_weight=weight),
                                             optimizer=optimizer,
                                             scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 0.99 ** epoch))

    print('start train')
    train.train_loop(model=efficientnet, epochs=20, train_dataloader=train_dataloader, valid_dataloader=val_dataloader,
                     loss_fn=efficientnet_manager.loss_fn, optimizer=efficientnet_manager.optimizer, device=device, 
                     model_name='EfficientNet-'+str(band_size)+'bands', save_best=True, scheduler=efficientnet_manager.scheduler)
    print('end train')

def train_simplecnn(band_size, is_load=False, path=''):
    
    # else: 
    #     simple_cnn, optimizer = efficient_net.load_EfficientNet('EfficientNet-8bands_best.pkl', band_size, device)
    simple_cnn = SimpleCNN.SimpleCNN_Forestfire2(band_size).to(device)
    class_weight = [93790/26997]
    weight = torch.FloatTensor(class_weight).to(device)


    optimizer = torch.optim.Adam(simple_cnn.parameters(), weight_decay=1e-2, lr=2e-5)
    simple_cnn_manager = train.TrainManager(loss_fn=torch.nn.BCEWithLogitsLoss(pos_weight=weight),
                                             optimizer=optimizer,
                                             scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 0.99 ** epoch))

    print('start train')
    train.train_loop(model=simple_cnn, epochs=15, train_dataloader=train_dataloader, valid_dataloader=val_dataloader,
                     loss_fn=simple_cnn_manager.loss_fn, optimizer=simple_cnn_manager.optimizer, device=device, 
                     model_name='SimpleCNN-'+str(band_size)+'bands', save_best=True, scheduler=simple_cnn_manager.scheduler)
    print('end train')


def train_shcnn(band_size, path=None):
    if path!=None:
        sh_cnn, optimizer = SHCNN.load_SHCNN(path, band_size, device)
    else:
        sh_cnn = SHCNN.SH_Forestfire(band_size).to(device)
    
    class_weight = [93790/26997]
    weight = torch.FloatTensor(class_weight).to(device)


    optimizer = torch.optim.Adam(sh_cnn.parameters(), weight_decay=2e-3, lr=2e-5)
    sh_cnn_manager = train.TrainManager(loss_fn=torch.nn.BCEWithLogitsLoss(pos_weight=weight),
                                             optimizer=optimizer,
                                             scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 0.95 ** epoch))

    print('start train')
    train.train_loop(model=sh_cnn, epochs=100, train_dataloader=train_dataloader, valid_dataloader=val_dataloader,
                     loss_fn=sh_cnn_manager.loss_fn, optimizer=sh_cnn_manager.optimizer, device=device, 
                     model_name='SHCNN-'+str(band_size)+'bands', save_best=True, scheduler=sh_cnn_manager.scheduler)
    print('end train')


delete_col=share.delete_col

trainset, train_dataloader = EA_dataset.EA_Train_DataLoader(delete_col=delete_col)
valset, val_dataloader = EA_dataset.EA_Valid_DataLoader(delete_col=delete_col)
band_size=13 - len(delete_col)

#train_resnet(band_size)
#train_efficientnet(band_size)
#train_simplecnn(band_size)
train_shcnn(band_size, './models/SHCNN-8bands_best.pkl')