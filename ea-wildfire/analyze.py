from pre_import import *

def make_pred(model, dataloader, device):
    model.eval()
    loss=0
    acc=0
    pred=np.array([])
    prob=np.array([])
    weight = [93790/26997]
    weight = torch.FloatTensor(weight).to(device)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=weight)
    with torch.no_grad():
        for meta_info, data, label in dataloader:
            label = label.view((-1,1))
            meta_info = meta_info.to(device)
            data = data.to(device)
            label = label.to(device)
            output = model(meta_info, data)
            a=loss_fn(output,label).item()
            #print(output, label)
            loss+=a
            a = torch.round(torch.sigmoid(output))
            acc += torch.sum(a==label).detach().cpu().numpy()
            pred = np.append(pred, a.detach().cpu().numpy().flatten())
            prob = np.append(prob, torch.sigmoid(output).detach().cpu().numpy().flatten())
        loss/=len(dataloader)
        acc/=len(dataloader.dataset)
        print(f' loss: {loss} / acc: {acc*100}')
    model.train()
    return pred

def make_pred_prob(model, dataloader, device):
    model.eval()
    loss=0
    acc=0
    pred=np.array([])
    prob=np.array([])
    weight = [93790/26997]
    weight = torch.FloatTensor(weight).to(device)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=weight)
    with torch.no_grad():
        for meta_info, data, label in dataloader:
            label = label.view((-1,1))
            meta_info = meta_info.to(device)
            data = data.to(device)
            label = label.to(device)
            output = model(meta_info, data)
            a=loss_fn(output,label).item()
            #print(output, label)
            loss+=a
            a = torch.round(torch.sigmoid(output))
            acc += torch.sum(a==label).detach().cpu().numpy()
            pred = np.append(pred, a.detach().cpu().numpy().flatten())
            prob = np.append(prob, torch.sigmoid(output).detach().cpu().numpy().flatten())
        loss/=len(dataloader)
        acc/=len(dataloader.dataset)
        print(f' loss: {loss} / acc: {acc*100}')
    model.train()
    return pred, prob

def print_scores(name, label, pred):
    print(f'{name} set f1 score: {sklearn.metrics.f1_score(label, pred)}')
    print(f'{name} set recall score: {sklearn.metrics.recall_score(label, pred)}')
    print(f'{name} set precision score: {sklearn.metrics.precision_score(label, pred)}')
    print(f'{name} set Overall Accuracy: {sklearn.metrics.accuracy_score(label, pred)}')
    
def stats_per_site(name, dataset, pred):
    idxs=[]
    for i in range(7):
        idxs.append(np.where(dataset.pre_data_info[:,2]==i)[0])
    
    for i, idx in enumerate(idxs):
        #print(idx)
        if len(idx)==0:continue
        print(f'class {i} analysis')
        print_scores(name, dataset.label[idx], pred[idx])
        print('-------------')
    
def plot_cm(label, pred):
    cm = confusion_matrix(label, pred)
    print('====Confusion Matrix====')
    s = [[str(e) for e in row] for row in cm]
    lens = [max(map(len, col)) for col in zip(*s)]
    fmt = '\t'.join('{{:{}}}'.format(x) for x in lens)
    table = [fmt.format(*row) for row in s]
    print('\n'.join(table))
    print('========================')
    
def print_all_analyze(label, pred, set_name):
    print(f'Analyze {set_name}')
    plot_cm(label, pred)
    print_scores(set_name, label, pred)

