import torch
from torch.utils.data import DataLoader 
import time
import os
import pickle
import sys
sys.path.append('/root/CLAIRE')
from dev.utils.utils import * 
import torch.nn as nn
import argparse


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-lr', '--learning_rate', type=float, default=5e-4)
    parser.add_argument('-e', '--epoch', type=int, default=2000)
    parser.add_argument('-i', '--input_dim', type=int, default=512)
    parser.add_argument('-d', '--hidden_dim', type=int, default=1280)
    parser.add_argument('-o', '--out_dim', type=int, default=128)
    parser.add_argument('-l', '--num_layers', type=int, default=5)    
    parser.add_argument('--adaptive_rate', type=int, default=200)
    args = parser.parse_args()
    return args

# Define Model with Flexible Layers
class LayerNormNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim, num_layers, device, dtype, drop_out=0.1):
        super(LayerNormNet, self).__init__()
        self.layers = nn.ModuleList()
        self.device = device
        self.dtype = dtype
        self.dropout = nn.Dropout(p=drop_out)

        # Input layer
        self.layers.append(nn.Sequential(
            nn.Linear(input_dim, hidden_dim, dtype=dtype, device=device),
            nn.LayerNorm(hidden_dim, dtype=dtype, device=device),
            nn.ReLU()
        ))

        # Hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim, dtype=dtype, device=device),
                nn.LayerNorm(hidden_dim, dtype=dtype, device=device),
                nn.ReLU()
            ))

        # Output layer
        self.layers.append(nn.Linear(hidden_dim, out_dim, dtype=dtype, device=device))

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.dropout(layer(x))
        x = self.layers[-1](x)
        return x

def get_negative_random(anchor_ec, esm_emb):
    result_ec = random.choices(list(esm_emb.keys()))[0]
    while result_ec == anchor_ec:
        result_ec = random.choices(list(esm_emb.keys()))[0]
    return result_ec

class Dataset(torch.utils.data.Dataset):
    def __init__(self, esm_emb, model_lookup_train, labels):
        self.full_list = labels
        self.esm_emb = esm_emb
        self.model_lookup_train = model_lookup_train
        
    def __len__(self):
        return len(self.full_list)

    def __getitem__(self, index):
        anchor_ec = self.full_list[index] 
        anchor = torch.from_numpy(self.model_lookup_train[index]).unsqueeze(0)
        pos = random.choice(self.esm_emb[anchor_ec]).unsqueeze(0)    
        neg_ec = get_negative_random(anchor_ec, self.esm_emb)
        neg = random.choice(self.esm_emb[neg_ec]).unsqueeze(0) 
        return anchor, pos, neg

def main():
    seed()
    args = parse()
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    # loading ESM embedding
    esm_emb_dict = pickle.load(open('../data/pred_rxn_EC123/esm_emb_dict_ec3.pkl', 'rb'))
    model_lookup_train = pickle.load(open('../data/model_lookup_train.pkl', 'rb'))
    labels = pickle.load(open('../data/pred_rxn_EC123/labels_train_ec3.pkl', 'rb'))
    print('### load......done')

    # initialize model 
    model = LayerNormNet(args.input_dim, args.hidden_dim, args.out_dim, args.num_layers, device, dtype)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999))
    criterion = nn.TripletMarginLoss(margin=1, reduction='mean')
    best_loss = float('inf')
    dataset = Dataset( esm_emb_dict, model_lookup_train, labels)
    train_loader = DataLoader(dataset, batch_size = 6000, shuffle = True)

    # training
    train_loss_plot = []
    for epoch in range(1, args.epoch + 1):
        temp_loss = []
        if epoch % args.adaptive_rate == 0 and epoch != args.epoch + 1:
            optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999))
            torch.save(model.state_dict(), '../results/model/pred_rxn_EC123/train_' + str(epoch) + '.pth')
            if epoch != args.adaptive_rate:
                os.remove('../results/model/pred_rxn_EC123/train_' + str(epoch-args.adaptive_rate) + '.pth')
            dataset = Dataset( esm_emb_dict, model_lookup_train, labels)
            train_loader = DataLoader(dataset, batch_size = 6000, shuffle = True)  

        start_time = time.time()
        model.train()
        total_loss = 0.
        for batch, data in enumerate(train_loader):
            optimizer.zero_grad()
            anchor, positive, negative = data
            anchor_out = model(anchor.to(device=device, dtype=dtype))
            positive_out = model(positive.to(device=device, dtype=dtype))
            negative_out = model(negative.to(device=device, dtype=dtype))

            loss = criterion(anchor_out, positive_out, negative_out)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        train_loss = total_loss/(batch + 1)

        # save the best model
        if (train_loss < best_loss and epoch > 0.8*args.epoch) :
            torch.save(model.state_dict(), '../results/model/pred_rxn_EC123/train_best_'+ str(epoch) + '.pth')
            best_loss = train_loss
            print(f'Best Performance: epoch  {epoch:3d}  |  loss {train_loss:6.8f}')

        end_time = time.time() - start_time
        print('-' * 65)
        print(f'| epoch:{epoch:2d}   |   time:{end_time:3.2f}s   |   training loss:{train_loss:3.6f}')
        print('-' * 65)

        temp_loss.append(epoch)
        temp_loss.append(train_loss)
        train_loss_plot.append(temp_loss)

    # save best model
    torch.save(model.state_dict(), '../results/model/pred_rxn_EC123/train_' + str(epoch) +'_final'+ '.pth')
    # save loss
    pickle.dump(train_loss_plot, open('../results/loss/pred_rxn_EC123/train_loss_10-4.pkl','wb'))


if __name__ == '__main__':
    main()
