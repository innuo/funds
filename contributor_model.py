import pandas as pd
import numpy as np
import pickle
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from dataloader import ContribDataSet, LobbyDataSet


class ContributorModel(nn.Module):
    def __init__(self, contributor_list, recipient_list, word_vectors, 
                 embedding_dim=10, phrase_embedding_dim = 10):
        super(ContributorModel, self).__init__()
        self.contributor_list = contributor_list
        self.recipient_list = recipient_list
        self.word_vectors = word_vectors
        self.embedding_dim = embedding_dim
        self.phrase_embedding_dim = phrase_embedding_dim

        word_vec_dim = 50 #TODO

        self.contributor_embedding = nn.ParameterDict()
        for cont in contributor_list:
            if cont.strip() != '':
                self.contributor_embedding[cont] = nn.Parameter(torch.randn(embedding_dim) * 0.01, requires_grad=True)

        self.recipient_embedding = nn.ParameterDict()
        for rec in recipient_list:
            self.recipient_embedding["%d"%rec] = torch.nn.Parameter(torch.randn(embedding_dim) * 0.01, requires_grad=True)
            
        self.phrase_model = nn.Sequential(
                                nn.Linear(word_vec_dim, phrase_embedding_dim),
                                nn.ReLU())

        
        self.contributor_model =  nn.Sequential(
                                nn.Linear(phrase_embedding_dim+embedding_dim, 30),
                                nn.ReLU(),
                                nn.Linear(30, 1),
                                nn.Sigmoid())

    def contributor_recipient_forward(self, contributors, recipient_ids):
        n = len(contributors)
        xc = torch.Tensor(n, self.embedding_dim)
        xr = torch.Tensor(n, self.embedding_dim)
        for i, c in enumerate(contributors):
            xc[i, :] = self.contributor_embedding[c].squeeze()
            xr[i, :] = self.recipient_embedding["%d"%recipient_ids[i]].squeeze()
        
        return xr, xc

    def contributor_subject_forward(self, contributors, subjects):

        y = []
        for i, c in enumerate(contributors):
            if c in  self.contributor_embedding:
                x = self.contributor_embedding[c]
                subs = subjects[i].split(",")
                xw = torch.zeros(self.phrase_embedding_dim)
                for w in subs:
                    if w in self.word_vectors:
                        xw += self.phrase_model(torch.tensor(self.word_vectors[w]))
            
                y.append(self.contributor_model(torch.cat(x.data, xw)))

        return y

    
    def learn_model(self, contrib_dataset, lobby_dataset, options):
        cdl = DataLoader(contrib_dataset, batch_size=options['batch_size'], shuffle=True)
        ldl = DataLoader(lobby_dataset, batch_size=options['batch_size'], shuffle=True)
        opt = optim.Adam(self.parameters(), lr=options['lr'])
        scheduler = StepLR(opt, step_size=5, gamma=0.8)

        ce_loss  = nn.CrossEntropyLoss()
        mse_loss = nn.MSELoss()

        num_batches = 0
        for epoch in range(options['num_epochs']):
            epoch_loss = 0.0
            for cb in cdl:
                opt.zero_grad()
                num_batches += 1
                lb = next(iter(ldl))
                
                xr, xc = self.contributor_recipient_forward(cb[1], cb[0])
                y1 = self.contributor_subject_forward(lb[0], lb[1])
                y0 = self.contributor_subject_forward(lb[0], random.shuffle(lb[1]))

                loss = mse_loss(xr, xc) + ce_loss(y1, torch.ones(y1.shape)) +  ce_loss(y0, torch.zeros(y0.shape))
                loss.backward()
                opt.step()
                epoch_loss += loss
            
            scheduler.step()
            
            print ("Epoch = %2d, total_loss = %.5f"%(epoch, epoch_loss))


if __name__ == '__main__':
    word_vectors_dict = pickle.load(open('data/glove_50d.pickle', 'rb'))
    contributions_df = pickle.load(open('data/contributions.pickle', 'rb'))
    lobby_subjects_dict = pickle.load(open('data/lobby_subjects.pickle', 'rb'))

    lds = LobbyDataSet(lobby_subjects_dict["lobby_df"])
    cdf = ContribDataSet(contributions_df)



    cm = ContributorModel(pd.unique(contributions_df.contributor),
                pd.unique(contributions_df.recipient_id), 
                word_vectors_dict,
                embedding_dim=10, phrase_embedding_dim = 10)

    options = {'batch_size':10, 'lr':0.001, 'num_epochs':10}

    cm.learn_model(cdf, lds, options)