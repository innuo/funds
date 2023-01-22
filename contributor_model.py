import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR


class ContributorModel():
    def __init__(self, contributor_list, recipient_list, word_vectors, 
                 embedding_dim=10, phrase_embedding_dim = 10):

        self.contributor_list = contributor_list
        self.recipient_list = recipient_list
        self.word_vectors = word_vectors

        word_vec_dim = 50 #TODO

        self.contributor_embedding = {}
        for cont in contributor_list:
            self.contributor_embedding[cont] = torch.nn.Parameter(torch.randn(embedding_dim) * 0.01, requires_grad=True)

        self.recipient_embedding = {}
        for cont in recipient_list:
            self.recipient_embedding[cont] = torch.nn.Parameter(torch.randn(embedding_dim) * 0.01, requires_grad=True)
            
        self.phrase_model_linear = nn.Sequential(
                                nn.Linear(word_vec_dim, phrase_embedding_dim),
                                nn.ReLU(),
                                nn.BatchNorm1d)

        
        self.contributor_model =  nn.Sequential(
                                nn.Linear(phrase_embedding_dim+embedding_dim, 30),
                                nn.ReLU(),
                                nn.BatchNorm1d,
                                nn.Sigmoid())

    def forward(self, x):
        


        return x

    
    def learn_model(self, contrib_dataset, lobby_dataset, options):
        cdl = DataLoader(contrib_dataset, batch_size=options['batch_size'], shuffle=True)
        ldl = DataLoader(lobby_dataset, batch_size=options['batch_size'], shuffle=True)
        opt = optim.Adam(self.forward_generator.parameters(), lr=options['lr'])
        scheduler = StepLR(opt, step_size=5, gamma=0.8)

        ce_loss  = nn.CrossEntropyLoss()
        mse_loss = nn.MSELoss()

        num_batches = 0
        for epoch in range(options['num_epochs']):
            for i, data in enumerate(zip(cdl, ldl)):
                opt.zero_grad()
                num_batches += 1



                #x = torch.tensor(x_df.values)
                x_non_missing = x == x   
                x_missing = x != x 

                z_mean, z_std     = self.latent_generator(x)

                z = torch.randn(z_mean.shape) * z_std + z_mean

                z_prior = torch.randn(z.shape)

                ind = torch.rand (z.shape).argsort (dim = 0)
                z_s = torch.zeros (z.shape).scatter_ (0, ind, z)

                x_gen, x_gen_one_hot_dict = self.forward_generator(z, do_df=pd.DataFrame()) #TODO: conditioned
                z_dist_loss  = mmd_loss(z, z_prior) #+ mmd_loss(z, z_s) 
 
                total_loss = options['z_dist_scalar'] * z_dist_loss #+ options['x_dist_scalar'] * x_dist_loss
                            
                for v in self.variable_dict.keys():
                    variable_id = self.variable_dict[v]['id']
                    inds = torch.nonzero(x_non_missing[:,variable_id]).squeeze()

                    if self.variable_dict[v]['type'] == 'categorical':
                        target = x[:,variable_id].type(torch.LongTensor)
                        total_loss += ce_loss(x_gen_one_hot_dict[v][inds], target[inds])
                    else:
                        target = x[:,variable_id].type(torch.FloatTensor)
                        total_loss += mse_loss(x_gen[inds, variable_id], target[inds])

                total_loss.backward()
                opt.step()

                if (num_batches * options['batch_size']) % 5000  == 0:
                    print ("Epoch = %2d, num_b = %5d, total_loss = %.5f, z_loss = %.5f"%
                           (epoch, num_batches, total_loss, z_dist_loss))

            scheduler.step()
