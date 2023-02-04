from torch.utils.data.dataset import Dataset
import pandas as pd 
import random

class ContribDataSet(Dataset): 
    def __init__(self, contributions_df):
        self.contributions = list(contributions_df.itertuples(index=False))
        self.unique_recipients = pd.unique(contributions_df.recipient_id)
        self.unique_contributors = pd.unique(contributions_df.contributor)

    def __getitem__(self, index):
        return list(self.contributions[index])

    def __len__(self):
        return len(self.contributions)

class LobbyDataSet(Dataset): 
    def __init__(self, lobby_df):
        self.lobbyists = list(lobby_df.itertuples(index=False))     
    
    def __getitem__(self, index):
        a = list(self.lobbyists[index])
        b = [a[0], ",".join(a[1])]

        return b

    def __len__(self):
        return len(self.lobbyists)

    def sample_subjects(self, n):
        subjects = []
        ls = random.sample(self.lobbyists, n)
        for l in ls:
            subjects.append(ls.subject_words)
        return subjects

