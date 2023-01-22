from torch.utils.data.dataset import Dataset 
import pickle
import random

class ContribDataSet(Dataset): 
    def __init__(self, contributions_df):
        self.contributions = list(contributions_df.itertuples(index=False))

    def __getitem__(self, index):
        self.contributions[index]

    def __len__(self):
        return len(self.contributions)

class LobbyDataSet(Dataset): 
    def __init__(self, lobby_df):

        self.lobbyists = list(contributions_df.itertuples(index=False))     
        # self.lobbyist_dict = {}
        
        # for row in lobby_df():
        #     if row['lobbyist'] not in lobby_df:
        #         self.lobbyist_dict[row['lobbyist']] = []
        #     self.lobbyist_dict[row['lobbyist']] =  row['subject_words']
        
        # self.lobbyists = list(self.lobbyist_dict)

    def __getitem__(self, index):
        self.lobbyists[index]

    def __len__(self):
        return len(self.lobbyists)

    def sample_subjects(self, n):
        subjects = []
        ls = random.sample(self.lobbyists, n)
        for l in ls:
            subjects.append(ls.subject_words)
        return subjects

if __name__ == '__main__':
    word_vectors_dict = pickle.load(open('data/glove_50d.pickle', 'rb'))
    contributions_df = pickle.load(open('data/contributions.pickle', 'rb'))
    lobby_subjects_dict = pickle.load(open('data/lobby_subjects.pickle', 'rb'))
