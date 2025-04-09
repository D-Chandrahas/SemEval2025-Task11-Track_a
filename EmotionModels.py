import torch
from torch.optim import Adam
from torch.nn import BCEWithLogitsLoss
from BaseModels import Roberta, ClassificationHead, EncoderClassifier
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from numpy import float32
from collections import defaultdict


class EmotionDataset(Dataset):
    cols = ["text", "anger", "disgust", "fear", "joy", "sadness", "surprise"]
    labels = cols[1:]
    def __init__(self, path):
        (dtypes := defaultdict(lambda: float32))["text"] = str
        self.data = pd.read_csv(path, header=0, dtype=dtypes)
        self.data = self.data.reindex(columns=self.cols, fill_value=float32(-1))
        self.loss_mask = [(x >= 0) for x in self.data.iloc[0, 1:]]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data.iloc[index, 0], torch.tensor(self.data.iloc[index, 1:].array, dtype=torch.float32)
    

class EmotionDataloader(DataLoader):
    def __init__(self, dataset, batch_size=1):
        self.loss_mask = dataset.loss_mask
        super().__init__(dataset, batch_size)


class EmotionClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.device = "cpu"
        encoder = Roberta("FacebookAI/xlm-roberta-base")
        classification_head = ClassificationHead(encoder.config.hidden_size, 384, len(EmotionDataset.labels))
        self.model = EncoderClassifier(encoder, classification_head)

    def to(self, device):
        self.device = device
        self.model.to(device)
        return self
    
    def evaluate(self, data):
        if type(data) != list and type(data) != tuple:
            data = (data,)
        corr, tot = 0, 0
        self.eval()
        with torch.no_grad():
            for dataloader in data:

                try:
                    loss_mask = dataloader.loss_mask
                except AttributeError:
                    loss_mask = slice(None)

                for sequences, labels in dataloader:
                    labels = labels.to(self.device)
                    pred_labels = self.model(sequences)
                    corr += ((pred_labels[:, loss_mask] > 0) == labels[:, loss_mask]).all(dim=-1).sum().item()
                    tot += labels.size(0)

        return corr/tot
    
    def fit(self, train_data, valid_data, epochs, lr=1e-5, loss_target=0.0, save_path="./ckpts/", resume_from=None):
        return self.model.fit(Adam, {"lr": lr}, BCEWithLogitsLoss, {}, train_data, valid_data, epochs, loss_target, save_path, resume_from)
    
    def classify(self, text):
        if type(text) != str: raise TypeError("text must be a string")
        self.eval()
        with torch.no_grad():
            pred_labels =  self.model(text, batch_mode=False)
            return (pred_labels > 0).any(dim = 0).tolist()
        
    def __call__(self, text):
        return self.classify(text)
    
    def save(self, path):
        self.model.save(path)

    def load(self, path):
        return self.model.load(path)