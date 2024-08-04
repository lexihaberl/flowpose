import torch
from tqdm import tqdm

class Predictor():
    def __init__(self, model, device):
        self.model = model
        self.device = device
    
    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            pred = self.model(X.to(self.device)).squeeze(0)
        return pred
    
    def predict_all(self, data):
        predictions = []
        for idx in tqdm(range(len(data))):
            pred = self.predict(data[idx][0].unsqueeze(0))
            predictions.append(pred)
        return predictions