import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path

class Trainer():
    def __init__(self, 
                 model, 
                 optimizer,
                 loss_fn,
                 lr_scheduler,
                 train_data,
                 val_data,
                 device,
                 num_epochs: int, 
                 training_save_dir,
                 batch_size: int = 4,
                 val_frequency: int = 1) -> None:
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.lr_scheduler = lr_scheduler
        self.train_data = train_data
        self.val_data = val_data
        self.device = device
        self.num_epochs = num_epochs
        self.training_save_dir = training_save_dir
        self.batch_size = batch_size
        self.val_frequency = val_frequency
        self.train_loader = torch.utils.data.DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)
        self.val_loader = torch.utils.data.DataLoader(self.val_data, batch_size=self.batch_size, shuffle=False)

    def _train_epoch(self, epoch_idx: int):
        self.model.train()
        train_loss = 0
        for batch, (X, y) in enumerate(self.train_loader):
            X, y = X.to(self.device), y.to(self.device)

            # Compute prediction error
            pred = self.model(X)
            loss = self.loss_fn(pred, y)
            train_loss += loss.item()

            # Backpropagation
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
        
        train_loss /= len(self.train_loader)
        print(f"train loss: {train_loss}")
        return train_loss

        
    def _val_epoch(self, epoch_idx:int):
        num_batches = len(self.val_loader)
        self.model.eval()
        val_loss = 0
        with torch.no_grad():
            for X, y in self.val_loader:
                X, y = X.to(self.device), y.to(self.device)
                pred = self.model(X)
                val_loss += self.loss_fn(pred, y).item()
        val_loss /= num_batches
        print(f"val loss: {val_loss}")
        return val_loss

    def train(self) -> None:
        best_val_loss = np.inf
        for epoch_idx in tqdm(range(self.num_epochs)):
            train_loss = self._train_epoch(epoch_idx)
            if epoch_idx % self.val_frequency == 0 or epoch_idx == self.num_epochs - 1:
                val_loss = self._val_epoch(epoch_idx)
                if val_loss < best_val_loss:
                    print(f"Saving model, val loss improved from {best_val_loss} to {val_loss}")
                    best_val_loss = val_loss
                    self.model.save(f"{self.training_save_dir}/{self.model.name}_best")
            self.lr_scheduler.step()

        
alpha = 1/2
beta = 1/2
def frobenius_loss(pred, target):
    pred[:, :3] = pred[:, :3]
    # first three elements are translation, next 9 are flattened rotation matrix
    transl_loss = alpha*torch.mean((pred[:, :3] - target[:, :3])**2)
    rot_loss = beta*torch.mean((pred[:, 3:] - target[:, 3:])**2)
    return (transl_loss + rot_loss)
    

def train(model, train_data, val_data, num_epochs=30, lr=0.01, gamma=0.95, device='cuda', batch_size=32):
    optimizer = torch.optim.AdamW(model.parameters(), amsgrad=True, lr=lr) 
    val_frequency = 1
    loss_fn = frobenius_loss

    model_save_dir = Path("saved_models")
    model_save_dir.mkdir(exist_ok=True)

    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    
    trainer = Trainer(model, 
                    optimizer,
                    loss_fn,
                    lr_scheduler,
                    train_data,
                    val_data,
                    device,
                    num_epochs, 
                    model_save_dir,
                    batch_size,
                    val_frequency = val_frequency)
    trainer.train()
    return model