import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm_
from sklearn.metrics import roc_curve, auc, roc_auc_score, accuracy_score, average_precision_score , precision_recall_curve, average_precision_score

class VAE(nn.Module):
    def __init__(self, input_size, hidden_size, latent_size, learning_rate):
        super(VAE, self).__init__()
        self.latent_size = latent_size
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, latent_size * 2)  # Two times latent_size for mean and variance
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size)
)
        
    

# Loss function
        # self.loss_function = nn.BCEWithLogitsLoss(reduction='sum')
        self.loss_function = nn.MSELoss(reduction='sum')

        # Optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        # Loss function

    def reparameterize(self, mu, log_var, clip_value=15):
    # Clip the log_var values to avoid numerical instability
        # clipped_log_var = torch.clamp(log_var, max=clip_value)

        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # Encode
        encoding = self.encoder(x)
        mu, log_var = encoding[:, :self.latent_size], encoding[:, self.latent_size:]

        # Reparameterize
        z = self.reparameterize(mu, log_var)

        # Decode
        reconstruction = self.decoder(z)

        return reconstruction, mu, log_var

    def compute_loss(self, reconstruction, x, mu, log_var, normal_class_weight, kl_weight):
        # Reconstruction loss
        reconstruction_loss = self.loss_function(reconstruction, x)

        # KL divergence loss
        # kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) * kl_weight # normalized KL 

        # normal_class_loss = normal_class_weight * torch.sum((mu.pow(2) + log_var.exp() - log_var - 1))
        # print(f"KL Loss: {kl_loss:.3f}, Reconstruction Loss: {reconstruction_loss:.3f}")
        return reconstruction_loss + kl_loss  
        # return reconstruction_loss + kl_loss     


    def compute_reconstruction_loss(self, reconstruction, x):
        # Reconstruction loss
        reconstruction_loss = self.loss_function(reconstruction, x)

       
        return reconstruction_loss  
  


    def train_step(self, x, weight, kl_weight):
        self.optimizer.zero_grad()
        reconstruction, mu, log_var = self(x)
        loss = self.compute_loss(reconstruction, x, mu, log_var, weight, kl_weight)
        loss.backward()
        clip_grad_norm_(self.parameters(), max_norm=0.3)

        self.optimizer.step()

        return loss.item()
    


    def evaluate_sequence_of_samples(self, y_validation, predictions):
        return roc_curve(y_validation, predictions)
        


    def get_precision_recall_curve(self, y_validation, predictions):
        precision, recall, _ = precision_recall_curve(y_validation, predictions)
        avg_precision = average_precision_score(y_validation,predictions)
        return precision, recall, avg_precision
    


    def evaluate_sequence_of_samples1(self, y_threshold, thresholds, validation_losses):

        tpr_values = []
        fpr_values = []     
        for threshold in thresholds:
            predictions = (validation_losses > threshold).astype(int)
            tp = np.sum((predictions == 1) & (y_threshold == 1))
            fp = np.sum((predictions == 1) & (y_threshold == 0))
            tn = np.sum((predictions == 0) & (y_threshold == 0))
            fn = np.sum((predictions == 0) & (y_threshold == 1))

            tpr = tp / (tp + fn)
            fpr = fp / (fp + tn)
            tpr_values.append(tpr)
            fpr_values.append(fpr)

        return tpr, fpr
    

    def get_precision_recall_curve1(self, y_threshold, thresholds, validation_losses):
        precision_values = []
        recall_values = []
        average_precision = []
        for threshold in thresholds:
            predictions = (validation_losses > threshold).astype(int)
            tp = np.sum((predictions == 1) & (y_threshold == 1))
            fp = np.sum((predictions == 1) & (y_threshold == 0))
            tn = np.sum((predictions == 0) & (y_threshold == 0))
            fn = np.sum((predictions == 0) & (y_threshold == 1))

            avg_precision = average_precision_score(y_threshold,predictions)
            recall = tp / (tp + fn)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            precision_values.append(precision)
            recall_values.append(recall)
            average_precision.append(avg_precision)


        return precision_values, recall_values
