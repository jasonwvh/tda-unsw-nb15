import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Configuration and Hyperparameters
DATA_PATH = "../data/UNSW_NB15_training-set.csv"  # Path to your preprocessed UNSW-NB15 CSV file
SYNTHETIC_OUTPUT_PATH = "synthetic_unsw_nb15.csv"

BATCH_SIZE = 128
LATENT_DIM = 100
EPOCHS = 10
LR = 2e-4
BETA1 = 0.5
FEATURE_SCALE_RANGE = (0, 1)  # MinMax scaling range

# Utility Functions
from collections import defaultdict

def load_and_preprocess(data_path):
    df = pd.read_csv(data_path)

    y = df["attack_cat"].copy()
    X = df.drop(columns=["label", "attack_cat", "id"])

    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()

    onehot_info = defaultdict(list)
    X_encoded = pd.get_dummies(X, columns=categorical_cols)

    for cat_col in categorical_cols:
        onehot_info[cat_col] = [col for col in X_encoded.columns if col.startswith(cat_col + "_")]

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    scaler = MinMaxScaler(feature_range=FEATURE_SCALE_RANGE)
    X_scaled = scaler.fit_transform(X_encoded)

    return X_scaled.astype(np.float32), y_enc.astype(np.int64), scaler, le, X_encoded.columns.tolist(), onehot_info



# cGAN Model Architectures
class Generator(nn.Module):
    def __init__(self, latent_dim, num_classes, feature_dim, embed_dim=50):
        super(Generator, self).__init__()
        self.label_embed = nn.Embedding(num_classes, embed_dim)
        self.gen = nn.Sequential(
            nn.Linear(latent_dim + embed_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, feature_dim),
            nn.Tanh()  # Assuming features scaled to [-1, 1]. Adjust activation if using [0,1].
        )

    def forward(self, noise, labels):
        label_embedding = self.label_embed(labels)
        x = torch.cat([noise, label_embedding], dim=1)
        return self.gen(x)

class Discriminator(nn.Module):
    def __init__(self, num_classes, feature_dim, embed_dim=50):
        super(Discriminator, self).__init__()
        self.label_embed = nn.Embedding(num_classes, embed_dim)
        self.disc = nn.Sequential(
            nn.Linear(feature_dim + embed_dim, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, features, labels):
        label_embedding = self.label_embed(labels)
        x = torch.cat([features, label_embedding], dim=1)
        return self.disc(x)

# Training Loop
def train(generator, discriminator, dataloader, device, epochs):
    criterion = nn.BCELoss()
    optimizer_G = optim.Adam(generator.parameters(), lr=LR, betas=(BETA1, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=LR, betas=(BETA1, 0.999))

    for epoch in range(epochs):
        for real_features, real_labels in dataloader:
            batch_size = real_features.size(0)
            real_features = real_features.to(device)
            real_labels = real_labels.to(device)
            real_targets = torch.ones(batch_size, 1, device=device)
            fake_targets = torch.zeros(batch_size, 1, device=device)

            # Train Discriminator
            optimizer_D.zero_grad()
            real_preds = discriminator(real_features, real_labels)
            d_real_loss = criterion(real_preds, real_targets)

            noise = torch.randn(batch_size, LATENT_DIM, device=device)
            random_labels = torch.randint(0, num_classes, (batch_size,), device=device)
            fake_features = generator(noise, random_labels)
            fake_preds = discriminator(fake_features.detach(), random_labels)
            d_fake_loss = criterion(fake_preds, fake_targets)

            d_loss = d_real_loss + d_fake_loss
            d_loss.backward()
            optimizer_D.step()

            # Train Generator
            optimizer_G.zero_grad()
            gen_preds = discriminator(fake_features, random_labels)
            g_loss = criterion(gen_preds, real_targets)
            g_loss.backward()
            optimizer_G.step()

            print(f"[Epoch {epoch+1}/{epochs}]  D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")


def decode_onehot(synth_df, onehot_info):
    for col, onehot_cols in onehot_info.items():
        onehot_array = synth_df[onehot_cols].values
        max_indices = np.argmax(onehot_array, axis=1)
        recovered = [onehot_cols[i].replace(col + "_", "") for i in max_indices]
        synth_df[col] = recovered
        synth_df.drop(columns=onehot_cols, inplace=True)
    return synth_df


# Generate Synthetic Data
def generate_synthetic(generator, device, scaler, label_encoder, feature_columns, num_classes, onehot_info,
                       normal_class_name="Normal", total_samples=10000, normal_ratio=0.9):
    generator.eval()

    # How many samples of each type to generate
    num_normal = int(total_samples * normal_ratio)
    num_attack = total_samples - num_normal

    # Get class indices
    normal_class_idx = list(label_encoder.classes_).index(normal_class_name)
    attack_class_indices = [i for i in range(num_classes) if i != normal_class_idx]

    synthetic_data = []
    synthetic_labels = []

    with torch.no_grad():
        # Generate Normal samples
        noise = torch.randn(num_normal, LATENT_DIM, device=device)
        labels = torch.full((num_normal,), normal_class_idx, dtype=torch.long, device=device)
        fake_features = generator(noise, labels).cpu().numpy()
        fake_features = (fake_features + 1) / 2.0  # Rescale to [0,1]
        fake_orig = scaler.inverse_transform(fake_features)
        synthetic_data.append(fake_orig)
        synthetic_labels.extend([normal_class_idx] * num_normal)

        # Generate Attack samples (uniform random selection of attack types)
        attack_labels = np.random.choice(attack_class_indices, size=num_attack)
        noise = torch.randn(num_attack, LATENT_DIM, device=device)
        labels = torch.tensor(attack_labels, dtype=torch.long, device=device)
        fake_features = generator(noise, labels.to(device)).cpu().numpy()
        fake_features = (fake_features + 1) / 2.0
        fake_orig = scaler.inverse_transform(fake_features)
        synthetic_data.append(fake_orig)
        synthetic_labels.extend(attack_labels)

    synthetic_data = np.vstack(synthetic_data)
    synthetic_labels = np.array(synthetic_labels)
    synthetic_labels_readable = label_encoder.inverse_transform(synthetic_labels)

    synthetic_df = pd.DataFrame(synthetic_data, columns=feature_columns)
    synthetic_df["attack_cat"] = synthetic_labels_readable

    # Optional: decode one-hot columns to original categorical values
    synthetic_df = decode_onehot(synthetic_df, onehot_info)

    return synthetic_df


# Main Execution
if __name__ == "__main__":
    # X, y, scaler, le, feature_columns, onehot_info = load_and_preprocess(DATA_PATH)
    #
    # tensor_X = torch.tensor(X)
    # tensor_y = torch.tensor(y)
    # dataset = TensorDataset(tensor_X, tensor_y)
    # dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    #
    # feature_dim = X.shape[1]
    # num_classes = len(np.unique(y))
    #
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # generator = Generator(LATENT_DIM, num_classes, feature_dim).to(device)
    # discriminator = Discriminator(num_classes, feature_dim).to(device)
    #
    # train(generator, discriminator, dataloader, device, EPOCHS)
    #
    # synthetic_df = generate_synthetic(
    #     generator, device, scaler, le, feature_columns, num_classes,
    #     onehot_info=onehot_info,
    #     normal_class_name="Normal",  # adjust to your dataset's normal class name
    #     total_samples=10000,
    #     normal_ratio=0.95
    # )
    # synthetic_df.to_csv(SYNTHETIC_OUTPUT_PATH, index=False)
    # print(f"Synthetic dataset saved to {SYNTHETIC_OUTPUT_PATH}")

    df = pd.read_csv(SYNTHETIC_OUTPUT_PATH)
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    atts = ['dur', 'sbytes', 'dbytes']
    a_df = df[df['attack_cat'] != 'Normal']
    b_df = df[df['attack_cat'] == 'Normal']
    fig, axs = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    for i, col in enumerate(atts):
        axs[i].plot(b_df.index, b_df[col], color='green', linewidth=1, alpha=0.5, label='Benign')
        axs[i].plot(a_df.index, a_df[col], color='red', linewidth=1, label='Attack')
        axs[i].set_title(f'{col}: Benign vs Attack')
        axs[i].set_ylabel(col)
        axs[i].legend()
    plt.xlabel('Time')
    plt.tight_layout()
    plt.show()