import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import kneighbors_graph
import seaborn as sns
import matplotlib.pyplot as plt

# Check if GPU is available and set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load data
train_data = pd.read_csv('data/train.csv')
test_data = pd.read_csv('data/test.csv')

# Display the first few rows of the training data to understand its structure
print(train_data.head())

# Data preprocessing
train_data = train_data.dropna()  # Drop rows with missing values
test_data = test_data.dropna()

feature_columns = [
    'Gear oil temperature (°C)', 
    'Front bearing temperature (°C)', 
    'Rear bearing temperature (°C)', 
    'Gear oil inlet temperature (°C)', 
    'Generator bearing front temperature (°C)', 
    'Generator bearing rear temperature (°C)', 
    'Rotor bearing temp (°C)', 
    'Stator temperature 1 (°C)'
]

# Normalize the features
scaler = StandardScaler()
train_features = scaler.fit_transform(train_data[feature_columns])
test_features = scaler.transform(test_data[feature_columns])

# Convert to PyTorch tensors and move to GPU
train_tensor = torch.tensor(train_features, dtype=torch.float).to(device)
test_tensor = torch.tensor(test_features, dtype=torch.float).to(device)

# Build KNN graph
def build_knn_graph(data, k=5):
    adjacency_matrix = kneighbors_graph(data, k, mode='connectivity', include_self=True)
    edge_index = torch.tensor(np.array(adjacency_matrix.nonzero()), dtype=torch.long).to(device)
    return edge_index

# Build graph for training data
train_edge_index = build_knn_graph(train_features, k=5)
train_data_graph = Data(x=train_tensor, edge_index=train_edge_index).to(device)

# Build graph for test data (optional)
test_edge_index = build_knn_graph(test_features, k=5)
test_data_graph = Data(x=test_tensor, edge_index=test_edge_index).to(device)

# GCN + Transformer model
class GCN_Transformer(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_heads=4, num_layers=2):
        super(GCN_Transformer, self).__init__()
        
        # GCN layers
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        
        # Transformer Encoder Layer
        encoder_layers = torch.nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads)
        self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layers, num_layers=num_layers)

        # Output layer
        self.fc = torch.nn.Linear(hidden_dim, output_dim)
        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        # GCN layers
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        
        # Prepare for transformer (Ensure shape [seq_len, batch_size, feature_dim])
        x = x.unsqueeze(0)  # Add sequence dimension (seq_len=1 in this case)
        
        # Apply Transformer Encoder
        x = self.transformer_encoder(x)
        
        # Remove sequence dimension
        x = x.squeeze(0)
        
        # Output layer
        x = self.fc(x)
        
        return x

# Training function
def train_gcn_transformer(data_graph, epochs=100, lr=0.01):
    model = GCN_Transformer(input_dim=data_graph.num_node_features, hidden_dim=16, output_dim=data_graph.num_node_features).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(data_graph)
        
        loss = F.mse_loss(output, data_graph.x)  # Use MSE loss for regression
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")
    
    return model

# Train the model
model = train_gcn_transformer(train_data_graph)

# Make predictions on test data
model.eval()
with torch.no_grad():
    test_predictions = model(test_data_graph)

# Convert predictions back to original scale
test_predictions = scaler.inverse_transform(test_predictions.cpu().numpy())

# Visualize predictions (example: plot first feature)
plt.figure(figsize=(10, 6))
plt.plot(test_predictions[:, 0], label='Predicted')
plt.plot(test_data[feature_columns[0]].values, label='Actual')
plt.title('Prediction vs Actual')
plt.legend()
plt.show()