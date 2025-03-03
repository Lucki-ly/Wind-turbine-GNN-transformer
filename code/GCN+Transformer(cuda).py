import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error
import seaborn as sns
sns.set()
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load data
data = pd.read_csv('data/dataset.csv')
submit = pd.read_csv('data/submission.csv')

data.columns = ['风机编号', '时间戳', '风速', '功率', '风轮转速']

# Add derived features
def speed(v):
    return 0 if v > 7 else 1
data['风轮转速_01'] = data['风轮转速'].apply(speed)

data['风轮直径'] = data['风机编号'].apply(lambda id_: 100.5 if id_ == 5 else 115 if id_ == 11 else 104.8 if id_ == 12 else 99)
data['额定功率'] = 2000
data['切入风速'] = data['风机编号'].apply(lambda id_: 2.5 if id_ == 11 else 3)
data['切出风速'] = data['风机编号'].apply(lambda id_: 22 if id_ in [5, 12] else 19 if id_ == 11 else 25)
data['风轮转速范围'] = data['风机编号'].apply(lambda id_: [5.5, 19] if id_ == 5 else [5, 14] if id_ == 11 else [5.5, 17] if id_ == 12 else [8.33, 16.8])
data['时间戳'] = pd.to_datetime(data['时间戳'])
data = data.sort_values('时间戳')

# Prepare numerical data
data_num = data[['风速', '功率', '风轮转速']]
X_std = StandardScaler().fit_transform(data_num.values)
data_std = pd.DataFrame(X_std)

# Construct graph data using KNN
def build_knn_graph(data, k=5):
    adjacency_matrix = kneighbors_graph(data, k, mode='connectivity', include_self=True)
    edge_index = torch.tensor(np.array(adjacency_matrix.nonzero()), dtype=torch.long)
    return edge_index

edge_index = build_knn_graph(X_std)
data_tensor = torch.tensor(X_std, dtype=torch.float)

# Check if GPU is available and set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Move data to the selected device (GPU or CPU)
data_tensor = data_tensor.to(device)
edge_index = edge_index.to(device)

data_graph = Data(x=data_tensor, edge_index=edge_index)

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

        # Output layer (optional, based on your task)
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
    model = GCN_Transformer(input_dim=data_graph.num_node_features, hidden_dim=16, output_dim=data_graph.num_node_features)
    model = model.to(device)  # Move model to the selected device
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(data_graph)
        
        loss = F.mse_loss(output, data_graph.x)  # Ensure dimensions match
        
        loss.backward()
        optimizer.step()

        # Calculate additional metrics
        predictions = output.detach().cpu().numpy()  # Move predictions to CPU for evaluation
        ground_truth = data_graph.x.cpu().numpy()  # Move ground truth to CPU for evaluation
        mse = mean_squared_error(ground_truth, predictions)
        mae = mean_absolute_error(ground_truth, predictions)

        # Binary classification for accuracy calculation
        binary_pred = (predictions >= np.mean(predictions, axis=0)).astype(int)
        binary_true = (ground_truth >= np.mean(ground_truth, axis=0)).astype(int)
        acc = np.mean([accuracy_score(binary_true[:, i], binary_pred[:, i]) for i in range(binary_true.shape[1])])

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}, Accuracy: {acc}, MSE: {mse}, MAE: {mae}")
    
    return model

# Train the model
model = train_gcn_transformer(data_graph)

# Get the predictions
data_graph.x = model(data_graph).detach()

# Visualize some of the results (optional)
plt.figure(figsize=(10, 6))
sns.histplot(data['风速'], kde=True, color='blue')
plt.title('Wind Speed Distribution')
plt.xlabel('Wind Speed (m/s)')
plt.ylabel('Frequency')
plt.show()