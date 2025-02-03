import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = Parameter(torch.empty(size=(in_features, out_features)))
        self.a = Parameter(torch.empty(size=(2 * out_features, 1)))
        
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.dropout = nn.Dropout(dropout)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.a)
    
    def forward(self, h, adj):
        Wh = torch.mm(h, self.W)  # Linear transformation
        
        N = Wh.size(0)
        
        # Compute attention scores
        a_input = torch.cat([Wh.repeat(1, N).view(N * N, -1), Wh.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))
        
        # Mask out unconnected nodes
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        
        # Apply softmax
        attention = F.softmax(attention, dim=1)
        attention = self.dropout(attention)
        
        # Apply attention weights
        h_prime = torch.matmul(attention, Wh)
        
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        super(GAT, self).__init__()
        self.dropout = dropout

        # Multi-head attention layers
        self.attentions = nn.ModuleList([GraphAttentionLayer(nfeat, nhid, dropout, alpha, True) for _ in range(nheads)])
        
        # Output layer (single-head)
        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout, alpha, False)
    
    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)  # Multi-head attention
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.out_att(x, adj)  # Output layer
        return F.log_softmax(x, dim=1)


num_nodes = 5      # Number of nodes in the graph
num_features = 10  # Number of input features per node
num_classes = 3    # Number of output classes
hidden_dim = 8     # Hidden layer size
dropout = 0.6      # Dropout rate
alpha = 0.2        # LeakyReLU alpha
num_heads = 4      # Number of attention heads

X = torch.rand((num_nodes, num_features))

# Create a random adjacency matrix (binary, undirected)
adjacency_matrix = (torch.rand((num_nodes, num_nodes)) > 0.5).float()
adjacency_matrix.fill_diagonal_(1)  # Ensure self-connections

# Initialize the GAT model
model = GAT(nfeat=num_features, nhid=hidden_dim, nclass=num_classes, 
            dropout=dropout, alpha=alpha, nheads=num_heads)

# Run forward pass
output = model(X, adjacency_matrix)

# Print outputs
print("Output logits:\n", output)
print("Output shape:", output.shape)  # Expected: (num_nodes, num_classes)
