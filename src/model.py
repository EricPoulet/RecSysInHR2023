import torch
from torch import Tensor
from torch_geometric.data import HeteroData
from torch_geometric.nn import SAGEConv,HGTConv, to_hetero
from torch.nn import functional as F

class GNN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.conv1 = SAGEConv(hidden_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.conv3 = SAGEConv(hidden_channels, hidden_channels)
        #self.conv4 = SAGEConv(hidden_channels, hidden_channels)
        #self.conv5 = SAGEConv(hidden_channels, hidden_channels)
        #self.conv6 = SAGEConv(hidden_channels, hidden_channels)
        #self.conv7 = SAGEConv(hidden_channels, hidden_channels)
        #self.conv8 = SAGEConv(hidden_channels, hidden_channels)
        #self.conv9 = SAGEConv(hidden_channels, hidden_channels)
        #self.conv10 = SAGEConv(hidden_channels, hidden_channels)
    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        x = self.conv3(x, edge_index)
        #x = self.conv4(x, edge_index)
        #x = self.conv5(x, edge_index)
        #x = self.conv6(x, edge_index)
        #x = self.conv7(x, edge_index)
        #x = self.conv8(x, edge_index)
        #x = self.conv9(x, edge_index)
        #x = self.conv10(x, edge_index)
        return x
# Our final classifier applies the dot-product between source and destination
# node embeddings to derive edge-level predictions:
class Classifier(torch.nn.Module):
    def forward(self, x_user: Tensor, x_item: Tensor, edge_label_index: Tensor) -> Tensor:
        # Convert node embeddings to edge-level representations:
        edge_feat_user = x_user[edge_label_index[0]]
        edge_feat_item = x_item[edge_label_index[1]]
        # Apply dot-product to get a prediction per supervision edge:
        return (edge_feat_user * edge_feat_item).sum(dim=-1)

class Model(torch.nn.Module):
    def __init__(self, hidden_channels, data, list_abl= [1,1,1,1,1,1,1,1,1,1]):
        super().__init__()
        # Since the dataset does not come with rich features, we also learn two
        # embedding matrices for users and items:

        self.list_abl = list_abl

        self.job_lin = torch.nn.Linear(384 , hidden_channels)
        self.user_lin = torch.nn.Linear(384 , hidden_channels)

        self.user_emb = torch.nn.Embedding(data["candidate"].num_nodes, hidden_channels)
        self.job_emb = torch.nn.Embedding(data["job"].num_nodes, hidden_channels)

        if list_abl[0] == 1:
            self.skill_emb = torch.nn.Embedding(data["skill"].num_nodes, hidden_channels)
        if list_abl[1] == 1:
            self.contract_emb = torch.nn.Embedding(data["contract"].num_nodes, hidden_channels)
        if list_abl[2] == 1:
            self.origin_emb = torch.nn.Embedding(data["origin"].num_nodes, hidden_channels)
        if list_abl[3] == 1:
            self.experience_emb = torch.nn.Embedding(data["experience"].num_nodes, hidden_channels)
        if list_abl[4] == 1:
            self.salary_emb = torch.nn.Embedding(data["salary"].num_nodes, hidden_channels)
        if list_abl[5] == 1:
            self.zip_emb = torch.nn.Embedding(data["zip"].num_nodes, hidden_channels)
        if list_abl[6] == 1:
            self.category_emb = torch.nn.Embedding(data["category"].num_nodes, hidden_channels)
        if list_abl[7] == 1:
            self.recruiter_emb = torch.nn.Embedding(data["recruiter"].num_nodes, hidden_channels)
        if list_abl[8] == 1:
            self.company_emb = torch.nn.Embedding(data["company"].num_nodes, hidden_channels)
        if list_abl[9] == 1:
            self.concept_emb = torch.nn.Embedding(data["concept"].num_nodes, hidden_channels)

        # Instantiate homogeneous GNN:
        self.gnn = GNN(hidden_channels)
        # Convert GNN model into a heterogeneous variant:
        self.gnn = to_hetero(self.gnn, metadata=data.metadata())
        self.classifier = Classifier()
    def forward(self, data: HeteroData) -> Tensor:
        list_abl = self.list_abl
        x_dict = {
        }
        # mandatory
        x_dict["candidate"] = self.user_lin(data["candidate"].x) + self.user_emb(data["candidate"].node_id)
        x_dict["job"] =  self.job_lin(data["job"].x) + self.job_emb(data["job"].node_id)

        if list_abl[0] == 1:
            x_dict["skill"] = self.skill_emb(data["skill"].node_id)
        if list_abl[1] == 1:
            x_dict["contract"] = self.contract_emb(data["contract"].node_id)
        if list_abl[2] == 1:
            x_dict["origin"] = self.origin_emb(data["origin"].node_id)
        if list_abl[3] == 1:
            x_dict["experience"] = self.experience_emb(data["experience"].node_id)
        if list_abl[4] == 1:
            x_dict["salary"] = self.salary_emb(data["salary"].node_id)
        if list_abl[5] == 1:
            x_dict["zip"] = self.zip_emb(data["zip"].node_id)
        if list_abl[6] == 1:
            x_dict["category"] = self.category_emb(data["category"].node_id)
        if list_abl[7] == 1:
            x_dict["recruiter"] = self.recruiter_emb(data["recruiter"].node_id)
        if list_abl[8] == 1:
            x_dict["company"] = self.company_emb(data["company"].node_id)
        if list_abl[9] == 1:
            x_dict["concept"] = self.concept_emb(data["concept"].node_id)

        # `x_dict` holds feature matrices of all node types
        # `edge_index_dict` holds all edge indices of all edge types
        x_dict = self.gnn(x_dict, data.edge_index_dict)
        pred = self.classifier(
            x_dict["candidate"],
            x_dict["job"],
            data["candidate", "is_shortlist", "job"].edge_label_index,
        )
        return pred
