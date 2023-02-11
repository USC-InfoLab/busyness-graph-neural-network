import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_utils



class Model(nn.Module):
    def __init__(self,wandb_logger, units, stack_cnt, time_step,
                 multi_layer, horizon=1,
                 embedding_size_dict=None, embedding_dim_dict=None,
                 dropout_rate=0.5, leaky_rate=0.2,
                 device='cpu'):
        super(Model, self).__init__()
        self.wandb_logger=wandb_logger
        self.unit = units
        self.stack_cnt = stack_cnt
        self.unit = units
        self.alpha = leaky_rate
        self.time_step = time_step
        self.horizon = horizon
        self.weight_key = nn.Parameter(torch.zeros(size=(self.unit, 1)))
        nn.init.xavier_uniform_(self.weight_key.data, gain=1.414)
        self.weight_query = nn.Parameter(torch.zeros(size=(self.unit, 1)))
        nn.init.xavier_uniform_(self.weight_query.data, gain=1.414)
        # TODO: Remove batch_first flag
        # self.GRU = nn.GRU(self.time_step, self.unit) TODO: Uncomment
        self.seq1 = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        # self.GRU = nn.GRU(128, self.unit, batch_first=True)
        self.layer_norm = nn.LayerNorm(self.unit)
        self.param_eps = nn.Parameter(torch.tensor(0.0))
        self.param_t = nn.Parameter(torch.tensor(1.0))
        # TODO: REMOVE time_attention
        self.time_attention = Attention(self.unit, self.unit)
        # TODO: Remove multiheadattention layer
        self.self_attention = nn.MultiheadAttention(self.unit, 5, dropout_rate, device=device, batch_first=True)
        
        self.GRU_cells = nn.ModuleList(
            nn.GRU(128, self.unit, batch_first=True) for _ in range(self.unit)
        )
        
        self.fc_ta = nn.Linear(self.unit, self.time_step) #TODO remove this
        
        for i, cell in enumerate(self.GRU_cells):
            cell.flatten_parameters()
        
        #TODO: added embeddings layers here
        self.embeddings, total_embedding_dim = self._create_embedding_layers(
            embedding_size_dict, 
            embedding_dim_dict,
            device=device)
        
        self.cat_cols = list(embedding_size_dict.keys())
        
        
        self.multi_layer = multi_layer
        
       
        self.node_feature_dim = time_step + total_embedding_dim
        
    

        
        # TODO BLOCK ENDS: Is this alright for adding static features??
        
        
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.to(device)


        self.convs = nn.ModuleList()

        self.conv_hidden_dim = 32
        self.conv_layers_num = 2

        self.convs.append(pyg_nn.GCNConv(self.node_feature_dim, self.conv_hidden_dim)) 

        for l in range(self.conv_layers_num-1):
            self.convs.append(pyg_nn.GCNConv(self.conv_hidden_dim, self.conv_hidden_dim))


        self.fc = nn.Sequential(
            nn.Linear(int(self.conv_hidden_dim), int(self.node_feature_dim)),
            nn.LeakyReLU(),
            nn.Linear(int(self.node_feature_dim), self.horizon),
            # nn.Tanh(), #TODO: Delete this line
        )
        


    def latent_correlation_layer(self, x):
        batch_size, _, node_cnt = x.shape
        window = x.shape[1]
        new_x = x.permute(2, 0, 1)
        weighted_res = torch.empty(batch_size, node_cnt, node_cnt).to(x.get_device())
        for i, cell in enumerate(self.GRU_cells):
            x_sup = self.seq1(new_x[i].unsqueeze(-1))
            gru_outputs, hid = cell(x_sup)
            hid = hid.squeeze(0)
            gru_outputs = gru_outputs.permute(1, 0, 2).contiguous()
            weights = self.time_attention(hid, gru_outputs)
            updated_weights = weights.unsqueeze(1)
            gru_outputs = gru_outputs.permute(1, 0, 2)
            weighted = torch.bmm(updated_weights, gru_outputs)
            weighted = weighted.squeeze(1)
            weighted_res[:, i, :] = self.layer_norm(weighted + hid)
        _, attention = self.self_attention(weighted_res, weighted_res, weighted_res)

        attention = torch.mean(attention, dim=0) #[2000, 2000]
        return attention, weighted_res # TODO replace with above line



    def forward(self, x, static_features=None):
        attention, weighted_res = self.latent_correlation_layer(x) # TODO replace with above line
        edge_indices, edge_attrs = pyg_utils.dense_to_sparse(attention)
        
        
        weighted_res = self.fc_ta(weighted_res)
        # weighted_res = F.relu(weighted_res)
        
        X = weighted_res.unsqueeze(1).permute(0, 1, 2, 3).contiguous() #TODO replace with above line
        #TODO: should I add the static features (e.g., POI category vec) here??
        #TODO: appending static features vec to X
        if static_features is not None:
            # run through all the categorical variables through its
            # own embedding layer and concatenate them together
            cat_outputs = []
            for i, col in enumerate(self.cat_cols):
                embedding = self.embeddings[col]
                cat_output = embedding(static_features[:, i])
                cat_outputs.append(cat_output)
            
            cat_outputs = torch.cat(cat_outputs, dim=2)
            X = torch.cat((X, cat_outputs.unsqueeze(1)), dim=3)
        

        result = []
        for stack_i in range(self.conv_layers_num):
            X = self.convs[stack_i](X, edge_indices, edge_attrs)
            X = F.relu(X)
        forecast = self.fc(X)
        if forecast.size()[-1] == 1:
            return forecast.unsqueeze(1).squeeze(-1), attention.unsqueeze(0)
        else:
            return forecast.squeeze(1).permute(0, 2, 1).contiguous(), attention.unsqueeze(0)
        
    
    def _create_embedding_layers(self, embedding_size_dict, embedding_dim_dict, device):
        """construct the embedding layer, 1 per each categorical variable"""
        total_embedding_dim = 0
        cat_cols = list(embedding_size_dict.keys())
        embeddings = {}
        for col in cat_cols:
            embedding_size = embedding_size_dict[col]
            embedding_dim = embedding_dim_dict[col]
            total_embedding_dim += embedding_dim
            embeddings[col] = nn.Embedding(embedding_size, embedding_dim, device=device)
            
        return nn.ModuleDict(embeddings), total_embedding_dim
        
        
# TODO: REMOVE THIS CLASS
class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()
        
        self.attn = nn.Linear((enc_hid_dim * 2), dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias = False)
        
    def forward(self, hidden, encoder_outputs):
        
        #hidden = [batch size, dec hid dim]
        #encoder_outputs = [src len, batch size, enc hid dim * 2]
        
        batch_size = encoder_outputs.shape[1]
        src_len = encoder_outputs.shape[0]
        
        #repeat decoder hidden state src_len times
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        
        #hidden = [batch size, src len, dec hid dim]
        #encoder_outputs = [batch size, src len, enc hid dim * 2]
        
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim = 2))) 
        
        #energy = [batch size, src len, dec hid dim]

        attention = self.v(energy).squeeze(2)
        
        #attention= [batch size, src len]
        
        return F.softmax(attention, dim=1)
