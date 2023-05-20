import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_utils



class Model(nn.Module):
    def __init__(self,wandb_logger, units, stack_cnt, time_step,
                 multi_layer, horizon=1,
                 semantic_embs=None, semantic_embs_dim=168,
                 dropout_rate=0.5, leaky_rate=0.2,
                 device='cpu', gru_dim=128, num_heads=8,
                 dist_adj=None):
        super(Model, self).__init__()
        self.gru_dim = gru_dim
        self.wandb_logger=wandb_logger
        # self.stack_cnt = stack_cnt
        self.unit = units
        self.alpha = leaky_rate
        self.drop_out = nn.Dropout(p=dropout_rate)
        self.time_step = time_step
        self.horizon = horizon
        self.seq1 = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        self.layer_norm = nn.LayerNorm(self.gru_dim)
        self.time_attention = Attention(self.gru_dim, self.gru_dim)
        self.mhead_attention = nn.MultiheadAttention(self.gru_dim, num_heads, dropout_rate, device=device, batch_first=True)
        
        self.GRU_cells = nn.ModuleList(
            nn.GRU(128, gru_dim, batch_first=True) for _ in range(self.unit)
        )
        
        # self.fc_ta = nn.Linear(gru_dim, self.time_step) #TODO remove this
        self.fc_ta = nn.Linear(gru_dim, 128)
        
        for i, cell in enumerate(self.GRU_cells):
            cell.flatten_parameters()

        
        self.semantic_embs = torch.from_numpy(semantic_embs).to(device).float()
        
        self.linear_semantic_embs = nn.Linear(self.semantic_embs.shape[1], semantic_embs_dim) 
                
       
        # self.node_feature_dim = time_step + semantic_embs_dim
        self.node_feature_dim = 128 + semantic_embs_dim
        
    


        self.convs = nn.ModuleList()

        self.conv_hidden_dim = 32
        self.conv_layers_num = 3

        self.convs.append(pyg_nn.GCNConv(self.node_feature_dim, self.conv_hidden_dim)) 

        for l in range(self.conv_layers_num-1):
            self.convs.append(pyg_nn.GCNConv(self.conv_hidden_dim, self.conv_hidden_dim))


        self.fc = nn.Sequential(
            nn.Linear(int(self.conv_hidden_dim + self.node_feature_dim), int(self.node_feature_dim)),
            nn.ReLU(),
            nn.Linear(int(self.node_feature_dim), self.horizon),
            # nn.Tanh(), #TODO: Delete this line
        )

        if dist_adj is None:
            self.dist_adj = torch.ones((self.unit, self.unit)).to(device).float()
        else:
            self.dist_adj = torch.from_numpy(dist_adj).to(device).float()

        
        # self.attention_thres = AttentionThreshold(self.unit)
        # self.thres = nn.Parameter(torch.tensor(0.15), requires_grad=True)
        
        # self.GLUs = nn.Sequential(
        #    GLU(self.node_feature_dim, self.node_feature_dim * 4),
        #    GLU(self.node_feature_dim*4, self.node_feature_dim * 4),
        #    GLU(self.node_feature_dim*4, self.node_feature_dim * 4),
        #    GLU(self.node_feature_dim*4, self.node_feature_dim)
        # )
        
        self.att_alpha = nn.Parameter(torch.tensor(0.5), requires_grad=True)

        # initialize weights using xavier
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.GRU):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_normal_(param)
                    elif 'bias' in name:
                        param.data.zero_()
            elif isinstance(m, nn.LayerNorm):
                m.bias.data.zero_()
                m.weight.data.fill_(1.0)
            elif isinstance(m, nn.Dropout):
                m.p = dropout_rate
            elif isinstance(m, nn.MultiheadAttention):
                nn.init.xavier_uniform_(m.in_proj_weight)
                nn.init.xavier_uniform_(m.out_proj.weight)
                m.in_proj_bias.data.zero_()
                m.out_proj.bias.data.zero_()  
            elif isinstance(m, GLU):
                nn.init.xavier_uniform_(m.fc_left.weight)
                m.fc_left.bias.data.zero_()     
                nn.init.xavier_uniform_(m.fc_right.weight)  
                m.fc_right.bias.data.zero_()

        self.to(device)

        


    def latent_correlation_layer(self, x):
        batch_size, _, node_cnt = x.shape
        window = x.shape[1]
        new_x = x.permute(2, 0, 1)
        weighted_res = torch.empty(batch_size, node_cnt, self.gru_dim).to(x.get_device())
        for i, cell in enumerate(self.GRU_cells):
            cell.flatten_parameters()
            # TODO STG1
            x_sup = self.seq1(new_x[i].unsqueeze(-1))
            # x_sup = new_x[i].unsqueeze(-1)
            gru_outputs, hid = cell(x_sup)
            hid = hid.squeeze(0)
            gru_outputs = gru_outputs.permute(1, 0, 2).contiguous()
            weights = self.time_attention(hid, gru_outputs)
            updated_weights = weights.unsqueeze(1)
            gru_outputs = gru_outputs.permute(1, 0, 2)
            weighted = torch.bmm(updated_weights, gru_outputs)
            weighted = weighted.squeeze(1)
            weighted_res[:, i, :] = self.layer_norm(weighted + hid)
        _, attention = self.mhead_attention(weighted_res, weighted_res, weighted_res)

        attention = torch.mean(attention, dim=0) #[2000, 2000]
        attention = self.drop_out(attention)
        # attention = self._normalize_attention(attention)

        # attention is temporal attention. Combine it with spatial attention + add self-loops 
        # first use distance matrix as a gate
        ########
        # self.dist_adj = self.dist_adj.to(x.get_device())
        # attention_hat = (self.dist_adj * attention)
        #########
        
        # attention_hat = torch.nn.functional.normalize(attention_hat, p=2, dim=0)
        # for i in range(attention_hat.shape[0]):
        #     attention_hat[i, i] += 1
        return attention, weighted_res # TODO replace with above line



    def forward(self, x, static_features=None):
        attention, weighted_res = self.latent_correlation_layer(x) # TODO replace with above line

        # attention_mask = self.attention_thres(attention)
        # attention[~attention_mask] = 0


        
        
        weighted_res = self.fc_ta(weighted_res)
        weighted_res = F.relu(weighted_res)
        # weighted_res = F.relu(weighted_res)
        
        # X = weighted_res.unsqueeze(1).permute(0, 1, 2, 3).contiguous() #TODO replace with above line
        X = weighted_res.permute(0, 1, 2).contiguous()
        #TODO: should I add the static features (e.g., POI category vec) here??
        #TODO: appending static features vec to X
        if self.semantic_embs is not None:
            transformed_embeds = self.linear_semantic_embs(self.semantic_embs.to(x.get_device()))
            # transformed_embeds = self.semantic_embs.to(x.get_device())
            transformed_embeds = transformed_embeds.unsqueeze(0).repeat(X.shape[0], 1, 1)
            X = torch.cat((X, transformed_embeds), dim=2)
            
        embed_att = self.get_embed_att_mat_cosine(transformed_embeds)
        self.dist_adj = self.dist_adj.to(x.get_device())
        # attention = (((self.dist_adj + embed_att)/2) * attention)
        # attention = ((self.att_alpha*self.dist_adj) + (1-self.att_alpha)*embed_att) * attention
        attention = embed_att * attention
        # attention = ((self.dist_adj + embed_att) * attention)
        
        attention_mask = self.case_amplf_mask(attention)
        
        # attention_mask = self.attention_thres(attention)
        attention[~attention_mask] = 0
        
        edge_indices, edge_attrs = pyg_utils.dense_to_sparse(attention)
        
        # X = self.GLUs(X)

        
        X_gnn = self.convs[0](X, edge_indices, edge_attrs)
        X_gnn = F.relu(X_gnn)
        for stack_i in range(1, self.conv_layers_num):
            X_gnn = self.convs[stack_i](X_gnn, edge_indices, edge_attrs)
            X_gnn = F.relu(X_gnn)
        X_hat = torch.cat((X, X_gnn), dim=2)
        forecast = self.fc(X_hat)
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
    
    
    def _normalize_attention(self, attention):
        # Normalize each row of the attention matrix
        max_scores, _ = torch.max(attention, dim=1, keepdim=True)
        norm_scores = attention / max_scores
        return norm_scores
    
    
    def case_amplf_mask(self, attention, p=2.5, threshold=0.15):
        '''
        This function computes the case amplification mask for a 2D attention tensor 
        with the given amplification factor p.

        Parameters:
            - attention (torch.Tensor): A 2D attention tensor of shape [n, n].
            - p (float): The case amplification factor (default: 2.5).
            - threshold (float): The threshold for the mask (default: 0.05).

        Returns:
            - mask (torch.Tensor): A 2D binary mask of the same size as `attention`,
              where 0s denote noisy elements and 1s denote clean elements.
        '''
        # Compute the maximum value in the attention tensor
        max_val, _ = torch.max(attention.detach(), dim=1, keepdim=True)

        # Compute the mask as per the case amplification formula
        mask = (attention.detach() / max_val) ** p

        # Turn the mask into a binary matrix, where anything below threshold will be considered as zero
        mask = torch.where(mask > threshold, torch.tensor(1).to(attention.device), torch.tensor(0).to(attention.device))
        return mask
        
        
    
    
    def get_embed_att_mat_cosine(self, embed_tensor):
        # embe_vecs: the tensor with shape (batch, POI_NUM, embed_dim)
        # Compute the dot product between all pairs of embeddings
        similarity_matrix = torch.bmm(embed_tensor, embed_tensor.transpose(1, 2))

        # Compute the magnitudes of each embedding vector
        magnitude = torch.norm(embed_tensor, p=2, dim=2, keepdim=True)

        # Normalize the dot product by the magnitudes
        normalized_similarity_matrix = similarity_matrix / (magnitude * magnitude.transpose(1, 2))

        # Apply a softmax function to obtain a probability distribution
        # similarity_matrix_prob = F.softmax(normalized_similarity_matrix, dim=2).mean(dim=0)
        
        return normalized_similarity_matrix.mean(dim=0).abs()
        # return similarity_matrix_prob
    
    
    def get_embed_att_mat_euc(self, embed_tensor):
        # Compute the Euclidean distance between all pairs of embeddings
        similarity_matrix = torch.cdist(embed_tensor[0], embed_tensor[0])

        # Convert the distances to similarities using a Gaussian kernel
        sigma = 1.0  # adjust this parameter to control the width of the kernel
        similarity_matrix = torch.exp(-similarity_matrix.pow(2) / (2 * sigma**2))

        # Normalize the similarity matrix by row
        # row_sum = similarity_matrix.sum(dim=1, keepdim=True)
        # similarity_matrix_prob = similarity_matrix / row_sum
        
        # return similarity_matrix
        return similarity_matrix


class AttentionThreshold(nn.Module):
    def __init__(self, input_size, threshold_value=0.0012):
        super().__init__()
        self.threshold = nn.Parameter(torch.tensor(threshold_value), requires_grad=True)

    def forward(self, attention_matrix):
        # Apply the threshold to the attention matrix to get a binary mask
        mask = attention_matrix > self.threshold
        return mask
        
        
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
    
    
class GLU(nn.Module):
    def __init__(self, input_channel, output_channel):
        super(GLU, self).__init__()
        self.fc_left = nn.Linear(input_channel, output_channel)
        self.fc_right = nn.Linear(input_channel, output_channel)

    def forward(self, x):
        return torch.mul(self.fc_left(x), torch.sigmoid(self.fc_right(x)))
