import torch
import torch.nn as nn
import torch.nn.functional as F


class GLU(nn.Module):
    def __init__(self, input_channel, output_channel):
        super(GLU, self).__init__()
        self.linear_left = nn.Linear(input_channel, output_channel)
        self.linear_right = nn.Linear(input_channel, output_channel)

    def forward(self, x):
        return torch.mul(self.linear_left(x), torch.sigmoid(self.linear_right(x)))


class StockBlockLayer(nn.Module):
    def __init__(self, time_step, unit, multi_layer, stack_cnt=0):
        super(StockBlockLayer, self).__init__()
        self.time_step = time_step
        self.unit = unit
        self.stack_cnt = stack_cnt
        self.multi = multi_layer
        self.weight = nn.Parameter(
            torch.Tensor(1, 3 + 1, 1, self.time_step * self.multi,
                         self.multi * self.time_step))  # [K+1, 1, in_c, out_c]
        nn.init.xavier_normal_(self.weight)
        self.forecast = nn.Linear(self.time_step * self.multi, self.time_step * self.multi)
        self.forecast_result = nn.Linear(self.time_step * self.multi, self.time_step)
        if self.stack_cnt == 0:
            self.backcast = nn.Linear(self.time_step * self.multi, self.time_step)
        self.backcast_short_cut = nn.Linear(self.time_step, self.time_step)
        self.relu = nn.ReLU()
        self.GLUs = nn.ModuleList()
        self.output_channel = 4 * self.multi
        for i in range(3):
            if i == 0:
                self.GLUs.append(GLU(self.time_step * 4, self.time_step * self.output_channel))
                self.GLUs.append(GLU(self.time_step * 4, self.time_step * self.output_channel))
            elif i == 1:
                self.GLUs.append(GLU(self.time_step * self.output_channel, self.time_step * self.output_channel))
                self.GLUs.append(GLU(self.time_step * self.output_channel, self.time_step * self.output_channel))
            else:
                self.GLUs.append(GLU(self.time_step * self.output_channel, self.time_step * self.output_channel))
                self.GLUs.append(GLU(self.time_step * self.output_channel, self.time_step * self.output_channel))

    def spe_seq_cell(self, input):

        ### input: 32, 4, 1, 2000, 24
        batch_size, k, input_channel, node_cnt, time_step = input.size()
        input = input.view(batch_size, -1, node_cnt, time_step) ## 32, 4, 2000, 24
        #print(f"input{input.shape}")

        # ffted = torch.rfft(input, 1, onesided=False) ### older VERSION 32, 4, 2000, 24, 2
        # real = ffted[..., 0].permute(0, 2, 1, 3).contiguous().reshape(batch_size, node_cnt, -1)### 32, 2000, 96
        # img = ffted[..., 1].permute(0, 2, 1, 3).contiguous().reshape(batch_size, node_cnt, -1)


        ffted  = torch.fft.fft(input)
        real = ffted.real.permute(0, 2, 1, 3).contiguous().reshape(batch_size, node_cnt, -1)
        img = ffted.imag.permute(0, 2, 1, 3).contiguous().reshape(batch_size, node_cnt, -1)


        for i in range(3):
            real = self.GLUs[i * 2](real)
            img = self.GLUs[2 * i + 1](img)
        real = real.reshape(batch_size, node_cnt, 4, -1).permute(0, 2, 1, 3).contiguous() ##32, 4, 2000, 120
        img = img.reshape(batch_size, node_cnt, 4, -1).permute(0, 2, 1, 3).contiguous()

        # print(f"real{real.shape}")
        time_step_as_inner = torch.complex(real, img)
        iffted = torch.fft.ifft(time_step_as_inner)
        # print(f"iffted{iffted[0]}")

        # time_step_as_inner = torch.cat([real.unsqueeze(-1), img.unsqueeze(-1)], dim=-1) ### ordler 32, 4, 2000, 120, 2
        # iffted = torch.irfft(time_step_as_inner, 1, onesided=False) ###iffted =  32, 4, 2000, 120


        return iffted.real

    def forward(self, x, mul_L):
        mul_L = mul_L.unsqueeze(1)
        x = x.unsqueeze(1)
        gfted = torch.matmul(mul_L, x) ## 32, 4, 1, 2000, 24
        #print(f"gfted {gfted.shape}")
        gconv_input = self.spe_seq_cell(gfted).unsqueeze(2) ### [32, 4, 1, 2000, 120]
        #print(f"gconv {gconv_input.shape}")
        igfted = torch.matmul(gconv_input, self.weight)
        #print(f"igfted  {igfted .shape}")
        igfted = torch.sum(igfted, dim=1) #### [32, 4, 1, 2000, 120]
        forecast_source = torch.sigmoid(self.forecast(igfted).squeeze(1))
        forecast = self.forecast_result(forecast_source)
        if self.stack_cnt == 0:
            backcast_short = self.backcast_short_cut(x).squeeze(1)
            backcast_source = torch.sigmoid(self.backcast(igfted) - backcast_short)
        else:
            backcast_source = None
        return forecast, backcast_source


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
        
        
        
        self.multi_layer = multi_layer
        self.stock_block = nn.ModuleList()
        
        # TODO BLOCK STARTS: Is this alright for adding static features??
        
        # self.stock_block.extend(
        #     [StockBlockLayer(self.time_step, self.unit, self.multi_layer, stack_cnt=i) for i in range(self.stack_cnt)])
        # self.fc = nn.Sequential(
        #     nn.Linear(int(self.time_step), int(self.time_step)),
        #     nn.LeakyReLU(),
        #     nn.Linear(int(self.time_step), self.horizon),
        #     # nn.Tanh(), #TODO: Delete this line
        # )
        
        self.node_feature_dim = time_step + 0
        
        self.stock_block.extend(
            [StockBlockLayer(self.node_feature_dim, self.unit, self.multi_layer, stack_cnt=i) for i in range(self.stack_cnt)])
        self.fc = nn.Sequential(
            nn.Linear(int(self.node_feature_dim), int(self.node_feature_dim)),
            nn.LeakyReLU(),
            nn.Linear(int(self.node_feature_dim), self.horizon),
            # nn.Tanh(), #TODO: Delete this line
        )
        
        # TODO BLOCK ENDS: Is this alright for adding static features??
        
        
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.to(device)

    def get_laplacian(self, graph, normalize):
        """
        return the laplacian of the graph.
        :param graph: the graph structure without self loop, [N, N].
        :param normalize: whether to used the normalized laplacian.
        :return: graph laplacian.
        """
        if normalize:
            D = torch.diag(torch.sum(graph, dim=-1) ** (-1 / 2))
            L = torch.eye(graph.size(0), device=graph.device, dtype=graph.dtype) - torch.mm(torch.mm(D, graph), D)
        else:
            D = torch.diag(torch.sum(graph, dim=-1))
            L = D - graph
        return L

    def cheb_polynomial(self, laplacian):
        """
        Compute the Chebyshev Polynomial, according to the graph laplacian.
        :param laplacian: the graph laplacian, [N, N].
        :return: the multi order Chebyshev laplacian, [K, N, N].
        """
        N = laplacian.size(0)  # [N, N]
        laplacian = laplacian.unsqueeze(0)
        # first_laplacian = torch.zeros([1, N, N], device=laplacian.device, dtype=torch.float)
        first_laplacian = torch.eye(N, device=laplacian.device, dtype=torch.float).unsqueeze(0) #TODO remove this
        second_laplacian = laplacian
        # third_laplacian = (2 * torch.matmul(laplacian, second_laplacian)) - first_laplacian
        third_laplacian = (2 * laplacian * second_laplacian) - first_laplacian #TODO remove this
        # forth_laplacian = 2 * torch.matmul(laplacian, third_laplacian) - second_laplacian
        forth_laplacian = 2 * laplacian * third_laplacian - second_laplacian #TODO remove this
        multi_order_laplacian = torch.cat([first_laplacian, second_laplacian, third_laplacian, forth_laplacian], dim=0)
        return multi_order_laplacian

    def latent_correlation_layer(self, x):
        # input, _ = self.GRU(x.permute(2, 0, 1).contiguous()) ## input = [2000, 32, 2000] TODO: Uncomment
        batch_size, _, node_cnt = x.shape
        window = x.shape[1]
        # self.GRU.flatten_parameters()
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
            # weighted_res[:, i, :] = weighted
        # new_x = x.permute(2, 0, 1).reshape(-1, window).unsqueeze(-1).contiguous()
        # new_x = self.seq1(new_x)
        # gru_outputs, hid = self.GRU(new_x) ############## TODO: BE DELETED # [8000, 24, 1]
        # hid = hid.squeeze(0) #TODO: Delete this line
        # gru_outputs = gru_outputs.permute(1, 0, 2).contiguous() #TODO: Delete this line
        # # input = input.permute(1, 0, 2).contiguous() #([32, 2000, 2000] TODO: Uncomment this
        # # TODO: REMOVE THIS TIME ATTENTION
        # weights = self.time_attention(hid, gru_outputs)
        # eps = 1e-3
        # T = 1e-2
        # zeros = torch.zeros_like(weights)
        # # updated_weights = torch.where(weights < 1e-3, zeros, weights) # TODO: Remove this
        # # updated_weights = torch.sigmoid((weights - self.param_eps) * self.param_t * (weights - zeros) + zeros) # TODO remove
        # updated_weights = weights # TODO REMOVe
        # # TODO REMOVE THIS
        # updated_weights = updated_weights.unsqueeze(1)
        # gru_outputs = gru_outputs.permute(1, 0, 2)
        # weighted = torch.bmm(updated_weights, gru_outputs)
        # weighted = weighted.squeeze(1)
        # weighted = weighted.reshape(x.shape[0], x.shape[2], -1).contiguous()
        # hid = hid.reshape(x.shape[0], x.shape[2], -1).contiguous()
        # normed_gru_out = self.layer_norm(weighted + hid)
        # # TODO: Remove attention
        # _, attention = self.self_attention(normed_gru_out, normed_gru_out, normed_gru_out) #TODO: remove this
        _, attention = self.self_attention(weighted_res, weighted_res, weighted_res)

        # attention = self.self_graph_attention(input) ## [32, 2000, 2000] TODO: Uncomment this
        attention = torch.mean(attention, dim=0) #[2000, 2000]
        zeros = torch.zeros_like(attention)
        # new_attention = torch.where(attention < 1e-3, zeros, attention) # TODO: Remove
        new_attention = attention # TODO REMOVE
        # new_attention = torch.sigmoid((attention - self.param_eps) * self.param_t * (attention - zeros) + zeros) # TODO: Remove

        degree = torch.sum(new_attention, dim=1) # 2000
        # laplacian is sym or not
        new_attention = 0.5 * (new_attention + new_attention.T)  # TODO: Uncomment this
        degree_l = torch.diag(degree)
        diagonal_degree_hat = torch.diag(1 / (torch.sqrt(degree) + 1e-7))
        laplacian = torch.matmul(diagonal_degree_hat,
                                 torch.matmul(degree_l - new_attention, diagonal_degree_hat))
        mul_L = self.cheb_polynomial(laplacian) ##  [4, 2000, 2000] 4 ==k
        #print(f" mul_L {mul_L.shape}")
        # return mul_L, new_attention
        return mul_L, new_attention, weighted_res # TODO replace with above line

    def self_graph_attention(self, input):
        input = input.permute(0, 2, 1).contiguous()
        bat, N, fea = input.size()
        key = torch.matmul(input, self.weight_key) ### weight_key = [2000, 1]
        #print(f"weight  {self.weight_key.shape}")
        query = torch.matmul(input, self.weight_query)
        data = key.repeat(1, 1, N).view(bat, N * N, 1) + query.repeat(1, N, 1)
        data = data.squeeze(2)
        data = data.view(bat, N, -1)
        data = self.leakyrelu(data)
        attention = F.softmax(data, dim=2)
        attention = self.dropout(attention)

        
        
        return attention

    def graph_fft(self, input, eigenvectors):
        return torch.matmul(eigenvectors, input)

    def forward(self, x, static_features=None):
        # mul_L, attention = self.latent_correlation_layer(x) ## x: batch, window, node 32, 24, 2000
        mul_L, attention, weighted_res = self.latent_correlation_layer(x) # TODO replace with above line
        
        weighted_res = self.fc_ta(weighted_res)
        
        # X = x.unsqueeze(1).permute(0, 1, 3, 2).contiguous()## X: 32, 1, 2000, 24
        X = weighted_res.unsqueeze(1).permute(0, 1, 2, 3).contiguous() #TODO replace with above line
        #TODO: should I add the static features (e.g., POI category vec) here??
        #TODO: appending static features vec to X

        

        result = []
        for stack_i in range(self.stack_cnt):
            forecast, X = self.stock_block[stack_i](X, mul_L)
            result.append(forecast)
        forecast = result[0] + result[1]
        forecast = self.fc(forecast)
        # forecast = F.relu(forecast + 3) #TODO: delete this
        # forecast = forecast - 3 # TODO delete this
        if forecast.size()[-1] == 1:
            return forecast.unsqueeze(1).squeeze(-1), attention.unsqueeze(0)
        else:
            return forecast.permute(0, 2, 1).contiguous(), attention.unsqueeze(0)
        
    

        
        
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
