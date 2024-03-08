
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer
from transformers.activations import ACT2FN
import os

class GraphAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
            self,
            embed_dim: int,
            num_heads: int,
            dropout: float = 0.0,
            is_decoder: bool = False,
            bias: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert (
                self.head_dim * num_heads == self.embed_dim
        ), f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {num_heads})."
        self.scaling = self.head_dim ** -0.5
        self.is_decoder = is_decoder

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
            self,
            hidden_states: torch.Tensor,
            key_value_states=None,
            past_key_value=None,
            attention_mask=None,
            output_attentions: bool = False,
            extra_attn=None,
            only_attn=False,
    ):
        """Input shape: Batch x Time x Channel"""

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None
        bsz, tgt_len, embed_dim = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scaling
        # get key, value proj
        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            # cross_attentions
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:
            # reuse k, v, self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
            # self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_states, value_states)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
        if extra_attn is not None:
            attn_weights += extra_attn

        assert attn_weights.size() == (
            bsz * self.num_heads,
            tgt_len,
            src_len,
        ), f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is {attn_weights.size()}"

        if attention_mask is not None:
            assert attention_mask.size() == (
                bsz,
                1,
                tgt_len,
                src_len,
            ), f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = F.softmax(attn_weights, dim=-1)

        if output_attentions:
            # this operation is a bit akward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        if only_attn:
            return attn_weights_reshaped

        attn_probs = F.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)

        assert attn_output.size() == (
            bsz * self.num_heads,
            tgt_len,
            self.head_dim,
        ), f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is {attn_output.size()}"

        attn_output = (
            attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
                .transpose(1, 2)
                .reshape(bsz, tgt_len, embed_dim)
        )

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped, past_key_value

#from torch_geometric.nn import GCNConv, GATConv
class GraphPropagationAttention(nn.Module):
    def __init__(self, node_dim, edge_dim, num_heads=12, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = node_dim // num_heads
        self.scale = head_dim ** -0.5
        
        self.qkv = nn.Linear(node_dim, node_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(node_dim, node_dim)
        
        self.reduce = nn.Conv2d(edge_dim, num_heads, kernel_size=1)
        self.expand = nn.Conv2d(num_heads, edge_dim, kernel_size=1)
        if edge_dim != node_dim:
            self.fc = nn.Linear(edge_dim, node_dim)
        else:
            self.fc = nn.Identity()
        self.proj_drop = nn.Dropout(proj_drop)
    
    def forward(self, node_embeds, edge_embeds):
        # node-to-node propagation
        #padding_mask = torch.ones(1, node_embeds.size(1), device=node_embeds.device)
        #padding_mask = (padding_mask > 0).type(torch.bool)
        B, N, C = node_embeds.shape
        qkv = self.qkv(node_embeds).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)


        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale # [B, n_head, 1+N, 1+N]
        #print(edge_embeds.shape)
        attn_bias = self.reduce(edge_embeds) # [B, C, 1+N, 1+N] -> [B, n_head, 1+N, 1+N]
        attn = attn + attn_bias # [B, n_head, 1+N, 1+N]
        residual = attn

        #attn = attn.masked_fill(padding_mask, float("-inf"))
        attn = attn.softmax(dim=-1) # [B, C, N, N]
        attn = self.attn_drop(attn)
        node_embeds = (attn @ v).transpose(1, 2).reshape(B, N, C)

        # node-to-edge propagation
        edge_embeds = self.expand(attn + residual)  # [B, n_head, 1+N, 1+N] -> [B, C, 1+N, 1+N]

        # edge-to-node propagation
        #w = edge_embeds.masked_fill(padding_mask, float("-inf"))
        w=edge_embeds
        w = w.softmax(dim=-1)
        w = (w * edge_embeds).sum(-1).transpose(-1, -2)
        node_embeds = node_embeds + self.fc(w)
        node_embeds = self.proj(node_embeds)
        node_embeds = self.proj_drop(node_embeds)

        return node_embeds, edge_embeds

class GraphLayer(nn.Module):
    def __init__(self, config, graph_type,edge_dim,label_refiner=0,):
        super(GraphLayer, self).__init__()
        self.config = config

        self.graph_type = graph_type
        self.label_refiner=label_refiner
        if self.graph_type == 'graphormer':
            self.graph = GraphAttention(config.hidden_size, config.num_attention_heads,
                                        config.attention_probs_dropout_prob)
        elif self.graph_type == 'GCN':
            pass
            self.graph = GCNConv(config.hidden_size, config.hidden_size)
        elif self.graph_type == 'GAT':
            pass
            self.graph = GATConv(config.hidden_size, config.hidden_size, 1)
        elif self.graph_type=='GPA':
            self.graph=GraphPropagationAttention(node_dim=config.hidden_size, edge_dim=edge_dim, num_heads=config.num_attention_heads, qkv_bias=False, attn_drop= config.attention_probs_dropout_prob, proj_drop= config.attention_probs_dropout_prob)

        self.layer_norm = nn.LayerNorm(config.hidden_size)

        self.dropout = config.attention_probs_dropout_prob
        self.activation_fn = ACT2FN[config.hidden_act]
        self.activation_dropout = config.hidden_dropout_prob
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.final_layer_norm = nn.LayerNorm(config.hidden_size)
    
    
    def forward(self, label_emb, extra_attn):
        residual = label_emb
        if self.graph_type == 'graphormer':
            label_emb, attn_weights, _ = self.graph(
                hidden_states=label_emb, attention_mask=None, output_attentions=False,
                extra_attn=extra_attn,
            )
            label_emb = nn.functional.dropout(label_emb, p=self.dropout, training=self.training)
            label_emb = residual + label_emb
            label_emb = self.layer_norm(label_emb)
            if self.label_refiner:
            
                residual = label_emb
                label_emb = self.activation_fn(self.fc1(label_emb))
                label_emb = nn.functional.dropout(label_emb, p=self.activation_dropout, training=self.training)
                label_emb = self.fc2(label_emb)
                label_emb = nn.functional.dropout(label_emb, p=self.dropout, training=self.training)
                label_emb = residual + label_emb
                label_emb = self.final_layer_norm(label_emb)
        elif self.graph_type == 'GCN' or self.graph_type == 'GAT':
            label_emb = self.graph(label_emb.squeeze(0), edge_index=extra_attn)
            label_emb = nn.functional.dropout(label_emb, p=self.dropout, training=self.training)
            label_emb = residual + label_emb
            label_emb = self.layer_norm(label_emb)
        elif self.graph_type=='GPA':
            label_emb,_=self.graph(label_emb,extra_attn,)
            if self.label_refiner:
            
                residual = label_emb
                label_emb = self.activation_fn(self.fc1(label_emb))
                label_emb = nn.functional.dropout(label_emb, p=self.activation_dropout, training=self.training)
                label_emb = self.fc2(label_emb)
                label_emb = nn.functional.dropout(label_emb, p=self.dropout, training=self.training)
                label_emb = residual + label_emb
                label_emb = self.final_layer_norm(label_emb)
            
        else:
            raise NotImplementedError
        return label_emb




class GraphEncoder(nn.Module):
    def __init__(self, config, graph_type='GAT', edge_dim=40,layer=1, data_path=None, tokenizer='bert-base-uncased',label_refiner=0):
        super(GraphEncoder, self).__init__()
        self.config = config
        self.label_dict = torch.load(os.path.join(data_path, 'bert_value_dict.pt'))
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)

        self.label_dict = {i: self.tokenizer.decode(v) for i, v in self.label_dict.items()}
        self.label_name = []
        for i in range(len(self.label_dict)):
            self.label_name.append(self.label_dict[i])
        self.label_name = self.tokenizer(self.label_name, padding='longest')['input_ids']
        self.label_name = nn.Parameter(torch.tensor(self.label_name, dtype=torch.long), requires_grad=False)
        self.hir_layers = nn.ModuleList([GraphLayer(config,graph_type=graph_type,edge_dim=edge_dim,label_refiner=label_refiner) for i in range(layer)])

        self.label_num = len(self.label_name)

        self.graph_type = graph_type
  


        label_hier = torch.load(os.path.join(data_path, 'slot.pt'))
        path_dict = {}
        num_class = 0
        for s in label_hier:
            for v in label_hier[s]:
                path_dict[v] = s
                if num_class < v:
                    num_class = v
        if self.graph_type == 'graphormer' or self.graph_type == 'GPA' :
            num_class += 1
            for i in range(num_class):
                if i not in path_dict:
                    path_dict[i] = i
            self.inverse_label_list = {}

            def get_root(path_dict, n):
                ret = []
                while path_dict[n] != n:
                    ret.append(n)
                    n = path_dict[n]
                ret.append(n)
                return ret

            for i in range(num_class):
                self.inverse_label_list.update({i: get_root(path_dict, i) + [-1]})
            label_range = torch.arange(len(self.inverse_label_list))
            self.label_id = label_range
            node_list = {}

            def get_distance(node1, node2):
                p = 0
                q = 0
                node_list[(node1, node2)] = a = []
                node1 = self.inverse_label_list[node1]
                node2 = self.inverse_label_list[node2]
                while p < len(node1) and q < len(node2):
                    if node1[p] > node2[q]:
                        a.append(node1[p])
                        p += 1

                    elif node1[p] < node2[q]:
                        a.append(node2[q])
                        q += 1

                    else:
                        break
                return p + q

            self.distance_mat = self.label_id.reshape(1, -1).repeat(self.label_id.size(0), 1)
            hier_mat_t = self.label_id.reshape(-1, 1).repeat(1, self.label_id.size(0))
            self.distance_mat.map_(hier_mat_t, get_distance)
            self.distance_mat = self.distance_mat.view(1, -1)
            self.edge_mat = torch.zeros(len(self.inverse_label_list), len(self.inverse_label_list), 15,
                                        dtype=torch.long)
            for i in range(len(self.inverse_label_list)):
                for j in range(len(self.inverse_label_list)):
                    edge_list = node_list[(i, j)]
                    self.edge_mat[i, j, :len(edge_list)] = torch.tensor(edge_list) + 1
            self.edge_mat = self.edge_mat.view(-1, self.edge_mat.size(-1))

            self.id_embedding = nn.Embedding(len(self.inverse_label_list) + 1, config.hidden_size,
                                              len(self.inverse_label_list))
            if self.graph_type=='GPA':
            
                self.distance_embedding = nn.Embedding(20, edge_dim, 0)
                self.edge_embedding = nn.Embedding(len(self.inverse_label_list) + 1, edge_dim, 0)
            else:
                self.distance_embedding = nn.Embedding(20, 1, 0)
                self.edge_embedding = nn.Embedding(len(self.inverse_label_list) + 1, 1, 0)
            self.label_id = nn.Parameter(self.label_id, requires_grad=False)
            self.edge_mat = nn.Parameter(self.edge_mat, requires_grad=False)
            self.distance_mat = nn.Parameter(self.distance_mat, requires_grad=False)
        self.edge_list = [[v, i] for v, i in path_dict.items()]
        self.edge_list += [[i, v] for v, i in path_dict.items()]
        self.edge_list = nn.Parameter(torch.tensor(self.edge_list).transpose(0, 1), requires_grad=False) # 2,282...will be given i n edge_index of GCN/GAT
        #The edge_index tensor is a 2D tensor of shape (2, num_edges), where num_edges is the total number of edges in the graph.
        #Each column of the edge_index tensor represents an edge, and the two rows contain the indices of the source and target nodes of the 

    def forward(self, embeddings):
        label_mask = self.label_name != self.tokenizer.pad_token_id
        # full name
        label_emb = embeddings(self.label_name)
        label_emb = (label_emb * label_mask.unsqueeze(-1)).sum(dim=1) / label_mask.sum(dim=1).unsqueeze(-1)
        label_emb = label_emb.unsqueeze(0)

        label_attn_mask = torch.ones(1, label_emb.size(1), device=label_emb.device)

        extra_attn = None

        self_attn_mask = (label_attn_mask * 1.).t().mm(label_attn_mask * 1.).unsqueeze(0).unsqueeze(0)
        #print(self_attn_mask)
        expand_size = label_emb.size(-2) // self.label_name.size(0)
        if self.graph_type=='graphormer':
            
            label_emb += self.id_embedding(self.label_id[:, None].expand(-1, expand_size)).view(1, -1,
                                                                                                self.config.hidden_size)
            extra_attn = self.distance_embedding(self.distance_mat) + self.edge_embedding(self.edge_mat).sum(
                dim=1) / (
                                  self.distance_mat.view(-1, 1) + 1e-8)
            extra_attn = extra_attn.view(self.label_num, 1, self.label_num, 1).expand(-1, expand_size, -1,
                                                                                      expand_size)
            extra_attn = extra_attn.reshape(self.label_num * expand_size, -1)
        elif self.graph_type=='GPA':
            label_emb += self.id_embedding(self.label_id[:, None].expand(-1, expand_size)).view(1, -1,
                                                                                                self.config.hidden_size)
            extra_attn = self.distance_embedding(self.distance_mat) + self.edge_embedding(self.edge_mat).sum(
                dim=1) / (
                                  self.distance_mat.view(-1, 1) + 1e-8) # extra_attn is edge_emb
            #rint(edge_emb.shape)
            
            extra_attn= extra_attn.reshape(1,-1,label_emb.size(1),label_emb.size(1))
            #print(edge_emb.shape)

        elif self.graph_type== 'GCN' or self.graph_type == 'GAT':
            extra_attn = self.edge_list
        for hir_layer in self.hir_layers:
            label_emb = hir_layer(label_emb, extra_attn)

        
        return label_emb.squeeze(0)
        
