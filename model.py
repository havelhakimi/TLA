
from transformers import AutoConfig,AutoModel
import torch
from graph import GraphEncoder
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


nINF = -100



class PLM_Graph(nn.Module):
    def __init__(self,config,num_labels,mod_type,graph,graph_type,layer,data_path,bce_wt,dot,teal,gneg,gpos,teal_wt,alpha1,beta1,alpha2,disgrad,label_refiner=1):
        super(PLM_Graph, self).__init__()

        self.bert = AutoModel.from_pretrained(mod_type)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.num_labels=num_labels
        config.num_labels=num_labels
        self.bce_wt=bce_wt
        self.graph=graph
        self.dot=dot
        self.teal=teal
        self.teal_wt=teal_wt
        if self.teal:
            self.teal_loss=APLLoss(gamma_neg=gneg, gamma_pos=gpos,alpha1=alpha1,beta1=beta1,alpha2=alpha2,disable_torch_grad_focal_loss=disgrad)
        if self.graph:
          self.gc1 = GraphEncoder(config, graph_type=graph_type, layer=layer, data_path=data_path,tokenizer=mod_type,label_refiner=label_refiner)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.classifier2 = nn.Linear(num_labels * 768,num_labels)


        
        
    def forward(self, input_ids, attention_mask,labels):
        bert_output = self.bert(input_ids, attention_mask)['last_hidden_state']#[:, 0,:]
        bert_output = self.dropout(bert_output)
        if self.graph:


          label_embed = self.gc1(self.bert.embeddings)
          #label_embed = F.relu(label_embed)
          if self.dot:
            
            dot_product = torch.matmul(bert_output[:,0,:], label_embed.transpose(0,1))
            logits=dot_product

          else:
            attns = torch.matmul(bert_output, label_embed.transpose(0, 1))
            #print(att.shape)
            #print('hii')
            weight_label = F.softmax(attns.transpose(1, 2), dim=-1)
            label_align = torch.matmul(weight_label,bert_output )
            #print(label_align.shape)

            logits=self.classifier2(label_align.view(label_align.shape[0],-1))   
            
        else:
          logits=self.classifier(bert_output[:,0,:])
        

        loss=0
        
        if self.training:
            if labels is not None:
                loss_fct = torch.nn.BCEWithLogitsLoss()
                target = labels.to(torch.float32)
                loss += loss_fct(logits.view(-1, self.num_labels), target)*(self.bce_wt)
            
            if self.teal:
                loss+=(self.teal_loss(logits.view(-1, self.num_labels), target)*self.teal_wt)

        


    
        return {
            'loss': loss,
            'logits': logits,
            #'hidden_states': outputs.hidden_states,
            #'attentions': outputs.attentions,
            #'contrast_logits': contrast_logits,
            }
        

class APLLoss(nn.Module):
    '''Adapted from https://github.com/LUMIA-Group/APL/blob/main/APLloss.py'''
    def __init__(self, gamma_neg=4, gamma_pos=0, clip=0.05, eps=1e-8,alpha1=1,beta1=0,alpha2=2, disable_torch_grad_focal_loss=1):
        super(APLLoss, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip # a hard threshold to de-emphasize gradient of easily clasified negative examples 
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

        # parameters of Taylor expansion polynomials
        self.epsilon_pos = alpha1 
        self.epsilon_neg = beta1 
        self.epsilon_pos_pow = -alpha2 


    def forward(self, x, y):
        """"
        x: input logits with size (batch_size, number of labels).
        y: binarized multi-label targets with size (batch_size, number of labels).
        """
        # Calculating Probabilities
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic Taylor expansion polynomials
        los_pos = y * (torch.log(xs_pos.clamp(min=self.eps)) + self.epsilon_pos * (1 - xs_pos.clamp(min=self.eps)) + self.epsilon_pos_pow * 0.5 * torch.pow(1 - xs_pos.clamp(min=self.eps), 2) )
        los_neg = (1 - y) * (torch.log(xs_neg.clamp(min=self.eps)) + self.epsilon_neg * (xs_neg.clamp(min=self.eps)) )
        loss = los_pos + los_neg

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            loss *= one_sided_w

        return -loss.sum()
