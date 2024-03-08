import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
import random
import os

nINF= -100



class TLAloss(torch.nn.Module):
    
    def __init__(self,temp,norm,proj,hsize):
        super(TLAloss, self).__init__()
        self.temp=temp
        self.norm=norm
        self.proj=proj
        self.transform= nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(768,hsize),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hsize, 768),
        )

    def forward(self, text_embeddings, label_embeddings, target_labels):
    
        if self.norm:
            text_embeddings, label_embeddings = F.normalize(text_embeddings, p=2, dim=-1), F.normalize(label_embeddings, p=2, dim=-1)
        
        if self.proj:
            text_embeddings, label_embeddings =self.transform(text_embeddings), self.transform(label_embeddings)



        # Step 1: Calculate similarity between text embeddings and label embeddings

        similarity_matrix = F.cosine_similarity(text_embeddings.unsqueeze(1), label_embeddings.unsqueeze(0), dim=2)

        # Step 2: Identify positive labels for each text sample
        positive_labels = [torch.nonzero(label).view(-1).tolist() for label in target_labels]


        # Step 3 : Find hard negative labels
        hard_negative_labels = []
        for i, labels in enumerate(positive_labels):
            hard_negative_labels_sample = []
            # Find hardest negative labels for each positive label
            negative_similarities = similarity_matrix[i].clone()
            negative_similarities[labels] = nINF  # Set positive labels' similarities to negative infinity

            sorted_indices = torch.argsort(negative_similarities, descending=True)
            hard_negative_labels_sample.extend(sorted_indices[:len(labels)].tolist())
            hard_negative_labels.append(hard_negative_labels_sample)




        # Step 4 : Calculate NT-Xent loss
        loss = 0
        batch_size = text_embeddings.size(0)
        for i in range(batch_size):
            zi = text_embeddings[i]
            pos_indices, neg_index = positive_labels[i], hard_negative_labels[i]

            # Calculate positive alignment scores
            pos_alignment_scores = similarity_matrix[i, pos_indices] / self.temp

            
            # Calculate negative alignment score
            neg_alignment_scores = similarity_matrix[i, neg_index] / self.temp

     
            denom= torch.cat([torch.exp(pos_alignment_scores), torch.exp(neg_alignment_scores)]).sum()
            pos_loss = -torch.log(torch.exp(pos_alignment_scores) /denom) 
            pos_loss=pos_loss.mean()
            loss += pos_loss




        # Average loss over the batch
        loss /= batch_size

        #print(loss)
        return loss

