import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from basic_model import BasicModel


class conv_module(nn.Module):
    def __init__(self):
        super(conv_module, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=2)
        self.batch_norm1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=2)
        self.batch_norm2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=2)
        self.batch_norm3 = nn.BatchNorm2d(32)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, stride=2)
        self.batch_norm4 = nn.BatchNorm2d(32)
        self.relu4 = nn.ReLU()
        # self.fc = nn.Linear(32*4*4, 256)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(self.batch_norm1(x))
        x = self.conv2(x)
        x = self.relu2(self.batch_norm2(x))
        x = self.conv3(x)
        x = self.relu3(self.batch_norm3(x))
        x = self.conv4(x)
        x = self.relu4(self.batch_norm4(x))
        return x.view(-1, 16, 32*4*4)



class mlp_f1(nn.Module):
    def __init__(self):
        super(mlp_f1, self).__init__()
        self.fc1 = nn.Linear(768, 512)
        self.relu1 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 256)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(256, 256)


    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        # x = self.dropout(x)
        x = self.fc3(x)
        return x.view(-1, 256)    

class mlp_f2(nn.Module):
    def __init__(self):
        super(mlp_f2, self).__init__()
        self.fc1 = nn.Linear(512, 512)
        self.relu1 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 256)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(256, 256)


    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        # x = self.dropout(x)
        x = self.fc3(x)
        return x.view(-1, 256)
    
    
class panels_to_embeddings(nn.Module):
    def __init__(self, tag):
        super(panels_to_embeddings, self).__init__()
        self.in_dim = 512
        if tag:
            self.in_dim += 9
        self.fc = nn.Linear(self.in_dim, 256)

    def forward(self, x):
        return self.fc(x.view(-1, self.in_dim))

    
class IWReN(BasicModel):
    def __init__(self, args):
        super(IWReN, self).__init__(args)
        self.conv = conv_module()
        self.mlp_f1 = mlp_f1()
        self.mlp_f2 = mlp_f2()
        self.cos = nn.CosineSimilarity(dim=2)
        self.proj = panels_to_embeddings(args.tag)
        self.optimizer = optim.Adam(self.parameters(), lr=args.lr, betas=(args.beta1, args.beta2), eps=args.epsilon)
        self.meta_beta = args.meta_beta 
        self.use_tag = args.tag
        self.use_cuda = args.cuda
        self.tags = self.tag_panels(args.batch_size)

    def tag_panels(self, batch_size):
        tags = []
        for idx in range(0, 16):
            tag = np.zeros([1, 9], dtype=float)
            if idx < 8:
                tag[:, idx] = 1.0
            else:
                tag[:, 8] = 1.0
            tag = torch.tensor(tag, dtype=torch.float).expand(batch_size, -1).unsqueeze(1)
            if self.use_cuda:
                tag = tag.cuda()
            tags.append(tag)
        tags = torch.cat(tags, dim=1)
        return tags

    def group_panel_embeddings(self, embeddings):
        embeddings = embeddings.view(-1, 16, 256)
        embeddings_seq = torch.chunk(embeddings, 16, dim=1)
        context_pairs = []
        for context_idx1 in range(0, 8):
            for context_idx2 in range(0, 8):
                if not context_idx1 == context_idx2:
                    context_pairs.append(torch.cat((embeddings_seq[context_idx1], embeddings_seq[context_idx2]), dim=2))
        context_pairs = torch.cat(context_pairs, dim=1)
        panel_embeddings_pairs = []
        for answer_idx in range(8, len(embeddings_seq)):
            embeddings_pairs = context_pairs
            for context_idx in range(0, 8):
                # In order
                order = torch.cat((embeddings_seq[answer_idx], embeddings_seq[context_idx]), dim=2)
                reverse = torch.cat((embeddings_seq[context_idx], embeddings_seq[answer_idx]), dim=2)
                choice_pairs = torch.cat((order, reverse), dim=1)
                embeddings_pairs = torch.cat((embeddings_pairs, choice_pairs), dim=1)
            panel_embeddings_pairs.append(embeddings_pairs.unsqueeze(1))
        panel_embeddings_pairs = torch.cat(panel_embeddings_pairs, dim=1)
        return panel_embeddings_pairs.view(-1, 8, 72, 512)

    def group_panel_embeddings_batch(self, embeddings):
        embeddings = embeddings.view(-1, 16, 256)
        context_embeddings = embeddings[:,:8,:]
        choice_embeddings = embeddings[:,8:,:]
        context_embeddings_pairs = torch.cat((context_embeddings.unsqueeze(1).expand(-1, 8, -1, -1), context_embeddings.unsqueeze(2).expand(-1, -1, 8, -1)), dim=3).view(-1, 64, 512)
        
        context_embeddings = context_embeddings.unsqueeze(1).expand(-1, 8, -1, -1)
        choice_embeddings = choice_embeddings.unsqueeze(2).expand(-1, -1, 8, -1)
        choice_context_order = torch.cat((context_embeddings, choice_embeddings), dim=3)
        choice_context_reverse = torch.cat((choice_embeddings, context_embeddings), dim=3)
        embedding_paris = [context_embeddings_pairs.unsqueeze(1).expand(-1, 8, -1, -1), choice_context_order, choice_context_reverse]
        return torch.cat(embedding_paris, dim=2).view(-1, 8, 80, 512)

    def temp_panel_embeddings(self, embeddings):
        embeddings = embeddings.view(-1, 16, 256)
        context_embeddings, choice_embeddings = embeddings[:,:8,:], embeddings[:,8:,:]
        
        #generating f1 tensor
        r1f1 = torch.cat((context_embeddings[:, 0, :],context_embeddings[:, 1, :],context_embeddings[:, 2, :]), dim=1).unsqueeze(1)
        r2f1 = torch.cat((context_embeddings[:, 3, :],context_embeddings[:, 4, :],context_embeddings[:, 5, :]), dim=1).unsqueeze(1)
        baseR3 = torch.cat((context_embeddings[:,6,:],context_embeddings[:,7,:]),dim=1)
        r3f1 = torch.cat((baseR3.unsqueeze(1).expand(-1,8,-1), choice_embeddings), dim=2)
        
        f1 = torch.cat((r1f1, r2f1, r3f1), dim=1)
        
        #generating f2 tensor
        r1f2 = context_embeddings[:, 0:3, :].unsqueeze(1)
        r2f2 = context_embeddings[:, 3:6, :].unsqueeze(1)
        baseR3 = torch.cat((context_embeddings[:, 6, :].unsqueeze(1), context_embeddings[:, 7,:].unsqueeze(1)), dim=1)
        r3f2 = torch.cat((baseR3.unsqueeze(1).expand(-1, 8,-1, -1), choice_embeddings.unsqueeze(2)), dim=2)
        
        f2 = torch.cat((r1f2, r2f2, r3f2), dim=1)
        f2 = torch.cat((f2.unsqueeze(2).expand(-1,-1, 3, -1, -1),f2.unsqueeze(3).expand(-1,-1,-1, 3,-1)),dim=4)
        f2 = f2.view(-1, 10, 9, 512)
        f2 = torch.cat((f2[:,:,1,:].unsqueeze(2),f2[:,:,2,:].unsqueeze(2),f2[:,:,3,:].unsqueeze(2),f2[:,:,5,:].unsqueeze(2),f2[:,:,6,:].unsqueeze(2),f2[:,:,7,:].unsqueeze(2)),dim=2)
        
        return f1, f2

    def rn_sum_features(self, features):
        features = features.view(-1, 8, 80, 256)
        sum_features = torch.sum(features, dim=2)
        return sum_features

    def compute_loss(self, output, target, meta_target,meta_structure):
        target_loss = F.cross_entropy(output[0], target)
        return target_loss

    def forward(self, x):
        panel_features = self.conv(x.view(-1, 1, 80, 80))
        # print(panel_embeddings.size())
        if self.use_tag:
            panel_features = torch.cat((panel_features, self.tags), dim=2)
        panel_embeddings = self.proj(panel_features)
        f1, f2 = self.temp_panel_embeddings(panel_embeddings)
        f1 = self.mlp_f1(f1.view(-1, 768)).view(-1, 10, 256)
        f2 = torch.mean(self.mlp_f2(f2.view(-1, 512)).view(-1, 10, 6, 256), dim=2)
        
        forward, reverse = torch.cat((f1, f2), dim=2).unsqueeze(2), torch.cat((f2, f1), dim=2).unsqueeze(2)
        R = torch.cat((forward, reverse),dim=2)
        R12 = torch.mean(R[:, :2, :, :], dim=1, keepdim=True).view(-1, 1, 1024)
        R3 = R[:, 2:, :, :].view(-1, 8, 1024)
        pred = self.cos(R12, R3)
        return pred,0