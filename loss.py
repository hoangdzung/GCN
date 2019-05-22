import torch

def sigmoid_cross_entropy_with_logits(logits, labels):
    sig_aff = torch.sigmoid(logits)
    loss = torch.sum(labels * -torch.log(sig_aff+1e-10) + (1 - labels) * -torch.log(1 - sig_aff+1e-10))
    return loss

def n2v_loss(embedding, adj):
    embedding =  embedding/embedding.norm(p=2, dim=1, keepdim=True)
    embed_pairwise = torch.matmul(embedding, torch.transpose(embedding,0,1))
    embed_pairwise = embed_pairwise.view(-1)
    labels = adj.view(-1)
    node2vec_loss = sigmoid_cross_entropy_with_logits(embed_pairwise, labels)
    return node2vec_loss

def edge_balance_loss(embedding, lam=0.7):
    n = adj.shape[0]
    g = embedding.shape[1]
    embedding_T = torch.transpose(embedding,0,1)
    pred = torch.matmul(torch.matmul(embedding_T,adj),embedding)
    ncut_loss = torch.sum(torch.tril(pred, diagonal=-1)) + torch.sum(torch.triu(pred, diagonal=1)) 
    # balance_loss = torch.sum((torch.sum(self.assign_tensor, dim=1) - self.input_dim//self.num_parts)**2)
    balance_loss = torch.sum((torch.diagonal(pred) - torch.sum(torch.diagonal(pred))/g)**2)
        
    return lam*ncut_loss+(1-lam)*balance_loss