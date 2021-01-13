import torch
import torch.nn as nn 
from Layers import EncoderLayer, DecoderLayer
from Embed import Embedder, PositionalEncoder
from Sublayers import Norm
import copy

def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads, dropout):
        super().__init__()
        self.N = N
        self.embed = Embedder(vocab_size, d_model)
        self.pe = PositionalEncoder(d_model, dropout=dropout)
        self.layers = get_clones(EncoderLayer(d_model, heads, dropout), N)
        self.norm = Norm(d_model)
    def forward(self, src, mask):
        x = self.embed(src)
        x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x, mask)
        return self.norm(x)
    
class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads, dropout):
        super().__init__()
        self.N = N
        self.embed = Embedder(vocab_size, d_model)
        self.pe = PositionalEncoder(d_model, dropout=dropout)
        self.layers = get_clones(DecoderLayer(d_model, heads, dropout), N)
        self.norm = Norm(d_model)
    def forward(self, trg, e_outputs, src_mask, trg_mask):
        x = self.embed(trg)
        x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x, e_outputs, src_mask, trg_mask)
        return self.norm(x)

class Transformer(nn.Module):
    def __init__(self, src_vocab, trg_vocab, d_model, N, heads, dropout):
        super().__init__()
        self.encoder = Encoder(src_vocab, d_model, N, heads, dropout)
        self.decoder = Decoder(trg_vocab, d_model, N, heads, dropout)
        self.out = nn.Linear(d_model, trg_vocab)
    def forward(self, src, trg, src_mask, trg_mask):
        e_outputs = self.encoder(src, src_mask)
        #print("DECODER")
        print('e_outputs.shape', e_outputs.shape)
        d_output = self.decoder(trg, e_outputs, src_mask, trg_mask)
        output = self.out(d_output)
        return output


class TransformerVAE(nn.Module):
    def __init__(self, src_vocab, trg_vocab, d_model, N, heads, dropout, latent_size, e_outputs_length):
        super().__init__()
        self.encoder = Encoder(src_vocab, d_model, N, heads, dropout)
        self.mu = nn.Linear(d_model, latent_size)
        self.logvar = nn.Linear(d_model, latent_size)
        self.latent_size = latent_size
        self.e_outputs_length = e_outputs_length
        self.latent_to_e = nn.Linear(latent_size, d_model)

        self.decoder = Decoder(trg_vocab, d_model, N, heads, dropout)
        self.out = nn.Linear(d_model, trg_vocab)
    def forward(self, src, trg, src_mask, trg_mask):
        e_outputs = self.encoder(src, src_mask)
        # e_outputs is a batch_size * seq_len, need to average
        e_outputs_mean = e_outputs.mean(dim=1)
        # print('e_outputs_mean', e_outputs_mean.shape)
        mu = self.mu(e_outputs_mean)
        logvar = self.logvar(e_outputs_mean)
        z = self.reparameterize(mu, logvar)
        #print("DECODER")
        # need to go from the latent space to e_outputs
        # print('src_mask', src_mask.shape)
        z_e_outputs = self.latent_to_e(z.unsqueeze(1).repeat(1, src_mask.shape[2], 1))
        d_output = self.decoder(trg, z_e_outputs, src_mask, trg_mask)
        output = self.out(d_output)
        return output
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

def get_model(opt, src_vocab, trg_vocab):
    
    assert opt.d_model % opt.heads == 0
    assert opt.dropout < 1

    if opt.vae:
        model = TransformerVAE(src_vocab, trg_vocab, opt.d_model, opt.n_layers, opt.heads, opt.dropout, 100, 15)
    else:
        model = Transformer(src_vocab, trg_vocab, opt.d_model, opt.n_layers, opt.heads, opt.dropout)
       
    if opt.load_weights is not None:
        print("loading pretrained weights...")
        model.load_state_dict(torch.load(f'{opt.load_weights}/model_weights'))
    else:
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p) 
    
    if opt.device == 0:
        model = model.cuda()
    
    return model
    
