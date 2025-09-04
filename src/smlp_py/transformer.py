# SPDX-License-Identifier: Apache-2.0
# This file is part of smlp.

# implementation here is based on https://www.youtube.com/watch?v=3SvuujbjkFw
# and https://www.youtube.com/watch?v=kCc8FmEb1nY

import torch
import torch.nn as nn
#import torch.optim as optim
import torch.utils.data as data
import math
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model%num_heads == 0, "d_model must be divisible by num_heads"
        
        # initialize NN model dimensions
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            #  this constant is used to represent negative infinity; 
            # better to use float('-inf') or torch.finfo().min
            attn_scores = attn_scores.masked_fill(mask==0, -1e9)
        attn_probs = torch.softmax(attn_scores, dim=-1) 
        output = torch.matmul(attn_probs, V)

        return output

    def split_head(self, x):
        batch_size, seq_length, d_model = x.size() # TODO: replace d_model with _, it is not used
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)

    def combine_head(self, x):
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1,2).contiguous().view(batch_size, seq_length, self.d_model)

    def forward(self, Q, K, V, mask=None):
        Q = self.split_head(self.W_q(Q))
        K = self.split_head(self.W_k(K))
        V = self.split_head(self.W_v(V))

        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        output = self.W_o(self.combine_head(attn_output))

        return output

        
class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        
        # initialize nn model
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))
    

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length): # d_model is the length of embedded vectors? max_seq_length is context size?
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term) # even term
        pe[:, 1::2] = torch.cos(position * div_term) # odd term
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]
        
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask):
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x) #todo ff_output or ff? lecture says ff but it is not used
        x = self.norm2(x + self.dropout(ff_output))
        
        return x
        
        
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout, decoder_only):
        super(DecoderLayer, self).__init__()
        
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.decoder_only = decoder_only # decoder-only transformer or traansformer with encoder and decoder?
        
    def forward(self, x, enc_output, src_mask, tgt_mask):
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        if not self.decoder_only:
            attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
            x = self.norm2(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        
        return x
        
class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout, decoder_only=True):
        super(Transformer, self).__init__()
        
        # should the decoder-only transformer be created and trained? The encoder component will be irrelevant.
        self.decoder_only = decoder_only 
        
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)
        
        #print('self.decoder_embedding init', self.decoder_embedding)
        #inp1 = torch.LongTensor([1]); inp2 = torch.LongTensor([[1, 2, 4, 5], [4, 3, 2, 9]])
        #print(type(self.decoder_embedding), self.decoder_embedding(inp1).size(), self.decoder_embedding(inp1), self.decoder_embedding(inp2))
        
        if not self.decoder_only:
            self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
            self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for i in range(num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout, decoder_only) for i in range(num_layers)])
        
        self.fc = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)
        # TODO !!!!!! added 
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.max_seq_length = max_seq_length
        
    # set logger from a caller script
    def set_logger(self, logger):
        self._transformer_logger = logger 
        
    def generate_mask(self, src, tgt):
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2) # TODO  better to use unsqueeze(-2) instead of unsqueeze(2)
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3)
        seq_length = tgt.size(1)
        nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool()
        tgt_mask = tgt_mask & nopeak_mask; #print('tgt_mask', tgt_mask.size())
        
        return src_mask, tgt_mask
    
    def forward(self, src, tgt=None):
        if self.decoder_only:
            tgt = src  # Use src as tgt in decoder-only (causal) mode
            src_mask, tgt_mask = self.generate_mask(tgt, tgt)

            # FIX: replace -100 with 0 (safe default token)
            if (tgt == -100).any():
                self._transformer_logger.warning("Warning: -100 token(s) found in tgt; replaced with <pad> token (id=0)")
                tgt = tgt.masked_fill(tgt == -100, 0)
            tgt_embedded = self.dropout(self.positional_encoding(self.decoder_embedding(tgt)))

            dec_output = tgt_embedded
            for dec_layer in self.decoder_layers:
                dec_output = dec_layer(dec_output, None, None, tgt_mask)  # No encoder input
        else:
            # Standard encoder-decoder mode
            assert tgt is not None, "tgt must be provided in encoder-decoder mode"
            src_mask, tgt_mask = self.generate_mask(src, tgt)

            src_embedded = self.dropout(self.positional_encoding(self.encoder_embedding(src)))

            # FIX: replace -100 with 0 (safe default token)
            if (tgt == -100).any():
                self._transformer_logger.warning("Warning: -100 token(s) found in tgt; replaced with <pad> token (id=0)")
                tgt = tgt.masked_fill(tgt == -100, 0)
            tgt_embedded = self.dropout(self.positional_encoding(self.decoder_embedding(tgt)))

            enc_output = src_embedded
            for enc_layer in self.encoder_layers:
                enc_output = enc_layer(enc_output, src_mask)

            dec_output = tgt_embedded
            for dec_layer in self.decoder_layers:
                dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)

        output = self.fc(dec_output)
        return output

    # d_model is size of numeric embeddings of words?
    # max_seq_length is context size ????
    def set_train_test_data(self):
        self.src_data = torch.randint(self.src_vocab_size, (64, self.max_seq_length)) # (batch_size, seq_length)
        self.tgt_data = torch.randint(self.tgt_vocab_size, (64, self.max_seq_length)) # (batch_size, seq_length)
        self.val_src_data = torch.randint(self.src_vocab_size, (64, self.max_seq_length)) # (batch_size, seq_length)
        self.val_tgt_data = torch.randint(self.tgt_vocab_size, (64, self.max_seq_length)) # (batch_size, seq_length)
    
    def train_transformer_model(self): 
        #print('self.max_seq_length',self.max_seq_length)
        src_data, tgt_data, val_src_data, val_tgt_data = self.src_data, self.tgt_data, self.val_src_data, self.val_tgt_data
        # training
        #src_data = torch.randint(self.src_vocab_size, (64, self.max_seq_length)) # (batch_size, seq_length)
        #tgt_data = torch.randint(self.tgt_vocab_size, (64, self.max_seq_length)) # (batch_size, seq_length)
        #print('src_dara:', src_data)
        loss_func = nn.CrossEntropyLoss(ignore_index=0)
        optimizer = torch.optim.Adam(self.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)        
        self.train()
        for epoch in range(10): # TODO -- 100 or coammand line value
            optimizer.zero_grad()
            output = self(src_data, tgt_data[:, :-1])
            loss = loss_func(output.contiguous().view(-1, self.tgt_vocab_size), tgt_data[:, :-1].contiguous().view(-1))
            loss.backward()
            optimizer.step()
            self._transformer_logger(f"Epoch: {epoch+1} Loss: {loss.item()}")
            self.eval()
        
        
        # validation
        #val_src_data = torch.randint(self.src_vocab_size, (64, self.max_seq_length)) # (batch_size, seq_length)
        #val_tgt_data = torch.randint(self.tgt_vocab_size, (64, self.max_seq_length)) # (batch_size, seq_length)
        
        with torch.no_grad():
            val_output = self(val_src_data, val_tgt_data[:, :-1])
            val_loss = loss_func(val_output.contiguous().view(-1, self.tgt_vocab_size), val_tgt_data[:, :-1].contiguous().view(-1))
            self._transformer_logger(f"Validation loss: {val_loss.item()}")
            
        #print('self.decoder_embedding final', dir(self.decoder_embedding))
        inp1 = torch.LongTensor([1]); inp2 = torch.LongTensor([[1, 2, 4, 5], [4, 3, 2, 9]])
        #print(type(self.decoder_embedding), self.decoder_embedding(inp1).size(), self.decoder_embedding(inp1), self.decoder_embedding(inp2))

    # block_size is the maximum context length (in tokens) that your Transformer model can handle.
    # -It defines the input sequence length for both training and generation.
    # -The model is trained to predict the next token given up to block_size previous tokens.
    # During training, block_size is used to split your training data into input chunks.
    #   e.g. input_ids = [5, 10, 8, 22, 3, 9, 17, 25, ...]
    #   If block_size=4, you train on chunks like:
    #   [5,10,8,22] → target:  [10,8,22,3]
    #   [3,9,17,25] → target:  [9,17,25, ...]
    #   In most implementations, training dataset uses block_size to truncate input to the model.
    #   Model itself needs to know block_size only if it uses it internally (e.g., for caching or enforcing max length).
    # During generation, block_size ensures the model does not exceed the maximum context length it was trained with.
    #   If you try to feed it more than block_size tokens, it will likely throw an error (or behave unexpectedly).    
    def generate(self, idx, max_new_tokens, **kwargs):
        """
        idx: (B, T) array of indices in the current context
        max_new_tokens: int, number of tokens to generate
        kwargs: extra arguments to be ignored (e.g., token_type_ids)
        """
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.max_seq_length:]  # crop context
            logits = self(idx_cond)               # (B, T, vocab_size)
            logits = logits[:, -1, :]             # last token logits
            probs = torch.softmax(logits, dim=-1) # convert to probs
            next_token = torch.multinomial(probs, num_samples=1)  # sample
            idx = torch.cat((idx, next_token), dim=1)  # append
        return idx

    # not used currently -- HF tokenizer.decode() is used instead in llmTrainer.
    def decode(self, token_ids):
        # Assuming you have a tokenizer with a decode method or a vocab dict
        if hasattr(self, 'tokenizer') and callable(getattr(self.tokenizer, 'decode', None)):
            return self.tokenizer.decode(token_ids)
        elif hasattr(self, 'vocab') and isinstance(self.vocab, dict):
            inv_vocab = {v: k for k, v in self.vocab.items()}
            return ''.join([inv_vocab.get(id, '?') for id in token_ids])
        else:
            raise NotImplementedError("No decode method or vocab available.")

# https://www.youtube.com/watch?v=kCc8FmEb1nY    
class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size, block_size=128, n_embed=128, device=None):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
        self.block_size = block_size or 128  # for generate() to work
        
        # each token directly reads off the logits for the next token from a lookup table
        #self.token_embedding_table = nn.Embedding(vocab_size, vocab_size) # TODO !!! why is the second dimention vocab_size????? 
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.block_size = block_size
        self.device = device
        self.lm_head = nn.Linear(n_embed, vocab_size)
        
    def forward(self, idx, targets=None):
        #print('idx', type(idx), idx.size(), idx)
        # idx and targets are both (B, T) tensor of integers
        B, T = idx.shape
        #logits = self.token_embedding_table(idx) # (B, T, C)
        tok_emb = self.token_embedding_table(idx) # (B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=self.device)) # (T, C)
        x = tok_emb + pos_emb # (B, T, C)
        logits = self.lm_head(x) # (B, T, vocab_size)
        
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
            
        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the llast block size tokens
            idx_cond = idx[:, -self.block_size:]
            # get the predictions
            logits, loss = self(idx_cond) # self will go to forward() function
            # focus only on the llast time step
            logits = logits[:, -1, :] # becomes (B,C)
            # apply softmax to get probabillities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
            
        return idx


'''
Main Roles
-- A lightweight character-level model with support for:
-- Bigram model or Transformer model (with token embedding and attention),
-- Tokenization and encoding of input text,
-- Training on character sequences using CrossEntropyLoss,
-- Text generation based on trained model.

Key Parameters
-- Attribute: Description
-- block_size: Max length of input context (e.g., 8 tokens).
-- vocab_tokens: The list of tokens used for encoding/decoding.
-- use_blm: If True, uses Bigram model instead of Transformer.
-- max_iters: Number of training iterations (default: 30).


Key Methods
-- Method: Functionality
-- set_vocab_tokens(): Sets token vocabulary from preprocessed input.
-- read_text(path): Reads and tokenizes input text into encoded training and validation sets.
-- get_batch(split): Prepares a batch of (X, Y) context-target pairs for training.
-- estimate_loss(): Evaluates average training/validation loss over batches.
-- flow(path): Complete training and generation pipeline: trains model and writes generated output to file.

Output

After training, the model generates a sequence of tokens, decodes them into characters, and saves the output 
to a file using generated_text_filename().
'''
# https://www.youtube.com/watch?v=kCc8FmEb1nY    
class LanguageModel:  
    def __init__(self):
        self.batch_size = 32 # number of independent sequences to process in parallel
        self.block_size = 8 # maximum context length for training and predictions
        self.max_iters = 30# 00 TODO !!!!!!!!!!!! default is 3000
        self.eval_interval = 300
        self.learning_rate = 1e-3
        self.vocab_size = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.eval_iters = 200
        self.n_embed = 32
        
        self.encode = None
        self.decode = None
        
        # TODO !!! addid
        self.use_blm = False
        self.vocab_tokens = None
        
        # for reproducibility
        torch.manual_seed(1337)
    
    # set logger from a caller script
    def set_logger(self, logger):
        self._langmodel_logger = logger 
    
    def set_vocab_tokens(self, vocab_tokens):
        self.text_tokens = vocab_tokens
        self.vocab_tokens = sorted(list(set(vocab_tokens)))
    
    # report_file_prefix is a string used as prefix in all report files of SMLP
    def set_report_file_prefix(self, report_file_prefix):
        self.report_file_prefix = report_file_prefix
    
    # required for generating file names of the reports containing model-generated text
    def generated_text_filename(self):
        assert self.report_file_prefix is not None
        return self.report_file_prefix + '_generated_text.txt'
    
    # TODO -- add encoding as user controllable parameter????
    def read_text(self, path_to_file, nlp_vocab=None):
        with open(path_to_file, 'r', encoding='utf-8') as f:
            text = f.read()
        
        if self.vocab_tokens is None:
            chars = sorted(list(set(text))); #print('chars', chars)
        else:
            chars = self.vocab_tokens; #print('vocab_tokens', chars)
        
        self.vocab_size = len(chars); #print('vocab_size', self.vocab_size)
        '''
        TODO !!! check out other encodings of text to numbers / text tokenizers, e.g., Google's SentencePiece and OpenAI's tiktoken used in GPT
        import tiktoken
        enc = tiktoken.get_encoding('gpt2')
        #print(enc.n_vocap) # vocab size, 50257
        enc.encode('hii there') # produces [71, 4178, 612]
        enc.decode([71, 4178, 612]) # produces 'hii there'
        '''
        stoi = { ch:i for i,ch in enumerate(chars) } # string --> list[int]
        itos = { i:ch for i,ch in enumerate(chars) } # list[int] --> string
        #print('stoi', stoi); print('itos', itos)
        self.encode = lambda s: [stoi[c] for c in s] 
        self.decode = lambda l: ''.join([itos[i] for i in l])
        if self.vocab_tokens is None:
            data = torch.tensor(self.encode(text), dtype=torch.long)
        else:    
            data = torch.tensor(self.encode(self.text_tokens), dtype=torch.long)
        #print('data', data.shape, data.dtype, data[ : 1000])
        #data = torch.tensor(self.encode(text), dtype=self.embed_type); print('data', data.shape, data.dtype, data[ : 1000])
        n = int(0.9*len(data)); #print('n', n)
        train_data = data[ :n]; #print('train data size', train_data.size())
        val_data = data[ n: ]; #print('val data size', val_data.size())
        
        #print(train_data[ :self.block_size+1])
        
        x = train_data[:self.block_size]
        y = train_data[1:self.block_size+1]
        for t in range(self.block_size):
            context = x[:t+1]
            target = y[t]
            #print(f"context {context} --> target {target}")
            
        return train_data, val_data
    
    # generate a small batch of data of inputs x and targets y
    def get_batch(self, split, train_data, val_data):
        data = train_data if split == 'train' else val_data
        ix = torch.randint (len(data) - self.block_size, (self.batch_size, ))
        x = torch.stack([data[i:i+self.block_size] for i in ix])
        y = torch.stack([data[i+1:i+self.block_size+1] for i in ix])
        x, y = x.to(self.device), y.to(self.device)
        return x, y
    
    @torch.no_grad()
    def estimate_loss(self, model, train_data, val_data, loss_func):
        out = {}
        model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(self.eval_iters)
            for k in range(self.eval_iters):
                X, Y = self.get_batch(split, train_data, val_data)
                #print('applying the model -------------'); print(X.size(), type(X)); print(Y.size(), type(Y))
                if self.use_blm:
                    logits, loss = model(X, Y)
                else:
                    pred = model(X, Y); #print('pred', type(pred), pred.size())
                    #print('predictions', pred.contiguous().view(-1, self.vocab_size))
                    loss = loss_func(pred.contiguous().view(-1, self.vocab_size), Y.contiguous().view(-1)); #print('loss', loss)
                    #logits, loss = BigramLanguageModel(self.vocab_size, self.block_size, self.n_embed, self.device).forward(pred)
                    
                losses[k] = loss.item()
            out[split] = losses.mean()
        model.train()
        
        return out
            
            
    def flow(self, path_to_file):
        train_data, val_data = self.read_text(path_to_file)
        xb, yb = self.get_batch('train', train_data, val_data)
        #print('xb', xb.shape, 'yb', yb.shape)
        
        if self.use_blm:
            model = BigramLanguageModel(self.vocab_size, self.block_size, self.n_embed, self.device)
        else:
            model = Transformer(src_vocab_size=self.vocab_size, tgt_vocab_size=self.vocab_size, 
                d_model = 32, num_heads = 8, num_layers = 6, d_ff = 256, max_seq_length = 10, dropout = 0.1)
        
        blm = model.to(self.device)
        optimizer = torch.optim.Adam(blm.parameters(), lr=self.learning_rate)
        loss_func = nn.CrossEntropyLoss(ignore_index=0) # added !!!!
        for iter in range(self.max_iters):
            # every once in a while, evaluate loss on training and val sets
            # TODO !!!! drop False, enable self.estimate_loss()
            if False and iter % self.eval_interval == 0:
                losses = self.estimate_loss(blm, train_data, val_data, loss_func)
                print(f"step {iter}: train loss {losses['train']} val loss {losses['val']}")
            
            # sample a batch of data
            xb, yb = self.get_batch('train', train_data, val_data)
            
            '''
            # dispaly context-target pairs used in training
            for b in range(self.batch_size):
                for t in range(self.block_size):
                    context = xb[b, :t+1]
                    target = yb[b, t]
                    print(f"input {context.tolist()} --> target {target} ")
            '''
            
            # evaluate the loss
            if self.use_blm:
                logits, loss = blm(xb, yb)
            else:
                pred = blm(xb, yb); #print('pred', type(pred), pred.size())
                #print('predictions', pred.contiguous().view(-1, self.vocab_size))
                loss = loss_func(pred.contiguous().view(-1, self.vocab_size), yb.contiguous().view(-1)); print('loss', float(loss))
                #logits, loss = BigramLanguageModel(self.vocab_size, self.block_size, self.n_embed, self.device).forward(pred)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        
        print('loss', loss.item())
        context = torch.zeros((1, 1), dtype=torch.long, device=self.device)
        #context = torch.zeros((1, 1), dtype=self.embed_type, device=self.device)
        # TODO !!!!!!!!!!!!!!!!!! temp print(self.decode(blm.generate(context, max_new_tokens=500)[0].tolist()))
        # TODO !!!!!!! here we use BigramLanguageModel model for genration, Transformer-trained models are not 
        # supported for generation here. They are supported in smlp_llm module.
        gen_text = self.decode(BigramLanguageModel(self.vocab_size, self.block_size, self.n_embed, 
           self.device).generate(context, max_new_tokens=500)[0].tolist())
        #gen_text = self.decode(blm.generate(context, max_new_tokens=500)[0].tolist())

        print(gen_text) 
        #assert False
        gen_file = self.generated_text_filename()
        self._langmodel_logger.info('Writing the generated text into file ' + gen_file)
        f = open(gen_file, "w")
        f.write(gen_text)
        f.close()
    
'''
 tensor([[-0.6341, -1.5322,  1.4393,  0.9145, -0.3685,  0.4999,  0.7808, -0.2803,
          0.3839, -0.3487,  0.2024, -0.4766,  0.5524,  1.1519, -0.1288, -0.3035,
         -0.1205, -0.0652, -0.4682, -0.2069,  2.2270, -0.1608, -0.7716,  1.6541,
         -1.2496,  0.4100, -1.0941,  0.0598,  2.1701,  0.0340,  0.8675,  1.0118]]
         
 tensor([[-0.6349, -1.5332,  1.4383,  0.9135, -0.3695,  0.4995,  0.7818, -0.2793,
          0.3831, -0.3497,  0.2023, -0.4756,  0.5534,  1.1529, -0.1278, -0.3025,
         -0.1215, -0.0663, -0.4692, -0.2059,  2.2280, -0.1618, -0.7706,  1.6551,
         -1.2490,  0.4109, -1.0931,  0.0588,  2.1691,  0.0350,  0.8665,  1.0128]]
  
 # we want x[b,t] = mean_{i<=t} x[b,i]
 #loops -- not efficient
 nbow = torch.zeros(B,T,C)
 for b in range(B):
    for t in range(T):
        xprev = x[b, :t+1]
        xbow[b,t] = torch.mean(uprev, 0)
 # tenzor multiplications -- fast
 wei = torch.tril(torch.ones(T,T))
 wei = wei / wei.sum(1, keepdim=True)
 xbow2 = wei @ x # (T,T) @ (B,T,C) --> (B,T,T) @ (B,T,C) --> (B,T,C)
 torch.allclose(xbow, xbow2) # produces True, meaning the two implementation give same results, the second one is much faster due to parallel executions in GPU.

# 3rd way -- using softmax
tril = torch.tril(torch.ones(T,T))
wei = torch.zeros((T,T))
wei = wei.masked_fill(tril==0, float('-inf'))
wei = F.softmax(wei, dim=-1)
xbox3 = wei @ x
torch.allclose(xbow, xbow3) # produces True 
'''
    
    
    