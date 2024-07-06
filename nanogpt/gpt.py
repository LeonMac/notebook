import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.cuda as tc

from decimal import Decimal
import timeit
import os


## timing decorator
def timing(func):
    """decoration for timing calculation"""
    def get_timing(*args, **kwargs):
        _begin = timeit.default_timer()
        result = func(*args, **kwargs)
        duration = timeit.default_timer() - _begin
        d = Decimal(str(duration)).quantize(Decimal("0.001"), rounding = "ROUND_HALF_UP")
        d_str = f"time: {d} sec"
        print(f"{func.__name__} takes {d_str} seconds\n")
        return result, d_str
    return get_timing

def gpu_avlailabe() -> bool :
    return tc.is_available()

def memory_info(claim:str ):
    print(claim)
    print(f"allocated {tc.memory_allocated(device=None)}, max {tc.max_memory_allocated(device=None)}")
    print(f"reserved  {tc.memory_reserved(device=None)}, max {tc.max_memory_reserved(device=None)} ")


def define_dtype(d_type: str= 'long'):
    if d_type == 'long':
        return torch.long
    elif d_type == 'int':
        return torch.int
    pass


global prj_path;            prj_path  = os.getcwd()
global model_save_dir;      model_save_dir = 'model_save'
global model_arch_dir;      model_arch_dir = 'model_arch'

# hyperparameters
global device; device = 'cuda' if tc.is_available() else 'cpu'
global dropout_rate;  dropout_rate = 0.2
global learning_rate; learning_rate = 3e-4
global d_type; d_type='long'

# max_iters = 5000
# eval_interval = 500
# eval_iters = 200


def global_cofig(mdl_name:str, big:bool= True):
    global g_batch_size # how many independent sequences will we process in parallel?
    global g_block_size # what is the maximum context length for predictions?

    global g_n_embd  # number of embedding
    global g_n_head  # number of head
    global g_n_layer # number of layer
    global save_nn_name
    gpu = 'big'

    if big: # for GPU like 3090
        
        g_batch_size = 128 # how many independent sequences will we process in parallel?
        g_block_size = 256 # what is the maximum context length for predictions?

        g_n_embd  = 384
        g_n_head  = 8
        g_n_layer = 6

    else: # for GPU like 1080
        g_batch_size = 32 # how many independent sequences will we process in parallel?
        g_block_size = 64 # what is the maximum context length for predictions?

        g_n_embd  = 64  # number of embedding
        g_n_head  = 1   # head number
        g_n_layer = 1   # number of layer

        gpu = 'small'

    
    mdl_sav_name = '' if mdl_name == None else mdl_name
    save_nn_name=f"mdl[{mdl_sav_name}]-batch{g_batch_size}-block{g_block_size}-embd{g_n_embd}-head{g_n_head}-layer{g_n_layer}"
    print(f"global config for {gpu} GPU") 



torch.manual_seed(1337)

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
global g_vocab_size; g_vocab_size = len(chars)

# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Train and test splits
glb_data = torch.tensor(encode(text), dtype=define_dtype('long'))
n = int(0.9*len(glb_data)) # first 90% will be train, rest val
train_data = glb_data[:n]
val_data = glb_data[n:]

# dataset_type = ['train','eval']
# data loading
def get_batch(split, dev):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data

    ix = torch.randint(len(data) - g_block_size, (g_batch_size,))
    x  = torch.stack([data[i:i+g_block_size] for i in ix])
    y  = torch.stack([data[i+1:i+g_block_size+1] for i in ix])
    x, y = x.to(dev), y.to(dev)
    return x, y

eval_iters = 200
# @torch.no_grad() 装饰器下所有计算都不会跟踪梯度，不会进行任何梯度更新。这在评估模型或进行不涉及反向传播的计算时非常有用，
# 因为它可以减少内存消耗并提高计算速度。在做模型推理时候用。
@torch.no_grad()  
def estimate_losses(mdl, dev): # test loss on both train and eval dataset
    # eval_iters = 200
    out = {}
    mdl.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split, dev)
            logits, loss = mdl(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    # mdl.train()
    return out


@torch.no_grad()  
def eval_loss(mdl, dev): # test loss on eval dataset
    # eval_iters = 200
    mdl.eval()

    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        X, Y = get_batch('eval', dev)
        _, loss = mdl(X, Y)
        losses[k] = loss.item()

    # mdl.train()
    return losses.mean()

# super simple bigram model
class SimpleLLM(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):

        # idx and targets are both (B,T) tensor of integers
        logits = self.token_embedding_table(idx) # (B,T,C)

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
            # get the predictions
            logits, loss = self(idx)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx


## -----------------------------

class Head(nn.Module):
    """ one head of self-attention """
    def __init__(self, head_size):
        super().__init__()
        self.key   = nn.Linear(g_n_embd, head_size, bias=False)  # g_n_embd: input token dimension; head_size: final output dimension of each head.
        self.query = nn.Linear(g_n_embd, head_size, bias=False)
        self.value = nn.Linear(g_n_embd, head_size, bias=False)
        # register_buffer: 模型中注册一个张量（buffer）, mask。这个函数通常用于那些不是模型参数但在模型中需要使用的张量。这些张量不会在模型训练过程中更新，但它们对于模型的正向传播或反向传播是必需的。
        self.register_buffer('tril', torch.tril(torch.ones(g_block_size, g_block_size))) 

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B,T,C = x.shape
        k = self.key(x)   # (B,T,hs)
        q = self.query(x) # (B,T,hs)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,hs)
        out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])  # we get num_heads of head, each head provides head_size output dimension, all heads together provide num_heads*head_size output
        self.proj  = nn.Linear(head_size * num_heads, g_n_embd)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # torch.cat将这些张量收集到一个列表中，并沿着指定的维度dim进行拼接。dim=-1表示沿着each head out最后一个维度(hs)进行拼接。
        # 最终得到的out形状将是(batch_size, time-stamp * hs)。
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        out = self.dropout(self.proj(out)) # pass out through self.proj
        return out

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout_rate),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head # why do this?? this overide the global head_size parameter??
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        # simply making the Residual Connection foth both multi-head and also feedfoward
        # do normalization before the multihead, residual add after the multihead -- this is something different from the original paper "AiAYN", 
        x = x + self.sa(self.ln1(x))   #1.do layer normalization; 2.go through multihead; 3. add residual
        x = x + self.ffwd(self.ln2(x)) #1. do layer normalization; 2.go through ff; 3. add residual
        return x

class GPTLanguageModel(nn.Module):

    def __init__(self, dev):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(g_vocab_size, g_n_embd)
        self.position_embedding_table = nn.Embedding(g_block_size, g_n_embd)
        # multiple block connected on sequential
        self.blocks = nn.Sequential(*[Block(g_n_embd, n_head=g_n_head) for _ in range(g_n_layer)])
        self.ln_f = nn.LayerNorm(g_n_embd) # final layer normalization after all blocks
        self.lm_head = nn.Linear(g_n_embd, g_vocab_size)
        self.dev = dev

        # better init, not covered in the original GPT video, but important, will cover in followup video

        self.apply(self._init_weights) # 递归地对每个子模块init

    def _init_weights(self, module):
        """apply,执行初始化权重"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):  # idx: xb, targets: yb
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C) #batch, time(contex_size), Channel(embedding vector size) 
        pos_emb = self.position_embedding_table(torch.arange(T, device=self.dev)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C) #position embedding, pos_emb 的第一个维度（时间步维度）大小为 T，而 tok_emb 的第一个维度大小为 B。由于 pos_emb 的批次维度大小为1，它将被复制 B 次，以形成 (B, T, C) 形状的张量。
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

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
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -g_block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx
    

def load_pretrained_model(mdl:nn.Module, pre_train_model: str, post_load_action: str= 'train'):

    # if not isinstance(pre_train_model, str):
    #     raise ValueError(f"pre_train_model{pre_train_model} must be a string representing a file path")

    try:
        mdl.load_state_dict(torch.load(pre_train_model))
    except Exception as e:
        print(f"load_state_dict: {pre_train_model}, error: {e}")

    if post_load_action == 'train':
        mdl.train()  
    elif post_load_action == 'eval':
        mdl.eval()  
    return mdl

def create_model(dev, model_path: str= None, mode: str = 'train'):
    
    mdl = GPTLanguageModel(dev)
    # mdl = SimpleLLM(vocab_size)

    if model_path == None:
        print("init model from random...")
    else:
        mdl = load_pretrained_model(mdl, model_path, mode)

    mdl = mdl.to(dev)

    return mdl

def make_path(root_path_name: str, name:str, suffix: int, type:str='mdl'):
    if root_path_name == None:
        current_path = prj_path
    else:
        current_path = root_path_name

    f_name = f"{name}_{type}_{suffix}.pth"

    return os.path.join(current_path, model_save_dir, f_name)


@timing
def train_model(max_iter:int, eval_interval:int, load_name:str, save_name: str, dry_run: bool = False):
    print(f"train model")

    if load_name != None:
        model_load_path = make_path(None, load_name, max_iter, 'mdl' )
        opt_load_path   = make_path(None, load_name, max_iter, 'opt' )
    else:
        model_load_path = None
        opt_load_path   = None

    if save_name != None:
        model_save_path = make_path(None, save_name, max_iter, 'mdl' )
        opt_save_path   = make_path(None, save_name, max_iter, 'opt' )
    else:
        model_save_path = None
        opt_save_path   = None

    mdl = create_model(device, model_load_path, 'train')

    # print the number of parameters in the model
    print(f"load model with {sum(p.numel() for p in mdl.parameters())/1e6} M parameters")

    # create a PyTorch optimizer
    optimizer = torch.optim.AdamW(mdl.parameters(), lr=learning_rate)
    if opt_load_path != None:
        optimizer.load_state_dict(torch.load(opt_load_path, map_location=device))

    
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device)

    accumulation_steps = 4  # https://saturncloud.io/blog/how-to-solve-cuda-out-of-memory-error-in-pytorch/
    i = 1
    for iter in range(max_iter):
        if dry_run:
            print(f"dry_run : exist after before iter: {iter}")
            break
        
        # memory_info(f"iter [{iter}] before train:")
        # sample a batch of data
        xb, yb = get_batch('train', device)
        # xb, yb = xb.to(device), yb.to(device)
    
        # train loss
        logits, train_loss = mdl(xb, yb)

        # every once in a while evaluate the loss on train and val sets
        if iter % eval_interval == 0 or iter == iter - 1:
            # losses = estimate_losses(m, device)
            # print(f"[step {iter:<6}]: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            evl_loss = eval_loss(mdl, device)
            print(f"[step {iter:<6}]: train loss {train_loss:.4f}, eval loss {evl_loss:.4f}")
            # memory_info(f"")

        # if i == 1 :
        #     make_dot(train_loss, params=dict(list(mdl.named_parameters()))).render("model_torchviz", format="png")
        #     print('saving net structure')

        optimizer.zero_grad(set_to_none=True)
        train_loss.backward()
        if i % accumulation_steps == 0:
            optimizer.step()
        # optimizer.step()
        i+=1
        
        if gpu_avlailabe():
            tc.empty_cache()

        tc.reset_peak_memory_stats(device=None)
        # memory_info(f"iter [{iter}] after train:")

    # save_data = {
    # "model_state": m.state_dict(),
    # "optimizer_state": optimizer.state_dict()
    # }
   
    # opt_save = f"{save_name}_opt_{iter}.pth"
    print(f"model train completed, model will be saved as {model_save_path}, optimizer is saved as {opt_save_path}")
    torch.save(mdl.state_dict(), model_save_path)
    torch.save(optimizer.state_dict(), opt_save_path)

    # return m
    # del mdl
    # del optimizer
    # del xb
    # del yb


def test_model(save_name: str, sufix:int, max_token: int =300):
    print(f"test model")
    pre_train_path = make_path(None, save_name, sufix, 'mdl' )

    m = create_model(device, pre_train_path, 'eval')

    # generate from the model
    context = torch.zeros((1, 1), dtype=define_dtype('long'), device=device)
    print("\n")
    print("="*30)
    print(f"Test model by generating {max_token} tokens:")
    print(decode(m.generate(context, max_new_tokens=max_token)[0].tolist()))
    print("="*30)
    print("\n")


### model visualization

llm_level = ['head','multihead','block','gpt']

def gen_model(level:str, dev:str):
    mdl = None
    head_size = g_n_embd//g_n_head #??
    if level == 'head':
        mdl = Head(head_size)
    elif level == 'head':
        mdl =  nn.ModuleList([Head(head_size) for _ in range(g_n_head)])
    elif level == 'block':
        mdl =  nn.Sequential(*[Block(g_n_embd, n_head=g_n_head) for _ in range(g_n_layer)])
    elif level == 'gpt':
        mdl =  GPTLanguageModel(dev)
    else:
        print(f'incorrect level value {level}')
        exit(0)
    return mdl.to(dev)


def gen_data(level:str, dev:str):
    head_size = g_n_embd//g_n_head #??
    x, y = get_batch('train', dev) # this is required for gen data for all levels
    if level == 'head':
        pass
    elif level == 'multihead':
        pass
    elif level == 'block':
        B, T = x.shape

        token_embedding_table = nn.Embedding(g_vocab_size, g_n_embd).to(dev)
        position_embedding_table = nn.Embedding(g_block_size, g_n_embd).to(dev)
        tok_emb =  token_embedding_table(x)
        pos_emb =  position_embedding_table (torch.arange(T, device=dev))
        emb_input = (tok_emb+pos_emb).to(dev)
        mdl = gen_model(level, dev)
        y = mdl(emb_input)

        return emb_input, y
    elif level == 'gpt': # data input/output to llm
        # x, y = get_batch('train', dev)
        return x, y
    else:
        print(f'incorrect level value {level}')
        exit(0)


# def gen_gpt_data_model_for_visualization(model_level: str):
#     # if model_level == 'gpt':
#     m = gen_model (model_level, device)
#     # m = create_model(dev, None, 'train')
#     xb, yb = gen_data (model_level, device)
#     return m, xb, yb


