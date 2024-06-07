import torch
import torch.nn as nn
from torch.nn import functional as F

from decimal import Decimal
import timeit
import os
import sys

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

# hyperparameters
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def gpu_avlailabe() -> bool :
    return torch.cuda.is_available()

batch_size = 128 # how many independent sequences will we process in parallel?
block_size = 256 # what is the maximum context length for predictions?

# max_iters = 5000
# eval_interval = 500
# eval_iters = 200

learning_rate = 3e-4
n_embd = 384
n_head = 12
n_layer = 6
dropout_rate = 0.2
# ------------

torch.manual_seed(1337)

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()  
# @torch.no_grad() 装饰器下所有计算都不会跟踪梯度，不会进行任何梯度更新。这在评估模型或进行不涉及反向传播的计算时非常有用，
# 因为它可以减少内存消耗并提高计算速度。在做模型推理时候用。
def estimate_loss(model):
    eval_iters = 200
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1=nn.Linear(10,4) 
        self.act1=nn.Sigmoid()
  
    def forward(self,x):
        x=self.layer1(x)
        x=self.act1(x)
        return x
        
    def generate(self, idx, max_new_tokens):
        pass

## -----------------------------

class Head(nn.Module):
    """ one head of self-attention """
    def __init__(self, head_size):
        super().__init__()
        self.key   = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        # 模型中注册一个张量（buffer）, mask。这个函数通常用于那些不是模型参数但在模型中需要使用的张量。这些张量不会在模型训练过程中更新，但它们对于模型的正向传播或反向传播是必需的。
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size))) 

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
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
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
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPTLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

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

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C) #batch, time(contex_size), Channel(embedding vector size) 
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
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
            idx_cond = idx[:, -block_size:]
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
    

def load_pretrained_model(model:nn.Module, pre_train_model: str, post_load_action: str= 'train'):

    # if not isinstance(pre_train_model, str):
    #     raise ValueError(f"pre_train_model{pre_train_model} must be a string representing a file path")

    try:
        model.load_state_dict(torch.load(pre_train_model))
    except Exception as e:
        print(f"load_state_dict: {pre_train_model}, error: {e}")

    if post_load_action == 'train':
        model.train()  
    elif post_load_action == 'eval':
        model.eval()  
    return model

def create_model(model_path: str= None, mode: str = 'train'):
    
    model = GPTLanguageModel()
    # model = MyModel()

    if model_path == None:
        print("init model from random...")
    else:

        model = load_pretrained_model(model, model_path, mode)

    return model

def make_path(root_path_name: str, name:str, suffix: int, type:str='mdl'):
    if root_path_name == None:
        current_path = os.getcwd()
    else:
        current_path = root_path_name

    f_name = f"{name}_{type}_{suffix}.pth"

    return os.path.join(current_path, f_name)



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

    m = create_model(model_load_path, 'train').to(device)

    # print the number of parameters in the model
    print(f"load model with {sum(p.numel() for p in m.parameters())/1e6} M parameters")
    
    # create a PyTorch optimizer
    optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)
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
    
        # every once in a while evaluate the loss on train and val sets
        if iter % eval_interval == 0 or iter == iter - 1:
            losses = estimate_loss(m)
            print(f"[step {iter:<6}]: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    
        # sample a batch of data
        xb, yb = get_batch('train')
        xb, yb = xb.to(device), yb.to(device)
    
        # evaluate the loss
        logits, loss = m(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if i % accumulation_steps == 0:
            optimizer.step()
        # optimizer.step()
        i+=1
        
        if gpu_avlailabe():
            torch.cuda.empty_cache()

    # save_data = {
    # "model_state": m.state_dict(),
    # "optimizer_state": optimizer.state_dict()
    # }
   
    # opt_save = f"{save_name}_opt_{iter}.pth"
    print(f"model train completed, model will be saved as {model_save_path}, optimizer is saved as {opt_save_path}")
    torch.save(m.state_dict(), model_save_path)
    torch.save(optimizer.state_dict(), opt_save_path)

    # return m


def test_model(save_name: str, sufix:int, max_token: int =300):
    print(f"test model")
    pre_train_path = make_path(None, save_name, sufix, 'mdl' )

    m = create_model(pre_train_path, 'eval').to(device)

    # generate from the model
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    print("\n")
    print("="*30)
    print(f"Test model by generating {max_token} tokens:")
    print(decode(m.generate(context, max_new_tokens=max_token)[0].tolist()))
    print("="*30)
    print("\n")


if __name__ == "__main__":
    # 检查是否有足够的参数
    if len(sys.argv) == 2:
        print_iter = int(sys.argv[1])
    else:
        print("没有提供合适命令行参数。")

    DRY_RUN = False
    
    model_name_list = ['first','second','third','fourth','fifth']
    iter_list       = [100,    100,    100,   100,   100]

    # torch.cuda.memory._record_memory_history()

    for n in range(len(model_name_list)):

        if n == 0:
            load_name = None
        else:
            load_name = model_name_list[n-1]

        save_name = model_name_list[n]

        # print(f"n={n}, load_name = {load_name}, save_name={save_name}")

        train_model(iter_list[n], print_iter, load_name, save_name,  DRY_RUN)

        test_model(save_name, iter_list[n], 300)

    # torch.cuda.memory._dump_snapshot("my_snapshot.pickle")

