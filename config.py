# quantization param
use_adapter = True  # whether to use adapter to quantize
quant_num_bits = 5

# Network hyper param
hidden_channel = 128
res_hidden_channel = 32
n_res_layers = 2
num_embedding = 16
embedding_size = 4

# Training hyper param
batch_size = 64
lr = 1e-4

# Other hyper param
epochs = 30
seed = None
num_workers = 8
beta = 0.25
gradient_clip = 1
log_interval = 100
