import torch
import numpy as np
import os
import typing
import sys

from torch.nn.functional import cross_entropy
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import transformer_utils
import tokenizer_endecoder
import argparse
import json

# Data Loader
def data_loading(dataset: np.ndarray, batch_size: int, context_length: int, device: str = 'cpu') -> tuple[torch.Tensor, torch.Tensor]:
    # Maximum valid starting index (need context_length + 1 for label)
    max_start_idx = len(dataset) - context_length
    # Randomly sample starting indices (with replacement)
    start_indices = np.random.randint(0, max_start_idx, size=batch_size) # range is [0, max_start_idx - 1]
    # Extract sequences
    inputs = []
    labels = []
    for start_idx in start_indices:
        # Input: tokens from start_idx to start_idx + context_length
        inputs.append(dataset[start_idx:start_idx + context_length])
        # Label: tokens from start_idx+1 to start_idx + context_length + 1
        labels.append(dataset[start_idx + 1:start_idx + context_length + 1])
    
    # Convert to numpy arrays, then to torch tensors
    inputs_array = np.array(inputs)
    labels_array = np.array(labels)
    
    # Convert to torch tensors and move to device
    inputs_tensor = torch.from_numpy(inputs_array).to(device)
    labels_tensor = torch.from_numpy(labels_array).to(device)
    
    return inputs_tensor, labels_tensor


def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, iteration: int, out: str | os.PathLike | typing.BinaryIO | typing.IO):
    # torch.save(model.state_dict(), out + '/model.ckpt')
    # torch.save(optimizer.state_dict(), out + '/opt.ckpt')
    # torch.save(iteration, out + "/iter.ckpt")
    checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'iteration': iteration
        }
    torch.save(checkpoint, out)

def load_checkpoint(src: str | os.PathLike | typing.BinaryIO | typing.IO, model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> int:
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['iteration']

def parse_training_args():
    """Parse command-line arguments for training configuration."""
    parser = argparse.ArgumentParser(description='Train a transformer language model')
    
    # Option 1: Load all params from a config file
    parser.add_argument('--config', type=str, help='Path to JSON config file with all parameters')
    
    # Option 2: Individual CLI arguments
    # Do process?
    parser.add_argument('--preprocess_data', type=bool, default=False, help='Pre-process the data if needed. Tokenize the input text once and save for later.')
    # Data parameters
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--context_length', type=int, default=128, help='Context length for sequences')
    
    # Tokenizer parameters 
    # './tokenizer_output/tinystory_train_10000vocab/vocab.json', './tokenizer_output/tinystory_train_10000vocab/merges.txt', ['<|endoftext|>']
    parser.add_argument('--vocab_filepath', type=str, default='./tokenizer_output/tinystory_train_10000vocab/vocab.json', help = 'The input file for vocab for the tokenizer.')
    parser.add_argument('--merges_filepath', type=str, defaul= './tokenizer_output/tinystory_train_10000vocab/merges.txt', help = 'The merge rules for the tokenizer.')
    parser.add_argument('--special_token_list', type=list, default = ['<|endoftext|>'], help = 'The list of special tokens for the tokenizer.')

    # Model parameters
    parser.add_argument('--vocab_size', type=int, default=10000, help='Vocabulary size')
    parser.add_argument('--num_layers', type=int, default=4, help='Number of transformer layers')
    parser.add_argument('--d_model', type=int, default=256, help='Model dimension')
    parser.add_argument('--num_heads', type=int, default=6, help='Number of attention heads')
    parser.add_argument('--d_ff', type=int, default=1024, help='Feed-forward dimension')
    parser.add_argument('--theta', type=float, default=10000.0, help='RoPE theta parameter')
    
    # Optimizer parameters
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Initial learning rate')
    parser.add_argument('--betas', type=float, nargs=2, default=[0.9, 0.95], help='Adam beta parameters')
    parser.add_argument('--eps', type=float, default=1e-8, help='Adam epsilon')
    parser.add_argument('--weight_decay', type=float, default=1e-2, help='Weight decay')
    
    # Learning rate schedule parameters
    parser.add_argument('--max_learning_rate', type=float, default=1e-3, help='Max learning rate')
    parser.add_argument('--min_learning_rate', type=float, default=1e-4, help='Min learning rate')
    parser.add_argument('--warmup_iters', type=int, default=1000, help='Warmup iterations')
    parser.add_argument('--cosine_cycle_iters', type=int, default=10000, help='Cosine cycle iterations')
    
    # Gradient clipping
    parser.add_argument('--max_l2_norm', type=float, default=1.0, help='Max L2 norm for gradient clipping')
    
    # Training parameters
    parser.add_argument('--max_iters', type=int, default=10000, help='Maximum training iterations')
    parser.add_argument('--eval_interval', type=int, default=100, help='Evaluation interval')
    parser.add_argument('--checkpoint_interval', type=int, default=1000, help='Checkpoint save interval')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='Directory to save checkpoints')
    
    # Data paths
    parser.add_argument('--preprocess_train_data', type=str, default = './data/preproces/train_tokens.bin', help='Path to preprocessed train data in bin.')
    parser.add_argument('--preprocess_val_data', type=str, default = './data/preproces/val_tokens.bin', help='Path to preprocessed val data in bin.')
    parser.add_argument('--train_data', type=str, default = './data/TinyStoriesV2-GPT4-train.txt', help='Path to training data. It\'s default to a txt file.')
    parser.add_argument('--val_data', type=str, default = './data/TinyStoriesV2-GPT4-valid.txt', help='Path to validation data. It\'s default to a txt file.')
    # parser.add_argument('--tokenizer_vocab', type=str, required=True, help='Path to tokenizer vocab')
    # parser.add_argument('--tokenizer_merges', type=str, required=True, help='Path to tokenizer merges')
    
    # Device
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', 
                        help='Device to use for training')
    
    args = parser.parse_args()
    
    # If config file is provided, load it and override defaults
    if args.config:
        with open(args.config, 'r') as f:
            config_dict = json.load(f)
        # Update args with config file values (CLI args take precedence)
        for key, value in config_dict.items():
            if not hasattr(args, key) or getattr(args, key) == parser.get_default(key): # not set or it's default
                setattr(args, key, value)
    
    # Convert args to dictionary
    return vars(args)

def save_tokens_to_memmap_chunked(tokens_iter, output_path, dtype=np.uint16, chunk_size=1000000):
    """
    Convert an iterator of tokens to a memory-mapped file using chunked writing.
    More memory-efficient for very large datasets.
    
    Args:
        tokens_iter: Iterator yielding token IDs
        output_path: Path to save the binary file
        dtype: Data type for tokens
        chunk_size: Number of tokens to accumulate before writing
    """
    # Open file in binary write mode
    with open(output_path, 'wb') as f:
        chunk = []
        for token in tokens_iter:
            chunk.append(token)
            if len(chunk) >= chunk_size:
                # Write chunk to file
                arr = np.array(chunk, dtype=dtype)
                arr.tofile(f)
                chunk = []
        # Write remaining tokens
        if chunk:
            arr = np.array(chunk, dtype=dtype)
            arr.tofile(f)
    # Return as memmap
    return np.memmap(output_path, dtype=dtype, mode='r')

def pre_process(config:dict):
    if not config['preprocess_data']:
        pass
    # load tokenizer
    tokenizer = tokenizer_endecoder.TokenizerEnDeCoder.from_files(config['vocab_filepath'], config['merges_filepath'], config['special_token_list'])
    print('Created the tokenizer from files.')
    # load training data input path
    with open(config['train_data'], 'r', encoding='utf-8') as f:
        train_text = f.read()
    with open(config['val_data'], 'r', encoding='utf-8') as f:
        val_text = f.read()
    print('Loaded the train and val data txt files.')
    # tokenize
    train_tokens = tokenizer.encode_iterable(train_text)
    val_tokens = tokenizer.encode_iterable(val_text)
    save_tokens_to_memmap_chunked(train_tokens, config['preprocess_train_data'])
    save_tokens_to_memmap_chunked(val_tokens, config['preprocess_val_data'])
    print('save the tokens into memmap.')


def main_training(config: dict):
    """
    Ability to configure and control the various model and optimizer hyperparameters (in CLI).
    • Memory-efficient loading of training and validation large datasets with np.memmap.
    • Serializing checkpoints to a user-provided path.
    • Periodically logging training and validation performance (e.g., to console and/or an external
    service like Weights and Biases)
    """
    # CLI and load hyper parameters
    #   - data: batch_size, context_len
    #   - model: vocab_size:int, context_length:int, num_layers:int, d_model:int, num_heads:int, d_ff:int, theta:float = None
    #   - optmizer: lr: float = 1e-3, betas = (0.9, 0.95), eps = 1e-8, weight_decay = 1e-2
    #       - learning rate schedule: it, max_learning_rate, min_learning_rate, warmup_iters, cosine_cycle_iters
    #       - gradient_clipping
    #   - loss: 
    # load data
    # set up model, opt
    # for batch
    # model forward
    # compute loss
    # loss backward
    # opt step
    
    # Load np.memmap from files as train and val tokens.
    print(f"started the main training step...\n")
    train_data = np.memmap(config['preprocess_train_data'], dtype=np.uint16, mode='r')
    val_data = np.memmap(config['preprocess_val_data'], dtype=np.uint16, mode='r')
    print(f"Loaded {len(train_data)} training tokens")
    print(f"Loaded {len(val_data)} validation tokens")
    print(f"Initing the transfomer large models...")
    model = transformer_utils.MyTransfomerLM(
        vocab_size=config['vocab_size'],
        context_length=config['context_length'],
        num_layers=config['num_layers'],
        d_model=config['d_model'],
        num_heads=config['num_heads'],
        d_ff=config['d_ff'],
        theta=config.get('theta', 10000.0)
    ).to(config['device'])
    opt = transformer_utils.myAdamW(
        model.parameters(),
        config['learning_rate'],
        config['betas'],
        config['eps'],
        config['weight_decay']
    )
    # training loop
    for iter in range(config['max_iters']):
        # Zero gradients from previous iteration
        opt.zero_grad()
        
        # Load batch
        batch_training_data, batch_training_label = data_loading(train_data, config['batch_size'], config['contex_length'], config['device'])
        batch_val_data, batch_val_label = data_loading(val_data, config['batch_size'], config['contex_length'], config['device'])
        # forwards
        logits = model.forward(batch_training_data)
        
        # Compute loss
        loss = transformer_utils.my_cross_entropy(logits.view(-1, config['vocab_size']), batch_training_label.view(-1,))
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        transformer_utils.gradient_clipping(model.parameters(), config['max_l2_norm'])
        
        # Update learning rate schedule
        for group in opt.param_groups:
            group['lr'] = transformer_utils.learning_rate_schedule(iter,
                                                                   config['max_learning_rate'],
                                                                   config['min_learning_rate'],
                                                                   config['warmup_iters'],
                                                                   config['cosine_cycle_iters'])
        
        # Optimizer step
        opt.step()
        # log loss
        if iter % config['eval_interval'] == 0:
            print(f"iteration {iter}: loss {loss}")
        # output ckpt
        if iter % config['checkpoint_interval'] == 0:
            checkpoint_path = os.path.join(config['checkpoint_dir'], f'checkpoint_{iter}.ckpt')
            save_checkpoint(model, opt, iter, checkpoint_path)
    print("Training Complete.")

if __name__ == '__main__':
    config = parse_training_args()
    pre_process()
    main_training()