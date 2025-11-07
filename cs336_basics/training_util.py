from typing import Any
import torch
import numpy as np
import os
import typing
import sys
import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import transformer_utils
import tokenizer_endecoder
import argparse
import json
import matplotlib.pyplot as plt

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
    
    # Convert to torch tensors (long type for embedding lookup) and move to device
    inputs_tensor = torch.from_numpy(inputs_array).long().to(device)
    labels_tensor = torch.from_numpy(labels_array).long().to(device)
    
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

def evaluate_validation_loss(model: torch.nn.Module, val_data: np.ndarray, config: dict, num_batches: int = 10) -> float:
    """
    Evaluate validation loss over multiple batches.
    
    Args:
        model: The model to evaluate
        val_data: Validation dataset (memmap array)
        config: Configuration dictionary with batch_size, context_length, device, vocab_size
        num_batches: Number of batches to average over
    
    Returns:
        Average validation loss
    """
    model.eval()  # Set model to evaluation mode
    total_loss = 0.0
    
    with torch.no_grad():  # Disable gradient computation
        for _ in range(num_batches):
            batch_val_data, batch_val_label = data_loading(
                val_data, 
                config['batch_size'], 
                config['context_length'], 
                config['device']
            )
            logits = model.forward(batch_val_data)
            loss = transformer_utils.my_cross_entropy(
                logits.view(-1, config['vocab_size']), 
                batch_val_label.view(-1)
            )
            total_loss += loss.item()
    
    model.train()  # Set model back to training mode
    return total_loss / num_batches

def plot_losses(train_losses: list, val_losses: list, iterations: list, save_path: str):
    """
    Plot training and validation losses and save to file.
    
    Args:
        train_losses: List of training losses
        val_losses: List of validation losses
        iterations: List of iteration numbers
        save_path: Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, train_losses, label='Training Loss', linewidth=2)
    plt.plot(iterations, val_losses, label='Validation Loss', linewidth=2)
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training and Validation Loss', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Loss plot saved to {save_path}")

def parse_training_args():
    """Parse command-line arguments for training configuration."""
    parser = argparse.ArgumentParser(description='Train a transformer language model')
    
    # Option 1: Load all params from a config file
    parser.add_argument('--config', type=str, default = '', help='Path to JSON config file with all parameters')
    
    # Option 2: Individual CLI arguments
    # Preprocessing flag
    parser.add_argument('--preprocess-data', dest='preprocess_data', action='store_true', help='Tokenize text data and save to memmap files')
    parser.add_argument('--no-preprocess-data', dest='preprocess_data', action='store_false', help='Skip preprocessing, use existing memmap files')
    parser.set_defaults(preprocess_data=False)  # Default: skip preprocessing
    # Data parameters
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--context_length', type=int, default=256, help='Context length for sequences')
    
    # Tokenizer parameters 
    # './tokenizer_output/tinystory_train_10000vocab/vocab.json', './tokenizer_output/tinystory_train_10000vocab/merges.txt', ['<|endoftext|>']
    parser.add_argument('--vocab_filepath', type=str, default='./tokenizer_output/tinystory_train_10000vocab/vocab.json', help = 'The input file for vocab for the tokenizer.')
    parser.add_argument('--merges_filepath', type=str, default= './tokenizer_output/tinystory_train_10000vocab/merges.txt', help = 'The merge rules for the tokenizer.')
    parser.add_argument('--special_token_list', type=list, default = ['<|endoftext|>'], help = 'The list of special tokens for the tokenizer.')

    # Model parameters
    parser.add_argument('--vocab_size', type=int, default=10000, help='Vocabulary size')
    parser.add_argument('--num_layers', type=int, default=4, help='Number of transformer layers')
    parser.add_argument('--d_model', type=int, default=256, help='Model dimension')
    parser.add_argument('--num_heads', type=int, default=4, help='Number of attention heads')
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
    parser.add_argument('--cosine_cycle_iters', type=int, default=9000, help='Cosine cycle iterations')
    
    # Gradient clipping
    parser.add_argument('--max_l2_norm', type=float, default=1.0, help='Max L2 norm for gradient clipping')
    
    # Training parameters
    parser.add_argument('--max_iters', type=int, default=10000, help='Maximum training iterations')
    parser.add_argument('--eval_interval', type=int, default=100, help='Evaluation interval')
    parser.add_argument('--checkpoint_interval', type=int, default=500, help='Checkpoint save interval')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--plot_dir', type=str, default='./plots', help='Directory to save loss plots')
    parser.add_argument('--num_val_batches', type=int, default=10, help='Number of validation batches to average for validation loss')
    
    # Data paths
    parser.add_argument('--preprocess_train_data', type=str, default = './data/preprocess/tinystory/train_tokens.bin', help='Path to preprocessed train data in bin.')
    parser.add_argument('--preprocess_val_data', type=str, default = './data/preprocess/tinystory/val_tokens.bin', help='Path to preprocessed val data in bin.')
    parser.add_argument('--train_data', type=str, default = './data/TinyStoriesV2-GPT4-train.txt', help='Path to training data. It\'s default to a txt file.')
    parser.add_argument('--val_data', type=str, default = './data/TinyStoriesV2-GPT4-valid.txt', help='Path to validation data. It\'s default to a txt file.')
    
    # Testing parameters
    parser.add_argument('--test_mode', action='store_true', help='Enable test mode to load only a small portion of data')
    parser.add_argument('--test_data_size', type=int, default=10000, help='Number of characters to load in test mode')
    # parser.add_argument('--tokenizer_vocab', type=str, required=True, help='Path to tokenizer vocab')
    # parser.add_argument('--tokenizer_merges', type=str, required=True, help='Path to tokenizer merges')
    
    # Device
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', 
                        help='Device to use for training')
    
    # Parse args to get config file path first
    args, remaining_argv = parser.parse_known_args()
    
    # If config file is provided, load it and set as defaults
    if args.config:
        with open(args.config, 'r') as f:
            config_dict = json.load(f)
        # Set config values as new defaults
        parser.set_defaults(**config_dict)
    
    # Parse all args again (CLI args will override config defaults)
    args = parser.parse_args()
    
    # Convert args to dictionary
    return vars(args)

def save_tokens_to_memmap(tokens_iter, output_path, dtype=np.uint16):
    """
    Convert an iterator of tokens to a memory-mapped file.
    
    Args:
        tokens_iter: Iterator yielding token IDs
        output_path: Path to save the binary file
        dtype: Data type for tokens (uint16 supports vocab up to 65536)
    
    Returns:
        np.memmap: Memory-mapped array of tokens
    """
    # Collect all tokens from iterator into a list
    tokens_list = list[Any](tokens_iter)
    # Convert to numpy array
    tokens_array = np.array(tokens_list, dtype=dtype)
    # Save to binary file
    tokens_array.tofile(output_path)
    # Return as memmap (read-only mode for training)
    return np.memmap(output_path, dtype=dtype, mode='r')

def save_tokens_to_memmap_chunked(tokens_iter, output_path, dtype=np.uint16, chunk_size=1000000):
    """
    Convert an iterator of tokens to a memory-mapped file using chunked writing.
    More memory-efficient for very large datasets.
    
    NOTE: If tokens_iter is a generator from encode_iterable, the actual tokenization
    happens lazily as we iterate through it here (not when the generator was created).
    
    Args:
        tokens_iter: Iterator yielding token IDs
        output_path: Path to save the binary file
        dtype: Data type for tokens
        chunk_size: Number of tokens to accumulate before writing
    """
    # Open file in binary write mode
    print(f"Tokenizing and saving to {output_path}...")
    print("(Tokenization happens during iteration - this is the slow part)\n")
    start_time = time.time()
    
    with open(output_path, 'wb') as f:
        chunk = []
        chunk_count = 0
        total_tokens = 0
        
        for token in tokens_iter:
            chunk.append(token)
            if len(chunk) >= chunk_size:
                # Write chunk to file
                arr = np.array(chunk, dtype=dtype)
                arr.tofile(f)
                total_tokens += len(chunk)
                chunk_count += 1
                chunk = []
                # Progress update every chunk
                elapsed = time.time() - start_time
                speed = total_tokens / elapsed if elapsed > 0 else 0
                print(f'  Chunk {chunk_count}: {total_tokens:,} tokens processed ({speed:.0f} tokens/sec)')
        
        # Write remaining tokens
        if chunk:
            arr = np.array(chunk, dtype=dtype)
            arr.tofile(f)
            total_tokens += len(chunk)
            print(f'  Final: {len(chunk):,} remaining tokens written')
    
    elapsed_time = time.time() - start_time
    print(f"✓ Saved {total_tokens:,} tokens to {output_path} in {elapsed_time:.1f}s ({total_tokens/elapsed_time:.0f} tokens/sec)\n")
    
    # Return as memmap
    return np.memmap(output_path, dtype=dtype, mode='r')

def pre_process(config:dict, save_in_chunk:bool = True):
    if not config['preprocess_data']:
        return
    
    # Ensure output directories exist
    os.makedirs(os.path.dirname(config['preprocess_train_data']), exist_ok=True)
    os.makedirs(os.path.dirname(config['preprocess_val_data']), exist_ok=True)
    
    # load tokenizer
    print('='*60)
    print('PREPROCESSING DATA')
    print('='*60)
    tokenizer = tokenizer_endecoder.TokenizerEnDeCoder.from_files(
        config['vocab_filepath'], 
        config['merges_filepath'], 
        config['special_token_list'],
        use_fast_merge=True  # Use optimized O(n log m) algorithm - MUCH faster!
    )
    print('✓ Created tokenizer from files (using fast merge algorithm)\n')
    
    # load training data input path
    print('Loading text files...')
    train_file_size = os.path.getsize(config['train_data']) / (1024**2)  # MB
    val_file_size = os.path.getsize(config['val_data']) / (1024**2)  # MB
    
    with open(config['train_data'], 'r', encoding='utf-8') as f:
        if config.get('test_mode', False):
            # In test mode, only read a small portion
            train_text = f.read(config['test_data_size'])
            print(f'  Test mode: loaded first {config["test_data_size"]:,} characters from train data')
        else:
            train_text = f.read()
            print(f'  Train: {config["train_data"]} ({train_file_size:.1f} MB, {len(train_text):,} characters)')
            
    with open(config['val_data'], 'r', encoding='utf-8') as f:
        if config.get('test_mode', False):
            # In test mode, only read a small portion
            val_text = f.read(config['test_data_size'])
            print(f'  Test mode: loaded first {config["test_data_size"]:,} characters from val data')
        else:
            val_text = f.read()
            print(f'  Val:   {config["val_data"]} ({val_file_size:.1f} MB, {len(val_text):,} characters)')
    
    print(f'\n✓ Loaded text files\n')
    
    # Note: encode_iterable returns a generator (lazy evaluation)
    # The actual tokenization happens when we iterate through it during saving
    print('Creating tokenization iterators...')
    train_tokens = tokenizer.encode_iterable(train_text, verbose=True, log_interval=500)
    val_tokens = tokenizer.encode_iterable(val_text, verbose=True, log_interval=100)
    print('✓ Iterators created (tokenization will happen during saving)\n')
    
    # process validation data first to get a idea of total train data processing time.
    print('='*60)
    print('PROCESSING VALIDATION DATA')
    print('='*60)
    if save_in_chunk:
        save_tokens_to_memmap_chunked(val_tokens, config['preprocess_val_data'])
    else:
        save_tokens_to_memmap(val_tokens, config['preprocess_val_data'])
    
    print('='*60)
    print('PROCESSING TRAIN DATA')
    print('='*60)
    if save_in_chunk:
        save_tokens_to_memmap_chunked(train_tokens, config['preprocess_train_data'])
    else:
        save_tokens_to_memmap(train_tokens, config['preprocess_train_data'])

    print('='*60)
    print('PREPROCESSING COMPLETE!')
    print('='*60)
    print(f'Train tokens saved to: {config["preprocess_train_data"]}')
    print(f'Val tokens saved to:   {config["preprocess_val_data"]}')
    print('You can now run training with --no-preprocess-data to skip this step.\n')


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
    
    # Ensure checkpoint and plot directories exist
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    os.makedirs(config['plot_dir'], exist_ok=True)
    
    train_data = np.memmap(config['preprocess_train_data'], dtype=np.uint16, mode='r')
    val_data = np.memmap(config['preprocess_val_data'], dtype=np.uint16, mode='r')
    print(f"Loaded {len(train_data)} training tokens")
    print(f"Loaded {len(val_data)} validation tokens")
    print(f"Initing the transformer language model...")
    
    # Calculate model size
    model = transformer_utils.MyTransfomerLM(
        vocab_size=config['vocab_size'],
        context_length=config['context_length'],
        num_layers=config['num_layers'],
        d_model=config['d_model'],
        num_heads=config['num_heads'],
        d_ff=config['d_ff'],
        theta=config.get('theta', 10000.0)
    ).to(config['device'])
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model initialized: {total_params:,} total parameters ({trainable_params:,} trainable)")
    print(f"Model size: {total_params * 4 / 1024**2:.2f} MB (FP32)")
    
    if torch.cuda.is_available() and config['device'] == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Initial GPU memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    opt = transformer_utils.myAdamW(
        model.parameters(),
        config['learning_rate'],
        config['betas'],
        config['eps'],
        config['weight_decay']
    )
    
    # Track losses for plotting
    train_losses = []
    val_losses = []
    loss_iterations = []
    
    # training loop
    for iter in range(config['max_iters']):
        # Zero gradients from previous iteration
        opt.zero_grad()
        
        # Load training batch
        batch_training_data, batch_training_label = data_loading(train_data, config['batch_size'], config['context_length'], config['device'])
        
        # Forward pass
        logits = model.forward(batch_training_data)
        
        # Compute loss
        loss = transformer_utils.my_cross_entropy(logits.view(-1, config['vocab_size']), batch_training_label.view(-1,))
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        transformer_utils.gradient_clipping(model.parameters(), config['max_l2_norm'])
        
        # Update learning rate schedule
        current_lr = transformer_utils.learning_rate_schedule(iter,
                                                              config['max_learning_rate'],
                                                              config['min_learning_rate'],
                                                              config['warmup_iters'],
                                                              config['cosine_cycle_iters'])
        for group in opt.param_groups:
            group['lr'] = current_lr
        
        # Optimizer step
        opt.step()
        
        # Evaluation and logging
        if iter % config['eval_interval'] == 0:
            # Compute validation loss
            val_loss = evaluate_validation_loss(model, val_data, config, config['num_val_batches'])
            
            # Track losses
            train_losses.append(loss.item())
            val_losses.append(val_loss)
            loss_iterations.append(iter)
            
            # GPU memory info (if using CUDA)
            gpu_mem_str = ""
            if torch.cuda.is_available() and config['device'] == 'cuda':
                gpu_mem_allocated = torch.cuda.memory_allocated() / 1024**3  # Convert to GB
                gpu_mem_reserved = torch.cuda.memory_reserved() / 1024**3
                gpu_mem_str = f", GPU Mem: {gpu_mem_allocated:.2f}GB/{gpu_mem_reserved:.2f}GB"
            
            # Print progress
            print(f"Iteration {iter}/{config['max_iters']}: Train Loss = {loss.item():.4f}, Val Loss = {val_loss:.4f}, LR = {current_lr:.6f}{gpu_mem_str}")
        
        # Checkpoint saving
        if iter % config['checkpoint_interval'] == 0 and iter > 0:
            checkpoint_path = os.path.join(config['checkpoint_dir'], f'checkpoint_{iter}.ckpt')
            save_checkpoint(model, opt, iter, checkpoint_path)
            print(f"Checkpoint saved at iteration {iter}")
            
            # Save plot at checkpoint
            if len(train_losses) > 0:
                plot_path = os.path.join(config['plot_dir'], f'loss_plot_iter_{iter}.png')
                plot_losses(train_losses, val_losses, loss_iterations, plot_path)
    
    # Final checkpoint and plot
    final_checkpoint_path = os.path.join(config['checkpoint_dir'], f'checkpoint_final.ckpt')
    save_checkpoint(model, opt, config['max_iters'], final_checkpoint_path)
    print(f"Final checkpoint saved")
    
    if len(train_losses) > 0:
        final_plot_path = os.path.join(config['plot_dir'], 'final_loss_plot.png')
        plot_losses(train_losses, val_losses, loss_iterations, final_plot_path)
    
    print("Training Complete.")

#######################################################
# Test command: uv run python cs336_basics/training_util.py --config test_config.json
#######################################################
if __name__ == '__main__':
    config = parse_training_args()
    pre_process(config, save_in_chunk=True)
    main_training(config)