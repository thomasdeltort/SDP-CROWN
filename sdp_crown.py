import os
import torch
import time
import argparse
import gc

from models import *
from sdp_utils import *
import sys
sys.path.insert(0,'/home/aws_install/robustess_project/SDP-CROWN')
# print(sys.path)
import auto_LiRPA
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm
print(auto_LiRPA.__file__)


def verified_sdp_crown(dataset, labels, model, radius, clean_output, device, classes, args, batch_size=2, return_robust_points=False, x_U=None, x_L=None):
    """
    Args:
        x_U (torch.Tensor, optional): Global upper bound for inputs (e.g., all 1s). 
                                      Shape can be [C, H, W] or [1, C, H, W].
        x_L (torch.Tensor, optional): Global lower bound for inputs (e.g., all 0s).
    """
    model.eval()
    # --- 1. Filter for correctly classified samples ---
    correct_images = dataset[clean_output].to(device)
    correct_labels = labels[clean_output].to(device)

    num_correct_samples = len(correct_images)
    samples = dataset.shape[0] 
    
    if num_correct_samples == 0:
        if return_robust_points: return 0.0, 0.0, torch.tensor([])
        return 0.0, 0.0

    print(f"Verifying {num_correct_samples} samples (Batch Size: {batch_size})...")

    # --- 2. Setup ---
    num_batches = (num_correct_samples + batch_size - 1) // batch_size
    total_time = 0.0
    num_robust_points = 0
    verification_fail_idx = [] 
    robust_indices_list = []


    for i in range(num_batches):
        # MONITOR: Print memory before building the new model
        print(f"--- Batch {i+1} Start ---")
        print(f"Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        print(f"Reserved:  {torch.cuda.memory_reserved() / 1024**2:.2f} MB")

        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, num_correct_samples)
        
        # Get Batch Data
        batch_images = correct_images[start_idx:end_idx]
        batch_labels = correct_labels[start_idx:end_idx]
        original_indices = clean_output[start_idx:end_idx] 
        
        current_batch_size = batch_images.shape[0]

        # --- 3. Prepare Bounds (x_U / x_L) for this Batch ---
        batch_x_U = None
        batch_x_L = None

        # Logic: If x_U is provided, expand it to match the current batch size.
        # Otherwise, fall back to default MNIST logic if applicable.
        
        # Handle x_U
        if x_U is not None:
            batch_x_U = x_U.to(device)
            # If shape is [C, H, W], unsqueeze to [1, C, H, W]
            if batch_x_U.ndim == batch_images.ndim - 1:
                batch_x_U = batch_x_U.unsqueeze(0)
            # Expand [1, C, H, W] to [Batch, C, H, W]
            if batch_x_U.shape[0] == 1:
                batch_x_U = batch_x_U.expand(current_batch_size, -1, -1, -1)
        elif "mnist" in args.model.lower():
            batch_x_U = torch.ones_like(batch_images)

        # Handle x_L
        if x_L is not None:
            batch_x_L = x_L.to(device)
            if batch_x_L.ndim == batch_images.ndim - 1:
                batch_x_L = batch_x_L.unsqueeze(0)
            if batch_x_L.shape[0] == 1:
                batch_x_L = batch_x_L.expand(current_batch_size, -1, -1, -1)
        elif "mnist" in args.model.lower():
            batch_x_L = torch.zeros_like(batch_images)

        # --- 4. Perturbation & Model Build (FRESH PER BATCH) ---
        ptb = PerturbationLpNorm(norm=2.0, eps=radius, x_U=batch_x_U, x_L=batch_x_L)
        image_batch = BoundedTensor(batch_images, ptb)
        
        # Create the model for THIS batch specifically
        # We must rebuild this to clear previous geometric constraints
        lirpa_model = BoundedModule(model, image_batch, device=device, verbose=0)
        
        C = build_C(batch_labels, classes)

        # --- 5. Initialize SDP Optimization ---
        
        lirpa_model.set_bound_opts({'optimize_bound_args': {
            'iteration': 300, 
            'lr_alpha': args.lr_alpha, 
            'early_stop_patience': 20, 
            'fix_interm_bounds': False, 
            'enable_opt_interm_bounds': True, 
            'enable_SDP_crown': True, 
            'lr_lambda': args.lr_lambda,
            # 'lr_decay': 0.999,
        }})
        # lirpa_model.set_bound_opts({'optimize_bound_args': {
        #     'iteration': 500, 
        #     'lr_alpha': 0.5, 
        #     'early_stop_patience': 50, 
        #     'fix_interm_bounds': False, 
        #     'enable_opt_interm_bounds': True, 
        #     'enable_SDP_crown': True, 
        #     'lr_lambda': 0.05,
        #     'lr_decay': 0.98,
        # }})

        # --- 6. Execution ---
        if device.type == 'cuda': torch.cuda.synchronize()
        start_time = time.time()
        
        crown_lb, _ = lirpa_model.compute_bounds(x=(image_batch,), method='CROWN-Optimized', C=C, bound_lower=True, bound_upper=False)
        # print(crown_lb)
        if device.type == 'cuda': torch.cuda.synchronize()
        end_time = time.time()
        
        batch_time = end_time - start_time
        total_time += batch_time

        # --- 7. Check Robustness ---
        if isinstance(crown_lb, torch.Tensor):
            # amin(dim=1) checks the smallest margin for each sample
            is_robust_batch = (crown_lb.amin(dim=1) > 0)
        else:
            is_robust_batch = torch.tensor([False] * current_batch_size, device=device)
        
        num_robust_points += is_robust_batch.sum().item()
        
        if return_robust_points:
             robust_indices_list.append(original_indices[is_robust_batch])
             
        print(f"Batch {i+1}/{num_batches}: {is_robust_batch.sum().item()}/{current_batch_size} verified. Time: {batch_time:.2f}s")
        
        # Memory Cleanup 
        del lirpa_model
        del image_batch
        del ptb
        del C
        del crown_lb  # If this tensor is attached to the graph, it keeps the graph alive!
            
        # Force Python's Garbage Collector to destroy the objects
        gc.collect()
            
        # Force PyTorch to release the freed memory back to the GPU
        torch.cuda.empty_cache()
            
        print(f"Batch {i+1} Cleaned. Memory: {torch.cuda.memory_allocated()/1024**2:.2f} MB")

    # --- 8. Final Metrics ---
    verification_fail_count = (samples - num_correct_samples) + (num_correct_samples - num_robust_points)
    verified_accuracy = (num_robust_points / samples) * 100.0
    average_time = total_time / num_correct_samples

    print(f'Total Verification Fail: {verification_fail_count}')
    print(f'Verified Accuracy: {verified_accuracy:.2f}%')
    print(f'Average Time: {average_time:.4f}s')
    
    if return_robust_points:
        all_robust_indices = torch.cat(robust_indices_list) if robust_indices_list else torch.tensor([])
        return verified_accuracy, average_time, all_robust_indices
        
    return verified_accuracy, average_time


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--radius', default=1, type=parse_float_or_fraction, help='L2 norm perturbation')
    parser.add_argument('--lr_alpha', default=0.5, type=float, help='alpha learning rate')
    parser.add_argument('--lr_lambda', default=0.05, type=float, help='lambda learning rate')
    parser.add_argument('--start', default=0, type=int, help='start index for the dataset')
    parser.add_argument('--end', default=200, type=int, help='end index for the dataset')
    parser.add_argument('--model', default='mnist_mlp',
    choices=[
        'mnist_mlp',
        'mnist_convsmall',
        'mnist_convlarge',
        'cifar10_cnn_a',
        'cifar10_cnn_b',
        'cifar10_cnn_c',
        'cifar10_convsmall',
        'cifar10_convdeep',
        'cifar10_convlarge',
        ])
    args = parser.parse_args()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model, dataset, labels, radius_rescale, classes = load_model_and_dataset(args, device)

    # Run original model for clean accuracy.
    with torch.no_grad():
        labels_tensor = labels.to(device)
        dataset_tensor = dataset.to(device)
        output = model(dataset_tensor)
        clean_output = torch.sum((output.max(1)[1] == labels_tensor).float()).cpu()
        predictions = output.argmax(dim=1)
        correct_indices = (predictions == labels_tensor).nonzero(as_tuple=True)[0]
    print(f'perturbation: {radius_rescale}')
    print(f'The clean output for the {args.end-args.start} samples is {clean_output/(args.end-args.start)*100}%')
    
    verified_sdp_crown(
        dataset = dataset, 
        labels = labels, 
        model = model, 
        radius = radius_rescale, 
        clean_output = correct_indices, 
        device = device, 
        classes = classes, 
        args = args
        )
