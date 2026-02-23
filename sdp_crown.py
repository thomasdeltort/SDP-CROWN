import os
import torch
import time
import argparse
import gc

from models import *
from sdp_utils import *
import sys
sys.path.insert(0,'..//SDP-CROWN')
# print(sys.path)
import auto_LiRPA
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm
print(auto_LiRPA.__file__)


def verified_sdp_crown(dataset, labels, model, radius, clean_output, device, classes, args, batch_size=1, return_robust_points=False, x_U=None, x_L=None, groupsort=False):
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
        
      
        if args.high_tau :
            lirpa_model.set_bound_opts({
            'optimize_bound_args': {
                'iteration': 1000,
                'lr_alpha': 0.5, 
                'lr_lambda': 0.5,
                'lr_decay': 0.998,
                'early_stop_patience': 100,
                'fix_interm_bounds': False,         # Change to True to stabilize the final output
                'enable_opt_interm_bounds': True, # Use pre-computed bounds for speed
                'enable_SDP_crown': True,               # Ensure Adam is used for dual variables
                }
            })
        else:
            lirpa_model.set_bound_opts({'optimize_bound_args': {
            'iteration': 300, 
            'lr_alpha': args.lr_alpha, 
            'early_stop_patience': 20, 
            'fix_interm_bounds': False, 
            'enable_opt_interm_bounds': True, 
            'enable_SDP_crown': True, 
            'lr_lambda': args.lr_lambda,
        }})

        

        # --- 6. Execution ---
        if device.type == 'cuda': torch.cuda.synchronize()
        start_time = time.time()
        
        crown_lb, _ = lirpa_model.compute_bounds(x=(image_batch,), method='CROWN-Optimized', C=C, bound_lower=True, bound_upper=False, groupsort=groupsort)
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



# def verified_sdp_crown_rescaled(dataset, labels, model, radius, clean_output, device, classes, args, batch_size=2, return_robust_points=False, x_U=None, x_L=None, groupsort=False):
#     """
#     Args:
#         x_U (torch.Tensor, optional): Global upper bound for inputs.
#         x_L (torch.Tensor, optional): Global lower bound for inputs.
#     """
#     model.eval()

#     # --- 1. SCALE FACTOR SETUP ---
#     SCALING_FACTOR = 0.01
#     last_layer = None
#     original_weight = None
#     original_bias = None

#     # Find the last weighted layer (Linear or Conv)
#     # We iterate backwards to find the layer producing the logits
#     for m in reversed(list(model.modules())):
#         if isinstance(m, (nn.Linear, nn.Conv2d)):
#             last_layer = m
#             break
            
#     if last_layer is not None:
#         print(f"[Numerical Stability] Scaling last layer ({type(last_layer).__name__}) weights/bias by {SCALING_FACTOR}...")
#         # Save original data to restore later
#         original_weight = last_layer.weight.data.clone()
#         if last_layer.bias is not None:
#             original_bias = last_layer.bias.data.clone()
        
#         # Apply scaling in-place
#         last_layer.weight.data *= SCALING_FACTOR
#         if last_layer.bias is not None:
#             last_layer.bias.data *= SCALING_FACTOR
#     else:
#         print("Warning: Could not find a valid last layer to scale. Proceeding without scaling.")

#     # --- 2. Filter for correctly classified samples ---
#     # Note: We rely on the clean_output passed in, assuming it was computed *before* scaling.
#     correct_images = dataset[clean_output].to(device)
#     correct_labels = labels[clean_output].to(device)

#     num_correct_samples = len(correct_images)
#     samples = dataset.shape[0] 
    
#     if num_correct_samples == 0:
#         # Restore before returning
#         if last_layer is not None:
#             last_layer.weight.data = original_weight
#             if original_bias is not None:
#                 last_layer.bias.data = original_bias
#         if return_robust_points: return 0.0, 0.0, torch.tensor([])
#         return 0.0, 0.0

#     print(f"Verifying {num_correct_samples} samples (Batch Size: {batch_size})...")

#     # --- 3. Setup ---
#     num_batches = (num_correct_samples + batch_size - 1) // batch_size
#     total_time = 0.0
#     num_robust_points = 0
#     robust_indices_list = []

#     try: # Try-block ensures we restore model weights even if code crashes
#         for i in range(num_batches):
#             print(f"--- Batch {i+1} Start ---")
            
#             start_idx = i * batch_size
#             end_idx = min((i + 1) * batch_size, num_correct_samples)
            
#             # Get Batch Data
#             batch_images = correct_images[start_idx:end_idx]
#             batch_labels = correct_labels[start_idx:end_idx]
#             original_indices = clean_output[start_idx:end_idx] 
            
#             current_batch_size = batch_images.shape[0]

#             # --- 4. Prepare Bounds (x_U / x_L) ---
#             batch_x_U = None
#             batch_x_L = None
            
#             if x_U is not None:
#                 batch_x_U = x_U.to(device)
#                 if batch_x_U.ndim == batch_images.ndim - 1:
#                     batch_x_U = batch_x_U.unsqueeze(0)
#                 if batch_x_U.shape[0] == 1:
#                     batch_x_U = batch_x_U.expand(current_batch_size, -1, -1, -1)
#             elif args.model and "mnist" in args.model.lower():
#                 batch_x_U = torch.ones_like(batch_images)

#             if x_L is not None:
#                 batch_x_L = x_L.to(device)
#                 if batch_x_L.ndim == batch_images.ndim - 1:
#                     batch_x_L = batch_x_L.unsqueeze(0)
#                 if batch_x_L.shape[0] == 1:
#                     batch_x_L = batch_x_L.expand(current_batch_size, -1, -1, -1)
#             elif args.model and "mnist" in args.model.lower():
#                 batch_x_L = torch.zeros_like(batch_images)

#             # --- 5. Perturbation & Model Build ---
#             ptb = PerturbationLpNorm(norm=2.0, eps=radius, x_U=batch_x_U, x_L=batch_x_L)
#             image_batch = BoundedTensor(batch_images, ptb)
            
#             # Build BoundedModule (Now uses the scaled model)
#             lirpa_model = BoundedModule(model, image_batch, device=device, verbose=0)
            
#             # Build C specification matrix
#             # IMPORTANT: C is purely logical (classes), it doesn't need scaling.
#             # The scaling is inside the model weights.
#             # C = BoundedModule.make_final_node_specification(batch_labels, classes)
#             C = build_C(batch_labels, classes)
            
#             # --- 6. Initialize SDP Optimization ---
#             # Using the "Best Practice" config for stability
#             lirpa_model.set_bound_opts({
#             'optimize_bound_args': {
#                 'iteration': 1000,
#                 'lr_alpha': 0.5, 
#                 'lr_lambda': 0.5,
#                 'lr_decay': 0.998,
#                 'early_stop_patience': 100,
#                 'fix_interm_bounds': False,         # Change to True to stabilize the final output
#                 'enable_opt_interm_bounds': True, # Use pre-computed bounds for speed
#                 'enable_SDP_crown': True,               # Ensure Adam is used for dual variables
#                 }
#             })
#             # lirpa_model.set_bound_opts({
#             # 'optimize_bound_args': {
#             #     'iteration': 1000,
#             #     'lr_alpha': 0.5, 
#             #     'lr_lambda': 0.5,
#             #     'lr_decay': 0.998,
#             #     'early_stop_patience': 100,
#             #     'fix_interm_bounds': False,         # Change to True to stabilize the final output
#             #     'enable_opt_interm_bounds': True, # Use pre-computed bounds for speed
#             #     'enable_SDP_crown': True,               # Ensure Adam is used for dual variables
#             #     }
#             # })

#             # --- 7. Execution ---
#             if device.type == 'cuda': torch.cuda.synchronize()
#             start_time = time.time()
            
#             # Compute bounds
#             crown_lb, _ = lirpa_model.compute_bounds(
#                 x=(image_batch,), 
#                 method='CROWN-Optimized', 
#                 C=C, 
#                 bound_lower=True, 
#                 bound_upper=False
#             )
            
#             if device.type == 'cuda': torch.cuda.synchronize()
#             end_time = time.time()
#             batch_time = end_time - start_time
#             total_time += batch_time

#             # --- 8. Check Robustness & Descale ---
#             if isinstance(crown_lb, torch.Tensor):
#                 # DESCALE THE CERTIFICATE
#                 # Since model output was scaled by 100, the margin is also scaled by 100.
#                 crown_lb = crown_lb / SCALING_FACTOR
                
#                 # Check smallest margin
#                 is_robust_batch = (crown_lb.amin(dim=1) > 0)
#             else:
#                 is_robust_batch = torch.tensor([False] * current_batch_size, device=device)
            
#             num_robust_points += is_robust_batch.sum().item()
            
#             if return_robust_points:
#                  robust_indices_list.append(original_indices[is_robust_batch])
                 
#             print(f"Batch {i+1}/{num_batches}: {is_robust_batch.sum().item()}/{current_batch_size} verified. Time: {batch_time:.2f}s")
            
#             # Memory Cleanup 
#             del lirpa_model, image_batch, ptb, C, crown_lb
#             gc.collect()
#             torch.cuda.empty_cache()

#     finally:
#         # --- 9. Restore Model Weights ---
#         if last_layer is not None:
#             print("[Numerical Stability] Restoring original model weights...")
#             last_layer.weight.data = original_weight
#             if original_bias is not None:
#                 last_layer.bias.data = original_bias

#     # --- 10. Final Metrics ---
#     verification_fail_count = (samples - num_correct_samples) + (num_correct_samples - num_robust_points)
#     verified_accuracy = (num_robust_points / samples) * 100.0
#     average_time = total_time / num_correct_samples if num_correct_samples > 0 else 0

#     print(f'Total Verification Fail: {verification_fail_count}')
#     print(f'Verified Accuracy: {verified_accuracy:.2f}%')
#     print(f'Average Time: {average_time:.4f}s')
    
#     if return_robust_points:
#         all_robust_indices = torch.cat(robust_indices_list) if robust_indices_list else torch.tensor([])
#         return verified_accuracy, average_time, all_robust_indices
        
#     return verified_accuracy, average_time

# import torch
# import gc
# import time
# from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm

# def verified_sdp_crown_fixed_bs(dataset, labels, model, radius, clean_output, device, classes, args, batch_size=2, return_robust_points=False, x_U=None, x_L=None, groupsort=False):
    model.eval()
    
    # --- 1. Filter for correctly classified samples ---
    correct_images = dataset[clean_output].to(device)
    correct_labels = labels[clean_output].to(device)

    num_correct_samples = len(correct_images)
    samples = dataset.shape[0] 
    
    if num_correct_samples == 0:
        if return_robust_points: return 0.0, 0.0, torch.tensor([])
        return 0.0, 0.0

    print(f"Verifying {num_correct_samples} samples (Variable Batch Size)...")

    # --- 2. Setup ---
    num_batches = (num_correct_samples + batch_size - 1) // batch_size
    total_time = 0.0
    num_robust_points = 0
    robust_indices_list = []

    for i in range(num_batches):
        # Clear previous batch memory explicitly before starting new builds
        gc.collect()
        torch.cuda.empty_cache()
        
        print(f"--- Batch {i+1}/{num_batches} Start ---")
        print(f"Memory Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")

        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, num_correct_samples)
        
        # Slicing and Detaching to ensure no graph carry-over
        batch_images = correct_images[start_idx:end_idx].detach().clone()
        batch_labels = correct_labels[start_idx:end_idx].detach().clone()
        original_indices = clean_output[start_idx:end_idx] 
        current_batch_size = batch_images.shape[0]

        # --- 3. Prepare Bounds (x_U / x_L) ---
        batch_x_U = None
        batch_x_L = None

        if x_U is not None:
            batch_x_U = x_U.to(device)
            if batch_x_U.ndim == batch_images.ndim - 1:
                batch_x_U = batch_x_U.unsqueeze(0)
            if batch_x_U.shape[0] == 1:
                batch_x_U = batch_x_U.expand(current_batch_size, *batch_images.shape[1:])
        elif "mnist" in args.model.lower():
            batch_x_U = torch.ones_like(batch_images)

        if x_L is not None:
            batch_x_L = x_L.to(device)
            if batch_x_L.ndim == batch_images.ndim - 1:
                batch_x_L = batch_x_L.unsqueeze(0)
            if batch_x_L.shape[0] == 1:
                batch_x_L = batch_x_L.expand(current_batch_size, *batch_images.shape[1:])
        elif "mnist" in args.model.lower():
            batch_x_L = torch.zeros_like(batch_images)

        # --- 4. Fresh Model Build per Batch ---
        # We pass a dummy of current_batch_size to ensure the BoundedModule 
        # is strictly defined by the current batch shape.
        ptb = PerturbationLpNorm(norm=2.0, eps=radius, x_U=batch_x_U, x_L=batch_x_L)
        image_batch = BoundedTensor(batch_images, ptb)
        
        lirpa_model = BoundedModule(model, torch.zeros_like(batch_images), device=device, verbose=0)
        
        # Build C based on current batch labels only
        C = build_C(batch_labels, classes)

        # --- 5. Optimization Settings ---
        lirpa_model.set_bound_opts({
            'optimize_bound_args': {
                'iteration': 1000,
                'lr_alpha': 0.5, 
                'lr_lambda': 0.5,
                'lr_decay': 0.998,
                'early_stop_patience': 100,
                'fix_interm_bounds': False, 
                'enable_opt_interm_bounds': True, 
                'enable_SDP_crown': True,
            }
        })

        # --- 6. Execution ---
        if device.type == 'cuda': torch.cuda.synchronize()
        start_time = time.time()
        
        try:
            crown_lb, _ = lirpa_model.compute_bounds(
                x=(image_batch,), 
                method='CROWN-Optimized', 
                C=C, 
                bound_lower=True, 
                bound_upper=False, 
                groupsort=groupsort
            )
        except RuntimeError as e:
            print(f"Critical error in Batch {i+1}: {e}")
            raise e

        if device.type == 'cuda': torch.cuda.synchronize()
        batch_time = time.time() - start_time
        total_time += batch_time

        # --- 7. Check Robustness ---
        if isinstance(crown_lb, torch.Tensor):
            is_robust_batch = (crown_lb.amin(dim=1) > 0)
        else:
            is_robust_batch = torch.tensor([False] * current_batch_size, device=device)
        
        num_robust_points += is_robust_batch.sum().item()
        
        if return_robust_points:
             robust_indices_list.append(original_indices[is_robust_batch])
             
        print(f"Batch {i+1}/{num_batches}: {is_robust_batch.sum().item()}/{current_batch_size} verified. Time: {batch_time:.2f}s")
        
        # --- 8. Aggressive Cleanup ---
        del lirpa_model
        del image_batch
        del ptb
        del C
        del crown_lb
        
    # --- 9. Final Metrics ---
    verification_fail_count = (samples - num_correct_samples) + (num_correct_samples - num_robust_points)
    verified_accuracy = (num_robust_points / samples) * 100.0
    average_time = total_time / num_correct_samples if num_correct_samples > 0 else 0

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
