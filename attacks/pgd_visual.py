import torch
import torch.nn.functional as F

def pgd_attack(model, images, input_ids, epsilon=0.3, alpha=0.03, num_iter=3):
    """
    Projected Gradient Descent (PGD) attack on visual inputs.
    Objective: Diverge from the original model predictions (Maximize KL Divergence).
    
    Args:
        model: VLM model
        images: Preprocessed image tensor [B, C, H, W]
        input_ids: Tokenized input prompt
        epsilon: Maximum perturbation magnitude
        alpha: Step size
        num_iter: Number of iterations
        
    Returns:
        perturbed_images: Adversarial image tensor
    """
    # Clone and detach
    perturbed_images = images.clone().detach()
    perturbed_images.requires_grad = True
    
    # Get clean logits (Target to diverge from)
    with torch.no_grad():
        clean_outputs = model(input_ids, images=images)
        clean_logits = clean_outputs.logits
        
    # Optimization loop
    for _ in range(num_iter):
        perturbed_images.requires_grad = True
        
        # Forward pass
        outputs = model(input_ids, images=perturbed_images)
        logits = outputs.logits
        
        # Loss: KL Divergence (Unreduced first to handle batches properly if needed, but reduction batchmean is fine)
        # We want to maximize the distance between clean and perturbed distributions.
        # F.kl_div(input, target) expects input to be log-probs.
        loss = F.kl_div(
            F.log_softmax(logits, dim=-1),
            F.softmax(clean_logits, dim=-1),
            reduction='batchmean'
        )
        
        # Gradient Ascent: Maximize KL Divergence
        grad = torch.autograd.grad(loss, perturbed_images)[0]
        
        # Update image
        perturbed_images = perturbed_images.detach() + alpha * grad.sign()
        
        # Project back to epsilon ball (L-inf)
        delta = torch.clamp(perturbed_images - images, -epsilon, epsilon)
        
        # Clamp to valid range (Approximate 0-1 range logic, though images are normalized)
        # Ideally we should un-normalize, clamp, and re-normalize, but simpler clamping prevents explosion.
        # Using original min/max as bounds ensures we don't drift too far from valid pixel space scale.
        perturbed_images = torch.clamp(images + delta, images.min(), images.max()).detach()
        
    return perturbed_images
