import torch.nn.functional as F
from sklearn.metrics import precision_recall_curve, auc, roc_curve, roc_auc_score
from layers import MySoftmax
import numpy as np
import torch


# Evaluate on OOD and adversarial attacks
# -------- OOD --------
# - AUROC
# - AUPR
# - FPR95
# - ECE
# -------- Adversarial --------
# - Gaussian noise
# - FGSM
# - PGD

def compare_two_models(model, vdp_model, id_loader, ood_loader, device="cpu"):
        model = model.to(device)
        model_vdp = vdp_model.to(device)
        print("Evaluating models")
        results = evaluate_ood(model, id_loader, ood_loader, vdp=False, device=device)
        print("Evaluating VDP model")
        results_vdp = evaluate_ood(vdp_model, id_loader, ood_loader, vdp=True, device=device)
        print("Evaluating calibration")
        ece, brier = get_calibration(model, id_loader, vdp=False, device=device)
        ece_vdp, brier_vdp = get_calibration(vdp_model, id_loader, vdp=True, device=device)
        print("Evaluating Gaussian noise")
        gaussian_noise = evaluate_with_gaussian_noise(model, id_loader, vdp=False, device=device)
        gaussian_noise_vdp = evaluate_with_gaussian_noise(vdp_model, id_loader, vdp=True, device=device)
        # fgsm, fgsm_vdp = evaluate_with_fgsm(model, model_vdp, id_loader)
        # pgd, pgd_vdp = evaluate_with_pgd(model, model_vdp, id_loader)

        print( results, results_vdp, f"ece {ece}, ece vdp: {ece_vdp}, brier {brier}, brier vdp: {brier_vdp}", gaussian_noise, gaussian_noise_vdp)

def evaluate_with_gaussian_noise(model, id_loader, vdp=True, noise_strengths=[0.01, 0.05, 0.1], device="cpu"):
    ''' Counts the number of predictions that do NOT change under Gaussian noise'''
    results = {n:[] for n in noise_strengths}
    for images, labels in id_loader:
        images = images.to(device)
        labels = labels.to(device)
        if vdp:
             outputs, _, _ = model(images)
        else:
            outputs = model(images)
        preds = outputs.argmax(dim=1)
        for noise_strength in noise_strengths:
            if vdp:
                outputs_noise, _, _ = model(images + torch.randn_like(images, device=device) * noise_strength)
            else:
                 outputs_noise = model(images + torch.randn_like(images, device=device) * noise_strength)
            results[noise_strength].append((outputs_noise.argmax(dim=1) == preds).sum().item())
    for n in noise_strengths:    
        correct = sum(results[n])
        results[n] = correct / len(id_loader.dataset)
    return results

def get_calibration(model, id_loader, vdp=False, n_bins=15, device="cpu"):
    ''' Computes the ECE and Brier score for the model'''    
    model.eval()
    # for the Brier score
    total_brier = 0.0
    total_samples = 0
    num_classes = 1000# len(torch.unique(torch.tensor([label for _, label in id_loader.dataset])))

    # for the ECE
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    bin_correct = torch.zeros(n_bins)
    bin_confidence = torch.zeros(n_bins)
    bin_count = torch.zeros(n_bins)

    for inputs, labels in id_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        if vdp:
            outputs, _, _ = model(inputs)
        else:
            outputs = model(inputs)
    
        probs = F.softmax(outputs, dim=1)

        # Brier score
        one_hot = F.one_hot(labels, num_classes=num_classes).float()
        brier_batch = torch.sum((probs - one_hot) ** 2, dim=1)
        total_brier += brier_batch.sum().item()
        total_samples += labels.size(0)

        confidences, predictions = probs.max(dim=1)
        confidences = confidences.detach().cpu()
        accuracies = predictions.eq(labels).float().detach().cpu()
        # ECE
        for i in range(n_bins):
            mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
            if mask.any():
                bin_correct[i] += accuracies[mask].sum()
                bin_confidence[i] += confidences[mask].sum()
                bin_count[i] += mask.sum()
    
    # Brier score
    total_brier /= total_samples

    # ECE
    non_empty = bin_count > 0
    ece = torch.sum(
        torch.abs((bin_correct[non_empty] / bin_count[non_empty]) -
                  (bin_confidence[non_empty] / bin_count[non_empty])) *
        (bin_count[non_empty] / bin_count.sum())
    ).item()

    return ece, total_brier

def evaluate_ood(model, in_loader, ood_loader, device="cpu", vdp=True):
    """
    Calculate AUROC, AUPR, and FPR at 95% TPR scores for distinguishing 
    in-distribution from out-of-distribution data using various uncertainty 
    metrics from a variational deep learning model.
    
    Args:
        model: The VDP model with mu and sigma outputs
        in_loader: DataLoader for in-distribution data
        ood_loader: DataLoader for out-of-distribution data
        device: Device to run computation on (CPU or GPU)
    
    Returns:
        Dictionary of metrics (AUROC, AUPR, FPR95) for different uncertainty measures
    """
    
    # Define function to extract all metrics from a single data loader
    def extract_metrics(loader, device):
        max_output, entropy = [], []
        for inputs, _ in loader:
            inputs = inputs.to(device)
            logits = model(inputs)
            probs = F.softmax(logits, dim=1)
            entropy.append(calculate_entropy(probs).detach().cpu().numpy())
            max_output.append(torch.max(logits, 1)[0].detach().cpu().numpy())
        return {
            'max_output': np.concatenate(max_output),
            'entropy': np.concatenate(entropy)
        }
    
    def compute_mc_uncertainties(mu, var, n_samples=10):
        """
        mu: [B, C] - mean logits
        var: [B, C] - variance logits (diagonal)
        Returns: predictive entropy, aleatoric uncertainty, epistemic uncertainty (shape: [B])
        """
        std = torch.sqrt(var)
        B, C = mu.shape
        eps = torch.randn(n_samples, B, C, device=mu.device)  # [N, B, C]
        logits_samples = mu.unsqueeze(0) + std.unsqueeze(0) * eps  # [N, B, C]
        probs_samples = F.softmax(logits_samples, dim=-1) # [N, B, C]
        mean_probs = probs_samples.mean(dim=0)
        predictive_entropy = -torch.sum(mean_probs * torch.log(mean_probs + 1e-10), dim=-1) #[B]
        entropy_per_sample = -torch.sum(probs_samples * torch.log(probs_samples + 1e-10), dim=-1) #[N, B]
        expected_entropy = entropy_per_sample.mean(dim=0)  # [B]
        epistemic = predictive_entropy - expected_entropy

        return predictive_entropy, expected_entropy, epistemic
    
    def calculate_entropy(probabilities):
        return -torch.sum(probabilities * torch.log(probabilities + 1e-10), dim=1)

    def extract_metrics_vdp(loader, device):
        max_output, mean_output_sigma, max_output_sigma = [], [], []
        entropy, mean_logits_sigma, max_logits_sigma = [], [], []
        full_mc, epistemic_mc, aleatoric_mc = [], [], []
        
        for inputs, _ in loader:
            inputs = inputs.to(device)
            logits_mu, logits_sigma, _ = model(inputs)
            output_mu, output_sigma = MySoftmax()(logits_mu, logits_sigma)
            full, aleatoric, epistemic = compute_mc_uncertainties(logits_mu, logits_sigma, n_samples=100)

            max_output.append(torch.max(output_mu, 1)[0].detach().cpu().numpy())
            mean_output_sigma.append(output_sigma.mean(1).detach().cpu().numpy())
            max_output_sigma.append(output_sigma.gather(1, torch.max(output_mu, 1)[1].unsqueeze(1)).squeeze().detach().cpu().numpy())
            entropy.append(calculate_entropy(output_mu).detach().cpu().numpy())
            mean_logits_sigma.append(logits_sigma.mean(1).detach().cpu().numpy())
            max_logits_sigma.append(logits_sigma.gather(1, torch.max(logits_mu, 1)[1].unsqueeze(1)).squeeze().detach().cpu().numpy())
            full_mc.append(full.detach().cpu().numpy())
            epistemic_mc.append(epistemic.detach().cpu().numpy())
            aleatoric_mc.append(aleatoric.detach().cpu().numpy())
        
        # Concatenate all batches
        return {
            'max_output': np.concatenate(max_output),
            'mean_output_sigma': np.concatenate(mean_output_sigma),
            'max_output_sigma': np.concatenate(max_output_sigma),
            'entropy': np.concatenate(entropy),
            'mean_logits_sigma': np.concatenate(mean_logits_sigma),
            'max_logits_sigma': np.concatenate(max_logits_sigma),
            'full_mc': np.concatenate(full_mc),
            'epistemic_mc': np.concatenate(epistemic_mc),
            'aleatoric_mc': np.concatenate(aleatoric_mc)
        }
    
    if vdp:
        in_metrics = extract_metrics_vdp(in_loader, device)
        ood_metrics = extract_metrics_vdp(ood_loader, device)
    else:
        in_metrics = extract_metrics(in_loader, device)
        ood_metrics = extract_metrics(ood_loader, device)
    
    # Calculate metrics for each uncertainty measure
    results = {}
    for metric_name in in_metrics.keys():
        # Combine data and create labels (0=OOD, 1=in-distribution)
        all_values = np.concatenate([ood_metrics[metric_name], in_metrics[metric_name]])
        all_labels = np.concatenate([np.zeros(len(ood_metrics[metric_name])), 
                                     np.ones(len(in_metrics[metric_name]))])
        
        # Determine if we need to invert the metric (some metrics are higher for OOD, others for in-distribution)
        auroc = roc_auc_score(all_labels, all_values)
        invert_metric = auroc < 0.5
        
        # Adjust values if needed (so higher value = more likely to be in-distribution)
        values_for_metrics = -all_values if invert_metric else all_values
        
        # Store the correct AUROC
        results[f"{metric_name}_auroc"] = 1 - auroc if invert_metric else auroc
        
        # Calculate AUPR (Area Under Precision-Recall Curve)
        precision, recall, _ = precision_recall_curve(all_labels, values_for_metrics)
        results[f"{metric_name}_aupr"] = auc(recall, precision)
        
        # Calculate FPR at 95% TPR
        fpr, tpr, thresholds = roc_curve(all_labels, values_for_metrics)
        # Find threshold closest to 95% TPR
        tpr_95_idx = np.argmin(np.abs(tpr - 0.95))
        results[f"{metric_name}_fpr95"] = fpr[tpr_95_idx]
    
    return results


