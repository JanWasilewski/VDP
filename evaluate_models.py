import torch.nn.functional as F
from sklearn.metrics import precision_recall_curve, auc, roc_curve, roc_auc_score
from layers import MySoftmax
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import time
import random
import pandas as pd
from IPython.display import display

# Evaluate on OOD and adversarial attacks
# -------- OOD --------
# - AUROC
# - AUPR
# - FPR95
# -------- Calibration --------
# - ECE
# - Brier
# -------- Adversarial --------
# - Gaussian noise
# - FGSM
# - PGD

def compare_models(models, id_loader, ood_loader, deterministic_idxs=[], noise_strengths=[0.01], num_to_attack=2000, device="cpu"):
        ood_list = []
        calibration_list = []
        gaussian_noise_list = []

        for m in models:
            m.eval()

        for idx, model in enumerate(models):
            model = model.to(device)
            vdp = idx not in deterministic_idxs
            results = evaluate_ood(model, id_loader, ood_loader, vdp=vdp, device=device)
            ece, brier = get_calibration(model, id_loader, vdp=vdp, device=device)
            gaussian_noise = evaluate_with_gaussian_noise(model, id_loader, vdp=vdp, noise_strengths=noise_strengths, device=device)
            ood_list.append(results)
            calibration_list.append((ece, brier))
            gaussian_noise_list.append(gaussian_noise)
            print(f"finished evaluation of {idx}th model")
        fgsm_list = evaluate_with_adversarial_samples(models, id_loader, deterministic_idxs, noise_strengths=noise_strengths, num_to_attack=num_to_attack, attack_type="fgsm", device=device)
        pgd_list = evaluate_with_adversarial_samples(models, id_loader, deterministic_idxs, noise_strengths=noise_strengths, num_to_attack=num_to_attack, attack_type="pgd", device=device)
        
        dfs_fmt, numeric_dfs = compute_top_results(ood_list)
        styled_tables = {}
        for metric, df_fmt in dfs_fmt.items():
            num_df = numeric_dfs[metric]
            def highlight_top1(row, num_df=num_df, metric=metric):
                if row.name != "top1":
                    return [''] * len(row)
                vals = num_df.loc["top1"].values
                # order indices
                if metric in ["AUPR", "AUROC"]:
                    order = (-vals).argsort()
                else:  # FPR95: lowest first
                    order = vals.argsort()
                styles = [''] * len(vals)
                styles[order[0]] = 'color: red'
                styles[order[1]] = 'color: blue'
                return styles

            styled = df_fmt.style.apply(highlight_top1, axis=1)
            styled_tables[metric] = styled
            print(metric)
            display(styled)
        
        display_robustness(gaussian_noise_list, fgsm_list, pgd_list)

        print(calibration_list)
        return ood_list, calibration_list, gaussian_noise_list, fgsm_list, pgd_list

def evaluate_with_adversarial_samples(
    models,
    id_loader,
    deterministic_idxs,
    noise_strengths,
    num_to_attack,
    batch_size=64,
    max_trials=5,
    attack_type='fgsm', 
    pgd_steps=10, 
    pgd_step_size=None,
    device="cpu",
    clip_min=0.0,
    clip_max=1.0,
):
    """
    For each ε in noise_strengths:
      1) Efficiently find num_to_attack points from id_loader.dataset that *all* models 
         classify correctly on the clean input.
      2) For each model, craft FGSM adversarial examples at strength ε
      3) Test each model against adversarial examples from all models
      4) Measure the fraction of those points whose predicted label remains correct.
    
    Args:
        models: List of models to evaluate
        id_loader: DataLoader containing the dataset
        deterministic_idxs: Indices of deterministic models (vs. probabilistic ones)
        noise_strengths: List of epsilon values for FGSM attack
        num_to_attack: Number of samples to attack
        batch_size: Batch size for evaluation
        max_trials: Maximum number of trials to find common correct samples
        device: Device to run evaluation on
        clip_min: Minimum value for input clipping
        clip_max: Maximum value for input clipping
        
    Returns:
      A dictionary of dictionaries:
      { ε: {
          'robustness': [
              [model0_vs_adv0, model0_vs_adv1, ...],
              [model1_vs_adv0, model1_vs_adv1, ...],
              ...
          ]
      }}
    """
    criterion = torch.nn.CrossEntropyLoss()
    dataset = id_loader.dataset
    results = {}
    
    for eps in noise_strengths:
        common_correct_indices = find_common_correct_indices(
            models, dataset, deterministic_idxs, num_to_attack, 
            batch_size, max_trials, device
        )
        
        print(f"ε={eps:.3f}: selected {len(common_correct_indices)} samples")
        
        # --- 2) Generate adversarial examples from each model and test cross-model robustness ---
        attack_subset = Subset(dataset, common_correct_indices)
        attack_loader = DataLoader(attack_subset, batch_size=batch_size, shuffle=False)
        
        cross_robust_counts = [[0 for _ in models] for _ in models]  # model i vs adversarial from j
        total_samples = 0
        
        for X, y in attack_loader:
            X, y = X.to(device), y.to(device)
            batch_size_actual = X.size(0)  # Handle last batch potentially smaller
            total_samples += batch_size_actual
            
            # If we allow misclassified exaples in the future
            # clean_preds = []
            # for idx, model in enumerate(models):
            #     with torch.no_grad():
            #         out = model(X) if idx in deterministic_idxs else model(X)[0]
            #         clean_preds.append(out.argmax(dim=1))
            

            all_adv_examples = []
            for idx, model in enumerate(models):
                model.train()
                X_adv = X.clone().detach().requires_grad_(True)
                if attack_type.lower() == 'fgsm':
                    # single‐step
                    X_adv.requires_grad_(True)
                    for m in models: m.zero_grad()
                    out = model(X_adv) if idx in deterministic_idxs else model(X_adv)[0]
                    loss = criterion(out, y)
                    loss.backward()

                    with torch.no_grad():
                        X_adv = X_adv + eps * X_adv.grad.sign()
                        X_adv = torch.clamp(X_adv, clip_min, clip_max)

                elif attack_type.lower() == 'pgd':
                    alpha = pgd_step_size or (eps / pgd_steps)
                    X_adv = X_adv.requires_grad_(True)
                    for step in range(pgd_steps):
                        for m in models: m.zero_grad()
                        out = model(X_adv) if idx in deterministic_idxs else model(X_adv)[0]
                        loss = criterion(out, y)
                        loss.backward()

                        with torch.no_grad(): # project back into the ball of radius eps around X
                            X_adv = X_adv + alpha * X_adv.grad.sign()
                            delta = torch.clamp(X_adv - X, min=-eps, max=eps)
                            X_adv = torch.clamp(X + delta, clip_min, clip_max)
                        X_adv = X_adv.detach().requires_grad_()
                        

                else:
                    raise ValueError(f"Unknown attack_type: {attack_type!r}")

                all_adv_examples.append(X_adv.detach().cpu())
            
            # Test each model against all adversarial examples
            for target_idx, model in enumerate(models):
                for source_idx, X_adv in enumerate(all_adv_examples):
                    X_adv = X_adv.to(device)
                    out_adv = model(X_adv) if target_idx in deterministic_idxs else model(X_adv)[0]
                    preds_adv = out_adv.argmax(dim=1)
                    correct = (preds_adv == y).sum().item() # Counts where prediction matches ground truth (true robustness)
                    cross_robust_counts[target_idx][source_idx] += correct
    
        cross_robust_accs = [[count / total_samples for count in row] for row in cross_robust_counts]
        results[eps] = {'robustness': cross_robust_accs}
    return results

def find_common_correct_indices(
    models, dataset, deterministic_idxs, num_needed, 
    batch_size, max_trials, device
):
    """
    Efficiently find indices of samples that all models classify correctly.
    
    Args:
        models: List of models
        dataset: The dataset to sample from
        deterministic_idxs: Indices of deterministic models
        num_needed: Number of samples needed
        batch_size: Batch size for processing
        max_trials: Maximum number of trials to attempt
        device: Device to use
        
    Returns:
        List of indices of commonly correct samples
    """
    all_indices = list(range(len(dataset)))
    common_correct = set()
    
    for trial in range(max_trials):
        if len(common_correct) >= num_needed:
            break
        remaining = num_needed - len(common_correct)
        sample_size = min(remaining * 2, len(all_indices) - len(common_correct)) # Sample twice as many as we need to increase chances of finding enough
        available_indices = list(set(all_indices) - common_correct)              # Sample from indices we haven't tried yet
        if not available_indices:
            break
        candidates = random.sample(available_indices, sample_size)
        subset = Subset(dataset, candidates)
        loader = DataLoader(subset, batch_size=batch_size, shuffle=False)
        correct_by_model = [set() for _ in models]
        
        for batch_idx, (X, y) in enumerate(loader):
            X, y = X.to(device), y.to(device)
            base_idx = batch_idx * batch_size
            for model_idx, model in enumerate(models):
                out = model(X) if model_idx in deterministic_idxs else model(X)[0]
                preds = out.argmax(dim=1)
                for i, (pred, label) in enumerate(zip(preds, y)):
                    if pred.item() == label.item():
                        if base_idx + i < len(candidates):  # Guard against batch size issues
                            correct_by_model[model_idx].add(candidates[base_idx + i])
        
        newly_correct = set.intersection(*correct_by_model) # intersection of all correct sets
        common_correct.update(newly_correct)
        print(f"Trial {trial+1}: Found {len(newly_correct)} new common correct samples, "
              f"total: {len(common_correct)}/{num_needed}")
    if len(common_correct) < num_needed:
        print(f"Warning: Could only find {len(common_correct)} common correct samples "
              f"after {max_trials} trials, requested {num_needed}")
    
    return list(common_correct)[:num_needed]


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
    num_classes = len(torch.unique(torch.tensor([label for _, label in id_loader.dataset])))

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


def display_robustness(gaussian_list, fgsm_dict, pgm_dict):
    epsilons = sorted({list(d.keys())[0] for d in gaussian_list})
    Gaussian_df = pd.DataFrame(index=epsilons)
    for i, d in enumerate(gaussian_list, start=1):
        Gaussian_df[f"model{i}"] = [d[eps] for eps in epsilons]
    Gaussian_df.index.name = "epsilon"

    # Build FGSM DataFrame
    epsilons = sorted(fgsm_dict.keys())
    FGSM_df = pd.DataFrame(index=epsilons)
    num_models = len(next(iter(fgsm_dict.values()))['robustness'])
    for j in range(num_models):
        FGSM_df[f"model{j+1}"] = [np.mean(fgsm_dict[eps]['robustness'][j]) for eps in epsilons]
    FGSM_df.index.name = "epsilon"

    # Build PGM DataFrame
    epsilons = sorted(pgm_dict.keys())
    PGM_df = pd.DataFrame(index=epsilons)
    num_models = len(next(iter(pgm_dict.values()))['robustness'])
    for j in range(num_models):
        PGM_df[f"model{j+1}"] = [np.mean(pgm_dict[eps]['robustness'][j]) for eps in epsilons]
    PGM_df.index.name = "epsilon"

    # Styling function: highlight highest in red, second-highest in blue per row
    def style_df_highlights(df):
        def highlight_row(row):
            vals = row.values.astype(float)
            order = (-vals).argsort()  # descending for highest first
            styles = [''] * len(vals)
            if len(vals) > 0:
                styles[order[0]] = 'color: red'
            if len(vals) > 1:
                styles[order[1]] = 'color: blue'
            return styles
        return df.style.apply(highlight_row, axis=1)

    # Apply styling and display
    styled_Gaussian = style_df_highlights(Gaussian_df)
    styled_FGSM = style_df_highlights(FGSM_df)
    styled_PGM = style_df_highlights(PGM_df)

    print("Gaussian")
    display(styled_Gaussian)

    print("FGSM")
    display(styled_FGSM)

    print("PGM")
    display(styled_PGM)


def compute_top_results(results_list, top_n=3, fmt="{:.4f} ({})"):
    metrics = ["aupr", "auroc", "fpr95"]
    orders = {m: (True if m == "fpr95" else False) for m in metrics}
    dfs_fmt = {}
    numeric_dfs = {}
    for m in metrics:
        asc = orders[m]
        idx = [f"top{i+1}" for i in range(top_n)]
        df_fmt = pd.DataFrame(index=idx)
        df_num = pd.DataFrame(index=idx)
        for i, res in enumerate(results_list, start=1):
            pairs = [(float(v), k.rsplit("_",1)[0]) 
                     for k, v in res.items() if k.endswith(f"_{m}")]
            sorted_pairs = sorted(pairs, key=lambda x: x[0], reverse=not asc)[:top_n]
            vals = [v for v, _ in sorted_pairs]
            crits = [crit for _, crit in sorted_pairs]
            # format
            fmted = [fmt.format(v, crit) for v, crit in sorted_pairs]
            # pad
            while len(fmted) < top_n:
                fmted.append("")
                vals.append(np.nan)
            df_fmt[f"model{i}"] = fmted
            df_num[f"model{i}"] = vals
        dfs_fmt[m.upper()] = df_fmt
        numeric_dfs[m.upper()] = df_num
    return dfs_fmt, numeric_dfs
