
from typing import Dict, List
import torch
from .utils import save_csv, has_matplotlib

@torch.no_grad()
def confusion_from_preds(y_true: torch.Tensor, y_pred: torch.Tensor, num_classes: int) -> torch.Tensor:
    k = num_classes
    idx = y_true * k + y_pred
    cm_flat = torch.bincount(idx, minlength=k*k)
    return cm_flat.view(k, k)

@torch.no_grad()
def per_class_metrics(cm: torch.Tensor, eps: float = 1e-12) -> Dict[str, torch.Tensor]:
    cm = cm.to(torch.float64)
    TP = torch.diag(cm); FP = cm.sum(0) - TP; FN = cm.sum(1) - TP; TN = cm.sum() - (TP + FP + FN)
    precision = TP / (TP + FP + eps)
    recall    = TP / (TP + FN + eps)
    f1        = 2 * precision * recall / (precision + recall + eps)
    npv       = TN / (TN + FN + eps)
    acc       = TP.sum() / (cm.sum() + eps)
    return {"precision": precision, "recall": recall, "f1": f1, "npv": npv, "acc": acc, "cm": cm.to(torch.long)}

@torch.no_grad()
def roc_curve_torch(y_true_bin: torch.Tensor, y_score: torch.Tensor):
    scores, order = torch.sort(y_score, descending=True)
    y = y_true_bin[order].to(torch.int64)
    P = y.sum().item(); N = y.numel() - P
    if P == 0 or N == 0:
        fpr = torch.tensor([0.0, 1.0], dtype=torch.float64)
        tpr = torch.tensor([0.0, 1.0], dtype=torch.float64)
        return fpr, tpr
    tp = torch.cumsum(y, dim=0).to(torch.float64)
    fp = torch.cumsum(1 - y, dim=0).to(torch.float64)
    tp = torch.cat([torch.tensor([0.0], dtype=torch.float64), tp])
    fp = torch.cat([torch.tensor([0.0], dtype=torch.float64), fp])
    tpr = tp / max(P, 1); fpr = fp / max(N, 1)
    return fpr, tpr

def auc_trapezoid(x: torch.Tensor, y: torch.Tensor) -> float:
    order = torch.argsort(x); x = x[order]; y = y[order]
    return float(torch.trapz(y, x).item())

def save_roc_curves(y_true: torch.Tensor, y_prob: torch.Tensor, class_names: List[str], out_dir: str):
    import os
    os.makedirs(out_dir, exist_ok=True)
    K = y_prob.shape[1]
    auc_rows = [["Class", "AUC"]]
    fprs_list, tprs_list, aucs = [], [], []
    for c in range(K):
        y_bin = (y_true == c).to(torch.int64)
        fpr, tpr = roc_curve_torch(y_bin, y_prob[:, c])
        auc_c = auc_trapezoid(fpr, tpr)
        auc_rows.append([class_names[c], f"{auc_c:.6f}"])
        aucs.append(auc_c)
        fprs_list.append(fpr); tprs_list.append(tpr)
        rows = [["FPR", "TPR"]]
        for i in range(len(fpr)):
            rows.append([float(fpr[i].item()), float(tpr[i].item())])
        save_csv(rows, os.path.join(out_dir, f"roc_class_{class_names[c]}.csv"))
    macro_auc = sum(aucs) / max(len(aucs), 1)
    y_true_onehot = torch.zeros((y_true.shape[0], K), dtype=torch.int64)
    y_true_onehot[torch.arange(y_true.shape[0]), y_true] = 1
    y_micro = y_true_onehot.reshape(-1).to(torch.int64)
    s_micro = y_prob.reshape(-1)
    fpr_micro, tpr_micro = roc_curve_torch(y_micro, s_micro)
    micro_auc = auc_trapezoid(fpr_micro, tpr_micro)
    auc_rows.append(["macro", f"{macro_auc:.6f}"])
    auc_rows.append(["micro", f"{micro_auc:.6f}"])
    save_csv(auc_rows, os.path.join(out_dir, "roc_auc.csv"))
    if has_matplotlib():
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 6))
        for c in range(K):
            fpr, tpr = fprs_list[c], tprs_list[c]
            plt.plot(fpr.tolist(), tpr.tolist(), label=f"{class_names[c]} (AUC={aucs[c]:.3f})")
        plt.plot(fpr_micro.tolist(), tpr_micro.tolist(), linestyle="--", label=f"micro (AUC={micro_auc:.3f})")
        plt.plot([0,1], [0,1], linestyle=":", label="chance")
        plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
        plt.title("ROC Curves"); plt.legend(loc="lower right", fontsize=8)
        plt.tight_layout(); plt.savefig(os.path.join(out_dir, "roc_curves.png"), dpi=150); plt.close()
