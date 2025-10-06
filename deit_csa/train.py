
import os, time
import torch
import torch.nn as nn
import torch.optim as optim

from .utils import set_seeds, count_trainable_params, save_csv
from .data import build_loaders
from .metrics import confusion_from_preds, per_class_metrics, save_roc_curves
from .models import make_deit_plus_multiscale_channel_spatial_boosted

# --- Config ---
DATASET_PATH = r'E:\Ashu2025\Mihic_Dataset\MIHIC_dataset\dataset'
IMG_SIZE_DEFAULT = 128
DEIT_NAME = 'deit_base_patch16_224'
EPOCHS = 15; LR = 1e-4; WEIGHT_DECAY = 1e-4
BATCH_TRAIN = 512; BATCH_EVAL = 256; NUM_WORKERS = 8
DROPOUT = 0.1; USE_AMP = torch.cuda.is_available(); LABEL_SMOOTH = 0.0; SEED = 42
BOOST_FULL_IMG_SIZE = 224

OUT_DIR = "results_ablation"; CKPT_DIR = os.path.join(OUT_DIR, "checkpoints")
os.makedirs(OUT_DIR, exist_ok=True); os.makedirs(CKPT_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
base_scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)

class ModelEMA:
    def __init__(self, model: nn.Module, decay=0.999):
        import copy
        self.ema = copy.deepcopy(model).eval()
        for p in self.ema.parameters(): p.requires_grad_(False)
        self.decay = decay
    @torch.no_grad()
    def update(self, model: nn.Module):
        for ema_p, p in zip(self.ema.parameters(), model.parameters()):
            ema_p.data.mul_(self.decay).add_(p.data, alpha=1.0 - self.decay)
    def to(self, device): self.ema.to(device); return self

def kd_loss(student_logits, teacher_logits, T=2.0):
    ps = torch.nn.functional.log_softmax(student_logits / T, dim=1)
    pt = torch.nn.functional.softmax(teacher_logits / T, dim=1)
    return torch.nn.functional.kl_div(ps, pt, reduction='batchmean') * (T * T)

def train_one_epoch_boosted(model, ema, loader, ce_loss, optimizer, device, clip=1.0, lambda_kd=0.4, T=2.0):
    model.train(); loss_sum=0.0; correct=0; total=0; t0=time.time()
    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)
    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True); labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=USE_AMP):
            logits_s = model(imgs); loss_ce  = ce_loss(logits_s, labels)
            with torch.no_grad(): logits_t = ema.ema(imgs)
            loss_kd = kd_loss(logits_s, logits_t, T=T); loss = (1.0 - lambda_kd) * loss_ce + lambda_kd * loss_kd
        scaler.scale(loss).backward(); scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip)
        scaler.step(optimizer); scaler.update(); ema.update(model)
        loss_sum += loss.item() * imgs.size(0); preds = logits_s.argmax(1); correct += (preds == labels).sum().item(); total += labels.size(0)
    return loss_sum/total, correct/total, time.time()-t0

@torch.no_grad()
def evaluate_tta(model, loader, ce_loss, device):
    model.eval(); loss_sum=0.0; correct=0; total=0
    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True); labels = labels.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=USE_AMP):
            logits1 = model(imgs); logits2 = model(torch.flip(imgs, dims=[3])); logits  = 0.5 * (logits1 + logits2); loss = ce_loss(logits, labels)
        loss_sum += loss.item() * imgs.size(0); preds = logits.argmax(1); correct += (preds == labels).sum().item(); total += labels.size(0)
    return loss_sum/total, correct/total

@torch.no_grad()
def gather_probs_labels(model, loader, device, tta=False):
    model.eval(); ys=[]; ps=[]; pr=[]
    for images, labels in loader:
        images = images.to(device, non_blocking=True); labels = labels.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=USE_AMP):
            logits = model(images)
            if tta: logits = 0.5 * (logits + model(torch.flip(images, dims=[3])))
            prob   = torch.softmax(logits, dim=1)
        preds  = prob.argmax(dim=1); ys.append(labels.cpu()); ps.append(preds.cpu()); pr.append(prob.cpu())
    return torch.cat(ys), torch.cat(ps), torch.cat(pr)

def run_all():
    set_seeds(SEED)
    # default loaders for 128, but boosted uses 224
    _train_loader, _val_loader, _test_loader, class_names = build_loaders(DATASET_PATH, IMG_SIZE_DEFAULT, BATCH_TRAIN, BATCH_EVAL, NUM_WORKERS)
    num_classes = len(class_names)
    print("Using device:", device); print("Classes:", class_names)

    img_size = BOOST_FULL_IMG_SIZE
    train_loader, val_loader, test_loader, class_names = build_loaders(DATASET_PATH, img_size, BATCH_TRAIN, BATCH_EVAL, NUM_WORKERS)
    model = make_deit_plus_multiscale_channel_spatial_boosted(num_classes=num_classes, deit_name=DEIT_NAME, img_size=img_size, drop=DROPOUT).to(device)
    param_count = count_trainable_params(model)
    print(f"[Boosted] Trainable parameters: {param_count:,} (~{param_count/1e6:.2f}M)")

    vit_params, local_head = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad: continue
        if n.startswith("local.") or n.startswith("sfuse.") or n.startswith("gate_ch") or n.startswith("classifier"):
            local_head.append(p)
        else:
            vit_params.append(p)
    optimizer = optim.AdamW([
        {"params": vit_params,   "lr": LR},
        {"params": local_head,   "lr": LR * 2.0},
    ], weight_decay=WEIGHT_DECAY)
    warmup_epochs = max(1, EPOCHS//5)
    warmup = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.2, total_iters=warmup_epochs)
    cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, EPOCHS - warmup_epochs))
    scheduler = optim.lr_scheduler.SequentialLR(optimizer, [warmup, cosine], milestones=[warmup_epochs])
    ce_loss = nn.CrossEntropyLoss(label_smoothing=max(LABEL_SMOOTH, 0.1))
    ema = ModelEMA(model, decay=0.999).to(device)

    best_val_acc = 0.0
    os.makedirs(CKPT_DIR, exist_ok=True)
    best_path = os.path.join(CKPT_DIR, "DeiT_plus_multiscale_channel_spatial.pth")

    for epoch in range(EPOCHS):
        tr_loss, tr_acc, tr_time = train_one_epoch_boosted(model, ema, train_loader, ce_loss, optimizer, device, clip=1.0, lambda_kd=0.4, T=2.0)
        va_loss, va_acc = evaluate_tta(ema.ema, val_loader, ce_loss, device)
        scheduler.step()
        print(f"[Boosted] Epoch {epoch+1:02d}/{EPOCHS} | Train: loss {tr_loss:.4f}, acc {tr_acc:.4f}, time {tr_time:.1f}s | Val: loss {va_loss:.4f}, acc {va_acc:.4f}")
        if va_acc > best_val_acc:
            best_val_acc = va_acc; torch.save({"model": ema.ema.state_dict(), "val_acc": best_val_acc}, best_path)

    if os.path.exists(best_path):
        state = torch.load(best_path, map_location=device)
        ema.ema.load_state_dict(state["model"]); used_model = ema.ema
        test_loss, test_acc = evaluate_tta(used_model, test_loader, ce_loss, device)
        print(f"[Boosted] Loaded best (val acc={state['val_acc']:.4f})")
        print(f"[Boosted] Test: loss {test_loss:.4f}, acc {test_acc:.4f}")

    y_true, y_pred, y_prob = gather_probs_labels(used_model, test_loader, device, tta=True)
    cm = confusion_from_preds(y_true, y_pred, len(class_names))
    metrics = per_class_metrics(cm)

    model_dir = os.path.join(OUT_DIR, "DeiT_plus_multiscale_channel_spatial"); os.makedirs(model_dir, exist_ok=True)
    rows = [["Class", "Precision", "Recall", "F1", "NPV"]]
    for i, cname in enumerate(class_names):
        rows.append([
            cname, float(metrics["precision"][i].item()),
            float(metrics["recall"][i].item()),
            float(metrics["f1"][i].item()),
            float(metrics["npv"][i].item()),
        ])
    rows.append(["ACC_overall", float(metrics["acc"].item()), "", "", ""])
    save_csv(rows, os.path.join(model_dir, "per_class_metrics.csv"))

    cm_rows = [[""] + class_names]
    for i, cname in enumerate(class_names):
        cm_rows.append([cname] + [int(cm[i, j].item()) for j in range(len(class_names))])
    save_csv(cm_rows, os.path.join(model_dir, "confusion_matrix.csv"))
    save_roc_curves(y_true, y_prob, class_names, os.path.join(model_dir, "roc"))
