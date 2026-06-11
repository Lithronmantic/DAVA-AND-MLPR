# scripts/strong_trainer.py
# -*- coding: utf-8 -*-
"""
StrongTrainer v3.8 (Graph Safe Edition)
鉁?淇 1: RuntimeError (Second Backward) - 澧炲姞 Meta Update 鐨勫紓甯告崟鑾蜂笌瀹夊叏璺宠繃
鉁?淇 2: SDPA Warning - 寮哄埗鍦?Meta Update 鏃剁鐢?Flash Attention
鉁?淇 3: NaN Guard - 淇濈暀鎵€鏈夋搴︾啍鏂笌鑷姩鎭㈠鏈哄埗
鉁?瀹屾暣鐗? 鏃犲垹鍑?
"""
import os, json, math, random, time
from pathlib import Path
from typing import Optional, Dict, Any, List
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, WeightedRandomSampler, Sampler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import yaml
import contextlib


def _sdp_math_only_ctx():
    """Return a context manager that forces the MATH SDP backend.

    Only the MATH backend supports create_graph=True (second-order gradients
    through attention).  Flash and efficient-attention backends do NOT implement
    the full backward-of-backward and raise NotImplementedError at runtime.

    Prefers the stable public API (torch.nn.attention.sdpa_kernel, PyTorch>=2.0)
    and falls back to the deprecated torch.backends.cuda.sdp_kernel.
    """
    try:
        from torch.nn.attention import sdpa_kernel, SDPBackend
        return sdpa_kernel([SDPBackend.MATH])
    except Exception:
        pass
    try:
        return torch.backends.cuda.sdp_kernel(
            enable_flash=False, enable_math=True, enable_mem_efficient=False
        )
    except Exception:
        return contextlib.nullcontext()


# -------------------- 鏍稿績缁勪欢瀵煎叆 --------------------
try:
    from cava_losses import CAVALoss
    from meta_reweighter import MetaReweighter, build_mlpr_features
    from meta_utils import meta_step_first_order_from_features
    from ssl_losses import ramp_up
    from ssl_strategy import build_ssl_strategy, OursMLPRStrategy
    from history_bank import HistoryBank
    from teacher_ema import EMATeacher
    from dataset import AVFromCSV, safe_collate_fn
    from enhanced_detector import EnhancedAVTopDetector
    from training_utils import compute_ema_decay
    from config_system import resolve_runtime_config, load_paper_exact_config, audit_against_paper_exact, save_audit_summary
except ImportError:
    import sys

    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from scripts.cava_losses import CAVALoss
    from scripts.meta_reweighter import MetaReweighter, build_mlpr_features
    from scripts.meta_utils import meta_step_first_order_from_features
    from scripts.ssl_losses import ramp_up
    from scripts.ssl_strategy import build_ssl_strategy, OursMLPRStrategy
    from scripts.history_bank import HistoryBank
    from scripts.teacher_ema import EMATeacher
    from scripts.dataset import AVFromCSV, safe_collate_fn
    from scripts.enhanced_detector import EnhancedAVTopDetector
    from scripts.training_utils import compute_ema_decay
    from scripts.config_system import resolve_runtime_config, load_paper_exact_config, audit_against_paper_exact, save_audit_summary

try:
    from dataset import safe_collate_fn_with_ids
except ImportError:
    try:
        from scripts.dataset import safe_collate_fn_with_ids
    except ImportError:
        def safe_collate_fn_with_ids(batch):
            return safe_collate_fn(batch)

# -------------------- AMP 娣峰悎绮惧害宸ュ叿 --------------------
try:
    from torch.amp import autocast as _autocast, GradScaler as _GradScaler

    AMP_DEVICE_ARG = True


    def amp_autocast(device_type, enabled=True, dtype=torch.float16):
        return _autocast(device_type, enabled=enabled, dtype=dtype)


    def AmpGradScaler(device_type, enabled=True):
        return _GradScaler(device_type, enabled=enabled)
except ImportError:
    from torch.cuda.amp import autocast as _autocast, GradScaler as _GradScaler

    AMP_DEVICE_ARG = False


    def amp_autocast(device_type, enabled=True, dtype=torch.float16):
        return _autocast(enabled=enabled)


    def AmpGradScaler(device_type, enabled=True):
        return _GradScaler(enabled=enabled)


# -------------------- 澧炲己鐗?Focal Loss (NaN 闃叉姢) --------------------
class FocalCrossEntropy(nn.Module):
    def __init__(self, gamma=2.0, label_smoothing=0.0, class_weights=None):
        super().__init__()
        self.gamma = float(gamma)
        self.label_smoothing = float(label_smoothing)
        self.register_buffer("class_weights", class_weights if class_weights is not None else None)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        if targets.ndim == 2: targets = targets.argmax(dim=1)

        with amp_autocast('cuda', enabled=False):
            logits_f32 = torch.clamp(logits.float(), min=-30, max=30)
            ce = F.cross_entropy(
                logits_f32, targets,
                weight=self.class_weights,
                label_smoothing=self.label_smoothing,
                reduction="none"
            )
            pt = torch.exp(-ce)
            focal_weight = (1 - pt) ** self.gamma
            loss = focal_weight * ce

            if torch.isnan(loss).any() or torch.isinf(loss).any():
                return None
            return loss.mean()


def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


class DistributedWeightedSampler(Sampler[int]):
    """Weighted sampler compatible with DDP.

    Samples `total_size` indices with replacement using the provided weights,
    then shards them across ranks so each process gets the same number of
    examples. This preserves the class-balancing intent of weighted sampling
    under DDP, which the previous code path silently disabled.
    """

    def __init__(
        self,
        weights: torch.Tensor,
        dataset_len: int,
        num_replicas: int,
        rank: int,
        replacement: bool = True,
        drop_last: bool = True,
        seed: int = 0,
    ) -> None:
        self.weights = weights.detach().cpu().double()
        self.dataset_len = int(dataset_len)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.replacement = bool(replacement)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self.epoch = 0

        if self.drop_last:
            self.num_samples = self.dataset_len // self.num_replicas
        else:
            self.num_samples = int(math.ceil(self.dataset_len / float(self.num_replicas)))
        self.num_samples = max(1, self.num_samples)
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        indices = torch.multinomial(
            self.weights,
            self.total_size,
            replacement=self.replacement,
            generator=g,
        ).tolist()
        indices = indices[self.rank:self.total_size:self.num_replicas]
        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)


# -------------------- 涓昏缁冨櫒绫?--------------------
class StrongTrainer:
    def __init__(
        self,
        cfg: Dict[str, Any],
        out_dir: str,
        resume_from: Optional[str] = None,
        local_rank: int = -1,          # ≥0 → DDP mode (set by train_ddp.py)
    ):
        self.cfg = resolve_runtime_config(cfg)
        cfg = self.cfg
        self.out_dir = Path(out_dir)

        # ── DDP / single-GPU device setup ──────────────────────────────────
        self.local_rank = local_rank
        self.ddp_mode = (local_rank >= 0) and dist.is_available() and dist.is_initialized()
        if self.ddp_mode:
            self.device = torch.device(f'cuda:{local_rank}')
            self.world_size = dist.get_world_size()
            self.global_rank = dist.get_rank()
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.world_size = 1
            self.global_rank = 0
        self.device_type = self.device.type
        # Only rank-0 writes logs / checkpoints / TensorBoard
        self.is_main = (self.global_rank == 0)

        if self.is_main:
            (self.out_dir / 'checkpoints').mkdir(parents=True, exist_ok=True)
            (self.out_dir / 'visualizations').mkdir(parents=True, exist_ok=True)
        if self.ddp_mode:
            dist.barrier()   # wait until rank-0 has created dirs

        self.resume_from = resume_from
        _set_seed(int(cfg.get("seed", 42)) + self.global_rank)

        # 鍒濆鍖?TensorBoard
        # TensorBoard -- only rank-0 writes logs
        if self.is_main:
            self.writer = SummaryWriter(log_dir=str(self.out_dir / 'runs'))
            print(f'[Rank0] TensorBoard logs: {self.out_dir / "runs"}')
        else:
            self.writer = None

        # 鎹熷け鍘嗗彶璁板綍鍣?
        self.loss_history = {
            'sup_loss': [], 'cava_loss': [], 'cava_align': [], 'cava_edge': [],
            'cava_prior': [], 'cava_gate_loss': [],
            'pseudo_loss': [], 'total_loss': [], 'ssl_mask_ratio': [],
            'gate_mean': [], 'gate_std': [], 'learning_rate': [], 'ema_decay': [],
            'val_acc_student': [], 'val_f1_student': [],
            'val_acc_teacher': [], 'val_f1_teacher': []
        }
        self.step_losses = {
            'sup_loss': [], 'cava_loss': [], 'pseudo_loss': [], 'total_loss': []
        }
        # Extended diagnostic logs written to extra_logs.json
        self._extra_logs = {
            'mlpr_inner': [],     # [step, loss] per bi-level update
            'mlpr_outer': [],     # [step, meta-val loss]
            'mlpr_w_mean': [],    # [step, mean_weight]
            'weight_samples': [], # [epoch, [sampled_w]] — for violin plots
            'weight_bins': [],    # [epoch, {bin: {correctness, count}}]
            'cava_delta': [],     # [epoch, delta stats]
        }
        self.epoch_records: List[Dict[str, Any]] = []
        # Per-epoch accumulators (reset at epoch start)
        self._epoch_wbin_accum = [{'sum_c': 0.0, 'n': 0} for _ in range(5)]
        self._epoch_delta_accum: list = []
        self._epoch_wsample: list = []

        # AMP
        self.amp_enabled = bool(cfg.get('training', {}).get('amp', True) and self.device.type == 'cuda')
        self.scaler = AmpGradScaler(self.device_type, enabled=self.amp_enabled)
        self.amp_disable_epoch = int(cfg.get("training", {}).get("amp_disable_epoch", 100))
        self.nan_count = 0
        self.consecutive_nan = 0
        self.total_steps = 0
        self.meta_fail_count = 0
        self.meta_update_count = 0   # cumulative executed outer steps
        self.meta_skip_count = 0     # cumulative skipped outer steps (all-masked batch)
        self._mlpr_weight_gen_fail_count = 0  # propagated from OursMLPRStrategy

        # 缁勪欢鍒濆鍖?
        self._setup_data(cfg)
        self._setup_model(cfg)
        self._setup_optimizer(cfg)
        self._setup_mlpr(cfg)
        self._setup_ssl(cfg)
        self._setup_ssl_strategy(cfg)   # unified strategy (after _setup_mlpr/_setup_ssl)

        # CAVA Setup
        self.cava_cfg = dict(cfg.get("cava", {}))
        self.cava_enabled = bool(self.cava_cfg.get("enabled", False))
        self.cava_loss_fn = CAVALoss(self.cava_cfg) if self.cava_enabled else None
        self._audit_config_against_paper_exact()

        # Checkpoint Loading
        self.start_epoch = 1
        self.best_f1 = -1.0
        self.no_improve = 0

        if self.resume_from is not None:
            self._load_checkpoint(self.resume_from)

    def _setup_data(self, cfg):
        data_cfg = cfg["data"]
        self.C = int(data_cfg["num_classes"])
        self.num_classes = self.C
        self.class_names = list(data_cfg["class_names"])
        root = data_cfg.get("data_root", "")

        l_csv = data_cfg["labeled_csv"]
        v_csv = data_cfg["val_csv"]
        u_csv = data_cfg.get("unlabeled_csv")

        self.ds_l = AVFromCSV(
            l_csv, root, self.C, self.class_names,
            video_cfg=cfg.get("video"), audio_cfg=cfg.get("audio"),
            is_unlabeled=False
        )
        self.ds_v = AVFromCSV(
            v_csv, root, self.C, self.class_names,
            video_cfg=cfg.get("video"), audio_cfg=cfg.get("audio"),
            is_unlabeled=False
        )
        self.ds_u = AVFromCSV(
            u_csv, root, self.C, self.class_names,
            video_cfg=cfg.get("video"), audio_cfg=cfg.get("audio"),
            is_unlabeled=True
        ) if (cfg.get("training", {}).get("use_ssl", False) and u_csv) else None

        self.stats = self._scan_stats(self.ds_l)
        if self.is_main:
            (self.out_dir / 'stats').mkdir(exist_ok=True, parents=True)
            with open(self.out_dir / 'stats' / 'class_stats.json', 'w') as f:
                json.dump(self.stats, f, ensure_ascii=False, indent=2)

        # ── Sampler: DDP uses DistributedSampler; single-GPU uses WeightedRandom ──
        tr = cfg.get("training", {})
        self.effective_batch_size = int(tr.get("batch_size", 16))
        runtime_bs_cfg = tr.get("runtime_batch_size", None)
        self.bs = self._resolve_runtime_batch_size(self.effective_batch_size, runtime_bs_cfg)

        if self.ddp_mode:
            # Per-GPU batch size: divide effective_batch by world_size
            per_gpu_bs = max(1, self.bs // self.world_size)
            self.grad_accum_steps = max(1, int(math.ceil(self.effective_batch_size / float(self.bs))))
            if data_cfg.get("sampler", "").lower() == "weighted":
                inv_freq = np.array(self.stats["inv_freq"], dtype=np.float32)
                weights = []
                for r in self.ds_l.rows:
                    idx = int(r.get("label_idx", 0))
                    weights.append(inv_freq[idx] if 0 <= idx < len(inv_freq) else 1.0)
                sampler_l = DistributedWeightedSampler(
                    weights=torch.tensor(weights, dtype=torch.double),
                    dataset_len=len(self.ds_l),
                    num_replicas=self.world_size,
                    rank=self.global_rank,
                    replacement=True,
                    drop_last=True,
                    seed=int(cfg.get("seed", 42)),
                )
            else:
                sampler_l = DistributedSampler(
                    self.ds_l, num_replicas=self.world_size, rank=self.global_rank, shuffle=True, drop_last=True
                )
            sampler_u = DistributedSampler(
                self.ds_u, num_replicas=self.world_size, rank=self.global_rank, shuffle=True, drop_last=True
            ) if self.ds_u is not None else None
        else:
            per_gpu_bs = self.bs
            self.grad_accum_steps = max(1, int(math.ceil(self.effective_batch_size / float(self.bs))))
            sampler_l = None
            if data_cfg.get("sampler", "").lower() == "weighted":
                inv_freq = np.array(self.stats["inv_freq"], dtype=np.float32)
                sampler_l = self._build_sampler(self.ds_l, inv_freq)
            sampler_u = None

        if self.grad_accum_steps > 1 and self.is_main:
            print(
                f"[RUNTIME_BATCH] effective_batch_size={self.effective_batch_size}, "
                f"per_gpu_batch={per_gpu_bs}, world_size={self.world_size}, "
                f"grad_accum_steps={self.grad_accum_steps}"
            )

        pin_mem = (self.device.type == 'cuda')
        self._sampler_l = sampler_l   # kept for epoch-level set_epoch call
        self._sampler_u = sampler_u   # kept for epoch-level set_epoch call (DDP shuffle)

        def _to(nw, default=60):
            return 0 if int(nw) == 0 else default

        self.loader_l = DataLoader(
            self.ds_l, batch_size=per_gpu_bs,
            sampler=sampler_l, shuffle=(sampler_l is None),
            num_workers=int(data_cfg.get("num_workers_train", 4)), pin_memory=pin_mem,
            drop_last=True, collate_fn=safe_collate_fn,
            timeout=_to(data_cfg.get("num_workers_train", 4)),
            persistent_workers=(int(data_cfg.get("num_workers_train", 4)) > 0)
        )
        self.loader_v = DataLoader(
            self.ds_v, batch_size=per_gpu_bs, shuffle=False,
            num_workers=int(data_cfg.get("num_workers_val", 2)), pin_memory=pin_mem,
            drop_last=False, collate_fn=safe_collate_fn,
            timeout=_to(data_cfg.get("num_workers_val", 2)),
            persistent_workers=(int(data_cfg.get("num_workers_val", 2)) > 0)
        )
        self.loader_u = None
        if self.ds_u is not None:
            self.loader_u = DataLoader(
                self.ds_u, batch_size=per_gpu_bs,
                sampler=sampler_u, shuffle=(sampler_u is None),
                num_workers=int(data_cfg.get("num_workers_unl", 4)), pin_memory=pin_mem,
                drop_last=True, collate_fn=safe_collate_fn_with_ids,
                timeout=_to(data_cfg.get("num_workers_unl", 4)),
                persistent_workers=(int(data_cfg.get("num_workers_unl", 4)) > 0)
            )

        # meta_val: dedicated clean set for MLPR outer-loop validation
        # Keeps meta-train (unlabeled pseudo-labels) and meta-val (clean labels) disjoint
        mv_csv = data_cfg.get("meta_val_csv")
        self.ds_mv = None
        self.loader_mv = None
        if mv_csv:
            try:
                self.ds_mv = AVFromCSV(
                    mv_csv, root, self.C, self.class_names,
                    video_cfg=cfg.get("video"), audio_cfg=cfg.get("audio"),
                    is_unlabeled=False
                )
                if self.ddp_mode:
                    sampler_mv = DistributedSampler(
                        self.ds_mv, num_replicas=self.world_size,
                        rank=self.global_rank, shuffle=True, drop_last=True
                    )
                    shuffle_mv = False
                else:
                    sampler_mv = None
                    shuffle_mv = True
                self._sampler_mv = sampler_mv  # saved for set_epoch() in train loop
                self.loader_mv = DataLoader(
                    self.ds_mv, batch_size=per_gpu_bs,
                    sampler=sampler_mv if self.ddp_mode else None,
                    shuffle=shuffle_mv,
                    num_workers=int(data_cfg.get("num_workers_val", 2)), pin_memory=pin_mem,
                    drop_last=True, collate_fn=safe_collate_fn,
                    timeout=_to(data_cfg.get("num_workers_val", 2)),
                    persistent_workers=(int(data_cfg.get("num_workers_val", 2)) > 0)
                )
                if self.is_main:
                    print(f"[DATA] meta_val_csv loaded: {len(self.ds_mv)} samples → loader_mv (dedicated MLPR outer-val)")
            except Exception as e:
                if self.is_main:
                    print(f"[WARN] meta_val_csv load failed ({e}), MLPR outer-val falls back to loader_v")
                self.ds_mv = None
                self.loader_mv = None

        self.labeled_steps_per_epoch = len(self.loader_l)
        if self.is_main:
            if self.ddp_mode:
                labeled_per_rank = len(self._sampler_l) if self._sampler_l is not None else len(self.ds_l)
                print(
                    f"[DATA] labeled_samples={len(self.ds_l)} total, "
                    f"per_rank={labeled_per_rank}, per_gpu_batch={per_gpu_bs}, "
                    f"labeled_steps_per_epoch={self.labeled_steps_per_epoch}"
                )
            else:
                print(
                    f"[DATA] labeled_samples={len(self.ds_l)}, batch={per_gpu_bs}, "
                    f"labeled_steps_per_epoch={self.labeled_steps_per_epoch}"
                )
            if self.labeled_steps_per_epoch <= 2:
                print(
                    "[WARN] Very few labeled optimizer batches per epoch. "
                    "This usually causes weak learning and makes early stopping unreliable. "
                    "Reduce training.batch_size or disable drop_last for low-label runs."
                )

    def _resolve_runtime_batch_size(self, target_bs: int, runtime_bs_cfg: Optional[int]) -> int:
        if runtime_bs_cfg is not None:
            return max(1, int(runtime_bs_cfg))
        if self.device.type != "cuda":
            return max(1, int(target_bs))
        try:
            mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        except Exception:
            return max(1, int(target_bs))
        bs = int(target_bs)
        if mem_gb <= 8.5:
            bs = min(bs, 16)
        elif mem_gb <= 12.5:
            bs = min(bs, 32)
        elif mem_gb <= 16.5:
            bs = min(bs, 64)
        return max(1, bs)

    def _setup_model(self, cfg):
        model_cfg = dict(cfg.get("model", {}))
        model_cfg["num_classes"] = self.C
        fusion_cfg = model_cfg.get("fusion", cfg.get("fusion", {}))
        base_model = EnhancedAVTopDetector({
            "model": model_cfg,
            "fusion": fusion_cfg,
            "cava": cfg.get("cava", {}),
            "video": cfg.get("video", {}),
            "audio": cfg.get("audio", {})
        }).to(self.device)

        tr_cfg = cfg.get("training", {})
        self._configure_aux_logits_runtime(
            base_model,
            emit_logits=bool(tr_cfg.get("emit_aux_logits", False)),
            model_name="student",
        )
        self.world_gpu_count = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0

        if self.ddp_mode:
            # DDP: each process owns exactly one GPU
            self.model = DDP(
                base_model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=bool(tr_cfg.get("ddp_find_unused", False)),
            )
            if self.is_main:
                print(f"[DDP] DistributedDataParallel enabled — world_size={self.world_size}")
        else:
            # Legacy single-process DataParallel (unchanged behaviour)
            self.multi_gpu = bool(tr_cfg.get("multi_gpu", False))
            if self.multi_gpu and self.device.type == "cuda" and self.world_gpu_count > 1:
                req = int(tr_cfg.get("num_gpus", self.world_gpu_count))
                dev_ids = list(range(min(self.world_gpu_count, max(1, req))))
                print(f"[MULTI_GPU] enabling DataParallel on GPUs: {dev_ids}")
                self.model = nn.DataParallel(base_model, device_ids=dev_ids).to(self.device)
            else:
                self.model = base_model

        if bool(cfg.get("model", {}).get("init_bias", False)):
            self._init_bias(self.model, self.stats["pi"])

    def _configure_aux_logits_runtime(self, model: nn.Module, *, emit_logits: bool, model_name: str) -> None:
        has_aux_heads = False
        for head_name in ("video_head", "audio_head"):
            head = getattr(model, head_name, None)
            if isinstance(head, nn.Module):
                has_aux_heads = True
                head.requires_grad_(False)

        if hasattr(model, "emit_aux_logits"):
            model.emit_aux_logits = bool(emit_logits and has_aux_heads)

        if has_aux_heads and self.is_main:
            emit_now = bool(getattr(model, "emit_aux_logits", False))
            print(
                f"[AUX] {model_name} auxiliary heads frozen for StrongTrainer "
                f"(no aux loss configured); emit_aux_logits={emit_now}"
            )

    def _student_model(self) -> nn.Module:
        # Both DDP and DataParallel wrap the real model under .module
        return self.model.module if isinstance(self.model, (nn.DataParallel, DDP)) else self.model

    def _snapshot_student_trainable_params(self) -> Dict[str, torch.Tensor]:
        """Clone live student params so meta-step cleanup can restore leaf tensors."""
        return {
            n: p.detach().clone()
            for n, p in self._student_model().named_parameters()
            if p.requires_grad
        }

    def _restore_student_trainable_params(
        self,
        snapshot: Optional[Dict[str, torch.Tensor]],
        *,
        context: str,
    ) -> int:
        """Repair any non-leaf live params and restore pre-meta values in-place."""
        if not snapshot:
            return 0

        repaired = 0
        with torch.no_grad():
            for name, p in self._student_model().named_parameters():
                snap = snapshot.get(name)
                if snap is None:
                    continue
                if (not p.is_leaf) or (p.grad_fn is not None):
                    try:
                        p.detach_()
                    except RuntimeError:
                        pass
                    p.requires_grad_(True)
                    repaired += 1
                if p.shape == snap.shape:
                    p.copy_(snap)

        if repaired and self.is_main:
            print(f"[MLPR] restored {repaired} live student params after {context}")
        return repaired

    def _state_dict_for_save(self) -> Dict[str, torch.Tensor]:
        return self._student_model().state_dict()

    @staticmethod
    def _strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {k[7:] if k.startswith("module.") else k: v for k, v in state_dict.items()}

    def _setup_optimizer(self, cfg):
        tr = cfg.get("training", {})
        loss_cfg = cfg.get("loss", {})
        model_cfg = cfg.get("model", {})

        self.loss_name = loss_cfg.get("name", "ce").lower()
        cw = loss_cfg.get("class_weights", None)
        class_weights = torch.tensor(cw, dtype=torch.float32, device=self.device) if cw is not None else None

        if self.loss_name == "focal_ce":
            self.criterion = FocalCrossEntropy(
                gamma=loss_cfg.get("gamma", 2.0),
                label_smoothing=loss_cfg.get("label_smoothing", 0.05),
                class_weights=class_weights
            ).to(self.device)
        else:
            self.criterion = nn.CrossEntropyLoss(
                weight=class_weights, label_smoothing=loss_cfg.get("label_smoothing", 0.05)
            ).to(self.device)

        self.epochs = int(tr.get("num_epochs", 30))
        base_lr = float(tr.get("learning_rate", 8e-5))
        bb_mult_req = float(tr.get("backbone_lr_mult", 0.1))
        self.wd = float(tr.get("weight_decay", 1e-3))
        self.grad_clip = float(tr.get("grad_clip_norm", 1.0))

        def _uses_pretrained(backbone_cfg, fallback_key: str) -> bool:
            if isinstance(backbone_cfg, dict):
                weights = str(backbone_cfg.get("weights", "")).strip().lower()
                if weights in {"", "none", "false", "random", "scratch"}:
                    return False
                return True
            return bool(model_cfg.get(fallback_key, False))

        video_pretrained = _uses_pretrained(model_cfg.get("video_backbone", {}), "pretrained")
        audio_pretrained = _uses_pretrained(model_cfg.get("audio_backbone", {}), "pretrained_audio")
        scratch_backbones = (not video_pretrained) and (not audio_pretrained)
        bb_mult = bb_mult_req
        if scratch_backbones and bb_mult < 1.0:
            bb_mult = 1.0
            if self.is_main:
                print(
                    f"[OPT] video/audio backbones are randomly initialized; "
                    f"overriding backbone_lr_mult {bb_mult_req:.4f} -> {bb_mult:.4f}"
                )

        head_params, bb_params = [], []
        for n, p in self.model.named_parameters():
            if not p.requires_grad: continue
            if "video_backbone" in n or "audio_backbone" in n:
                bb_params.append(p)
            else:
                head_params.append(p)

        self.opt = optim.AdamW(
            [{"params": head_params, "lr": base_lr}, {"params": bb_params, "lr": base_lr * bb_mult}],
            weight_decay=self.wd
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt, T_max=self.epochs, eta_min=1e-7)
        if self.is_main:
            head_count = sum(p.numel() for p in head_params)
            bb_count = sum(p.numel() for p in bb_params)
            print(
                f"[OPT] head_lr={base_lr:.6g} backbone_lr={base_lr * bb_mult:.6g} "
                f"(mult={bb_mult:.4f}) | head_params={head_count / 1e6:.2f}M "
                f"backbone_params={bb_count / 1e6:.2f}M"
            )

    def _setup_mlpr(self, cfg):
        self.mlpr_cfg = dict(cfg.get("mlpr", {}))
        self.mlpr_enabled = bool(self.mlpr_cfg.get("enabled", False))
        self._mlpr_feature_mode = str(self.mlpr_cfg.get("feature_mode", "legacy")).lower()

        use_hist = bool(self.mlpr_cfg.get("use_history_stats", True))
        use_cava = bool(self.mlpr_cfg.get("use_cava_signal", True))
        use_prob_vec = bool(self.mlpr_cfg.get("use_prob_vector", False))
        use_delay_prior = bool(self.mlpr_cfg.get("use_delay_prior_feature", True))

        if self._mlpr_feature_mode == "paper_7d":
            feat_dim = 7
            self._mlpr_feature_sources = [
                "max_prob",
                "entropy",
                "margin",
                "student_feat_norm",
                "history_mean",
                "history_std",
                "g_bar",
            ]
        elif self._mlpr_feature_mode in ("research_extended_7d", "extended_7d"):
            # build_mlpr_features always returns exactly 7 features in extended mode,
            # regardless of use_hist / use_cava flags.  Hard-code to 7 to avoid
            # MetaNet(in_dim=N≠7) → crash when flags differ from defaults.
            feat_dim = 7
            self._mlpr_feature_sources = [
                "max_prob",
                "entropy",
                "margin",
                "g_bar",
                "delta_prior_deviation",
                "loss_trend",
                "student_feat_norm",
            ]
            if not use_delay_prior:
                self._mlpr_feature_sources[4] = "delta_prior_deviation(disabled)"
        else:
            feat_dim = 3 + 1 + (2 if use_hist else 0) + (1 if use_cava else 0) + (self.C if use_prob_vec else 0)
            self._mlpr_feature_sources = [
                "max_prob", "entropy", "margin", "student_feat_norm",
                "history_mean", "history_std", "g_bar",
            ] + (["teacher_prob_vector"] if use_prob_vec else [])

        self.meta = MetaReweighter(
            cfg=self.cfg,
            feature_mode=self._mlpr_feature_mode,
            weight_clip=tuple(self.mlpr_cfg.get("weight_clip", [0.05, 0.95])),
            hidden_dim=int(self.mlpr_cfg.get("hidden_dim", 64)),
            num_hidden_layers=int(self.mlpr_cfg.get("num_hidden_layers", 1)),
            dropout=float(self.mlpr_cfg.get("dropout", 0.0)),
        ).to(self.device) if self.mlpr_enabled else None

        self.meta_opt = optim.Adam(self.meta.parameters(),
                                   lr=float(self.mlpr_cfg.get("meta_lr", 5e-5))) if self.mlpr_enabled else None
        self.hist_bank = HistoryBank(momentum=float(self.mlpr_cfg.get("history_momentum", 0.9))) if (
                self.mlpr_enabled and use_hist) else None

        self._mlpr_flags = {
            "use_hist": use_hist,
            "use_cava": use_cava,
            "use_prob_vec": use_prob_vec,
            "use_delay_prior": use_delay_prior,
        }
        self._mlpr_lambda_u = float(self.mlpr_cfg.get("lambda_u", 0.5))
        self._mlpr_meta_interval = int(self.mlpr_cfg.get("meta_interval", 50))
        self._mlpr_inner_lr = float(self.mlpr_cfg.get("inner_lr", 1e-4))
        if self.mlpr_enabled:
            print(f"[MLPR] feature_mode={self._mlpr_feature_mode}, feature_dim={feat_dim}")
            print(f"[MLPR] feature_sources={self._mlpr_feature_sources}")

    def _setup_ssl(self, cfg):
        tr_ssl = cfg.get("training", {})
        self.use_ssl = bool(tr_ssl.get("use_ssl", False) and self.ds_u is not None)
        ssl_cfg = cfg.get("training", {}).get("ssl", {})

        self.ema_decay_base = float(ssl_cfg.get("ema_decay_base", ssl_cfg.get("ema_decay", 0.999)))
        self.ema_decay_init = float(ssl_cfg.get("ema_decay_init", self.ema_decay_base))
        self.ssl_warmup_epochs = int(ssl_cfg.get("warmup_epochs", 3))
        self.ssl_final_thresh = float(ssl_cfg.get("final_thresh", 0.85))
        self.ssl_temp = float(ssl_cfg.get("consistency_temp", 1.0))
        self.ssl_alpha_schedule = str(ssl_cfg.get("alpha_schedule", "linear")).lower()
        self.ssl_alpha_ce_start = float(ssl_cfg.get("alpha_ce_start", 1.0))
        self.ssl_alpha_ce_end = float(ssl_cfg.get("alpha_ce_end", 0.5))
        self.ssl_alpha_kl_start = float(ssl_cfg.get("alpha_kl_start", 0.0))
        self.ssl_alpha_kl_end = float(ssl_cfg.get("alpha_kl_end", 0.5))
        self.ssl_alpha_ramp_epochs = max(
            1,
            int(ssl_cfg.get("alpha_ramp_epochs", max(1, self.epochs - self.ssl_warmup_epochs))),
        )

        # ------------------------------------------------------------------
        # Unlabeled-loss weight schedule: λ_u(t) = λ_u_max · r(t)
        #   - `lambda_u_max`        : peak value reached after ramp-up
        #   - `lambda_u_schedule`   : one of {"constant", "linear", "smooth",
        #                             "sigmoid"}.  Default keeps the legacy
        #                             constant behaviour when only the old
        #                             scalar `lambda_u` is provided.
        #   - `lambda_u_ramp_epochs`: epochs to go from r=0 → r=1, measured
        #                             from the end of `warmup_epochs`.
        # The legacy attribute `self.lambda_u` is preserved as an alias for
        # `self.lambda_u_max` so downstream tooling that still reads it (e.g.
        # logging, audit JSON) keeps working — it is *not* used for the
        # actual loss scaling, which is done via `_lambda_u_at(epoch)`.
        # ------------------------------------------------------------------
        legacy_lambda_u = float(ssl_cfg.get("lambda_u", 1.0))
        self.lambda_u_max = float(ssl_cfg.get("lambda_u_max", legacy_lambda_u))
        self.lambda_u = self.lambda_u_max  # legacy alias, see comment above
        self.lambda_u_schedule = str(ssl_cfg.get("lambda_u_schedule", "constant")).lower()
        self.lambda_u_ramp_epochs = max(
            1,
            int(ssl_cfg.get("lambda_u_ramp_epochs", self.ssl_alpha_ramp_epochs)),
        )

        self._use_dist_align = bool(ssl_cfg.get("use_dist_align", True))
        self._cls_thr = torch.full((self.C,), self.ssl_final_thresh, device=self.device)

        if self.use_ssl:
            teacher_model_cfg = dict(cfg.get("model", {}))
            teacher_model_cfg["num_classes"] = self.C
            teacher_fusion_cfg = teacher_model_cfg.get("fusion", cfg.get("fusion", {}))

            self.teacher = EnhancedAVTopDetector(
                {
                    "model": teacher_model_cfg,
                    "fusion": teacher_fusion_cfg,
                    "cava": cfg.get("cava", {}),
                    "video": cfg.get("video", {}),
                    "audio": cfg.get("audio", {})
                }
            ).to(self.device)
            self._configure_aux_logits_runtime(
                self.teacher,
                emit_logits=False,
                model_name="teacher",
            )

            self.teacher.load_state_dict(self._state_dict_for_save(), strict=False)
            for p in self.teacher.parameters(): p.requires_grad = False
            self.teacher.eval()
        else:
            self.teacher = None

        self._pi = torch.tensor(self.stats["pi"], dtype=torch.float32, device=self.device)

    def _ssl_loss_mix(self, epoch: int) -> tuple[float, float]:
        if self.ssl_alpha_schedule == "constant":
            return float(self.ssl_alpha_ce_end), float(self.ssl_alpha_kl_end)

        ramp_epoch = max(0, epoch - self.ssl_warmup_epochs)
        progress = min(1.0, ramp_epoch / max(1, self.ssl_alpha_ramp_epochs))
        alpha_ce = self.ssl_alpha_ce_start + progress * (self.ssl_alpha_ce_end - self.ssl_alpha_ce_start)
        alpha_kl = self.ssl_alpha_kl_start + progress * (self.ssl_alpha_kl_end - self.ssl_alpha_kl_start)
        return float(alpha_ce), float(alpha_kl)

    def _lambda_u_at(self, epoch: int) -> float:
        """Scheduled unlabeled-loss weight λ_u(t) = λ_u_max · r(t).

        Returns 0 during the SSL warm-up (when the unlabeled branch is gated
        off entirely by `ssl_active = epoch > warmup_epochs` upstream) and
        ramps up to `self.lambda_u_max` over `self.lambda_u_ramp_epochs`
        epochs once warm-up finishes.

        Supported schedules:
          - "constant" : λ_u(t) ≡ λ_u_max (legacy behaviour)
          - "linear"   : r(t) = t/N
          - "smooth"   : r(t) = 3t² − 2t³   (Mean-Teacher style cubic ramp)
          - "sigmoid"  : r(t) = exp(-5·(1−t)²)  (Laine & Aila 2017)
        with t ∈ [0, 1] computed as min(1, (epoch − warmup_epochs)/ramp_N).
        """
        # Hard gate: SSL branch is inactive during warm-up; λ_u is undefined
        # but we return 0 so any accidental multiplication stays a no-op.
        if epoch <= self.ssl_warmup_epochs:
            return 0.0
        if self.lambda_u_schedule == "constant":
            return float(self.lambda_u_max)

        ramp_epoch = max(0, epoch - self.ssl_warmup_epochs)
        N = max(1, int(self.lambda_u_ramp_epochs))
        t = max(0.0, min(1.0, ramp_epoch / float(N)))

        if self.lambda_u_schedule == "linear":
            r = t
        elif self.lambda_u_schedule in ("smooth", "polynomial", "cubic"):
            r = ramp_up(ramp_epoch, N)        # 3t² − 2t³
        elif self.lambda_u_schedule == "sigmoid":
            r = math.exp(-5.0 * (1.0 - t) ** 2)
        else:
            # Unknown schedule label → fall back to constant for safety
            r = 1.0
        return float(self.lambda_u_max * r)

    def _setup_ssl_strategy(self, cfg):
        """Instantiate the pluggable SSL pseudo-label strategy.

        The strategy controls ONLY the unlabeled branch (pseudo-label targets,
        sample weights, unsupervised loss).  Everything else (backbone, CAVA,
        MIL, EMA, optimizer) remains identical across all methods.
        """
        method = cfg.get("training", {}).get("ssl_method", "ours_mlpr").lower()
        self.ssl_strategy = build_ssl_strategy(cfg, trainer_ref=self)

        # Disable MLPR bookkeeping for non-MLPR strategies to avoid dead code paths
        if not isinstance(self.ssl_strategy, OursMLPRStrategy):
            self.mlpr_enabled = False

        if self.is_main:
            print(f"[SSL] Strategy: {method}  |  dist_align={self.ssl_strategy.use_dist_align}")

    def _audit_config_against_paper_exact(self):
        try:
            repo_root = Path(__file__).resolve().parents[1]
            paper_cfg = load_paper_exact_config(repo_root)
            summary = audit_against_paper_exact(self.cfg, paper_cfg)
            self.config_audit = summary

            print("[CONFIG_AUDIT] key settings:")
            for k, v in summary["current"].items():
                print(f"  - {k}: {v}")
            if summary["is_paper_exact"]:
                print("[CONFIG_AUDIT] profile matches paper_exact.")
            else:
                print(f"[CONFIG_AUDIT] differs from paper_exact: {summary['num_diffs']} item(s)")
                for row in summary["diffs"][:20]:
                    print(f"    * {row['key']}: current={row['current']} | paper_exact={row['paper_exact']}")

            save_audit_summary(self.out_dir / "stats" / "config_audit.json", summary)
            self.loss_history["config_audit_diffs"] = [float(summary["num_diffs"])]
            if self.writer is not None:
                self.writer.add_text("ConfigAudit/is_paper_exact", str(summary["is_paper_exact"]))
                self.writer.add_text("ConfigAudit/num_diffs", str(summary["num_diffs"]))
        except Exception as exc:
            self.config_audit = {"error": str(exc)}
            print(f"[CONFIG_AUDIT] skipped: {exc}")

    def _load_checkpoint(self, checkpoint_path: str):
        ckpt_path = Path(checkpoint_path)
        if not ckpt_path.exists():
            print(f"鈿狅笍 Checkpoint not found: {checkpoint_path}")
            return
        print(f"馃搨 Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        try:
            sd = checkpoint['state_dict']
            self.model.load_state_dict(sd, strict=False)
        except RuntimeError as e:
            try:
                sd2 = self._strip_module_prefix(checkpoint['state_dict'])
                self._student_model().load_state_dict(sd2, strict=False)
            except RuntimeError:
                print(f"鈿狅笍 Warning during model loading: {e}")

        if self.teacher is not None:
            try:
                self.teacher.load_state_dict(self._state_dict_for_save(), strict=False)
            except RuntimeError as e:
                print(f"鈿狅笍 Warning during teacher loading: {e}")

        if 'epoch' in checkpoint:
            self.start_epoch = checkpoint['epoch'] + 1
        if 'best_f1' in checkpoint:
            self.best_f1 = checkpoint['best_f1']
        print(f"馃幆 Checkpoint loaded! Resuming from epoch {self.start_epoch} (Best F1: {self.best_f1:.4f})")

    def _scan_stats(self, ds_l) -> Dict[str, Any]:
        C = self.C
        counts = np.zeros(C, dtype=np.int64)
        n = len(ds_l)
        for i in range(n):
            try:
                item = ds_l.rows[i]
                idx = item.get("label_idx")
                if idx is not None and 0 <= idx < C:
                    counts[idx] += 1
            except Exception:
                continue
        total = counts.sum()
        pi = (counts / total) if total > 0 else np.ones(C, dtype=np.float32) / C
        inv = 1.0 / np.clip(counts.astype(np.float32), 1.0, None)
        inv = inv / inv.mean()
        return {
            "counts": counts.tolist(),
            "pi": pi.astype(np.float32).tolist(),
            "inv_freq": inv.astype(np.float32).tolist(),
            "total": int(total),
        }

    def _build_sampler(self, ds_l, inv_freq):
        weights = []
        for r in ds_l.rows:
            idx = int(r.get("label_idx", 0))
            if 0 <= idx < len(inv_freq):
                weights.append(inv_freq[idx])
            else:
                weights.append(1.0)
        return WeightedRandomSampler(torch.tensor(weights, dtype=torch.double), len(weights))

    def _init_bias(self, model, pi):
        pi_tensor = torch.tensor(pi, dtype=torch.float32, device=self.device)

        def _try_set_bias(linear: nn.Linear):
            if isinstance(linear, nn.Linear) and linear.bias is not None:
                with torch.no_grad():
                    log_pi = torch.log(torch.clamp(pi_tensor, min=1e-8)).to(linear.bias.device)
                    linear.bias.copy_(log_pi)
                return True
            return False

        if hasattr(model, 'mil_head') and hasattr(model.mil_head, 'frame_classifier'):
            for m in reversed(list(model.mil_head.frame_classifier)):
                if _try_set_bias(m): break
        elif hasattr(model, 'classifier') and isinstance(model.classifier, nn.Linear):
            _try_set_bias(model.classifier)

    # v3.4 鏂板锛氭鏌ユā鍨嬫槸鍚﹀凡缁忔崯鍧?(鍏ㄦ槸 NaN)
    def _check_model_health(self):
        for name, param in self.model.named_parameters():
            if not torch.isfinite(param).all():
                print(f"馃拃 Model corrupted at layer: {name}")
                return False
        return True

    # v3.4 鏂板锛氳嚜鍔ㄥ洖婊氫笌闄嶇骇
    def _perform_auto_recovery(self):
        print("\n馃殤 [Auto-Recovery] Model poisoning detected!")
        print("馃攧 Rolling back to best_f1.pth...")

        ckpt_path = self.out_dir / 'checkpoints' / 'best_f1.pth'
        if not ckpt_path.exists():
            ckpt_path = self.out_dir / 'checkpoints' / 'latest.pth'

        if not ckpt_path.exists():
            # 训练初期（epoch 1）尚无 checkpoint，不应 abort。
            # 直接应用降级措施（FP32 + LR 减半）继续训练。
            print("⚠️  No checkpoint yet (early training). Skipping rollback.")
            print("📉 Degrading training mode for stability:")
            print("   1. Disabling AMP (FP16 -> FP32)")
            self.amp_enabled = False
            self.scaler = AmpGradScaler(self.device_type, enabled=False)
            print("   2. Reducing Learning Rate by 50%")
            for param_group in self.opt.param_groups:
                param_group['lr'] *= 0.5
            self.consecutive_nan = 0
            self.nan_count = 0
            print("✅  Stabilization applied. Resuming in FP32 mode.\n")
            return

        self._load_checkpoint(str(ckpt_path))

        print("馃搲 Degrading training mode for stability:")
        print("   1. Disabling AMP (FP16 -> FP32)")
        self.amp_enabled = False
        self.scaler = AmpGradScaler(self.device_type, enabled=False)

        print("   2. Reducing Learning Rate by 50%")
        for param_group in self.opt.param_groups:
            param_group['lr'] *= 0.5

        self.consecutive_nan = 0
        self.nan_count = 0
        print("鉁?Recovery complete. Resuming training in Safe Mode.\n")

    def _reset_scaler_if_needed(self):
        if self.scaler.is_enabled():
            self.scaler = AmpGradScaler(self.device_type, enabled=True)
        self.opt.zero_grad(set_to_none=True)
        self.consecutive_nan += 1

        # v3.4: 杩炵画瑙﹀彂鐔旀柇瓒呰繃 5 娆★紝鎴栬€呮ā鍨嬪凡缁?NaN锛岃Е鍙戣嚜鍔ㄥ洖婊?
        if self.consecutive_nan >= 5 or not self._check_model_health():
            self._perform_auto_recovery()

    def _ema_update(self, epoch: int):
        if self.teacher is None: return
        stu = self._student_model()
        ema_now = compute_ema_decay(
            epoch=epoch,
            ema_decay_base=self.ema_decay_base,
            ema_decay_init=self.ema_decay_init,
            warmup_epochs=self.ssl_warmup_epochs
        )
        with torch.no_grad():
            for t_p, s_p in zip(self.teacher.parameters(), stu.parameters()):
                # v3.6 Fixed s_param -> s_p
                t_p.data.mul_(ema_now).add_(s_p.data, alpha=1.0 - ema_now)
            for t_b, s_b in zip(self.teacher.buffers(), stu.buffers()):
                t_b.copy_(s_b)
        self._last_ema_decay = ema_now

    # ============================================================================
    # 淇敼1锛氬湪 strong_trainer.py 涓壘鍒?_meta_update_step 鏂规硶锛堝ぇ绾︾590琛岋級
    # 瀹屾暣鏇挎崲杩欎釜鏂规硶
    # ============================================================================

    def _meta_update_step(self, step_count: int):
        """v3.9: 瀹屽叏闅旂鐨勫厓瀛︿範鏇存柊"""
        if not self.mlpr_enabled or self.meta is None or self.meta_opt is None:
            return

        student_snapshot = None
        try:
            # 馃敶 鍏抽敭淇1锛氫繚瀛樺苟鍒囨崲妯″瀷鐘舵€?
            was_training = self.model.training
            self.model.eval()  # 鍒囨崲鍒拌瘎浼版ā寮忥紝闃叉BatchNorm鍒涘缓璁＄畻鍥?

            # 馃敶 鍏抽敭淇2锛氭竻绌烘墍鏈夋搴?
            self.model.zero_grad(set_to_none=True)
            self.opt.zero_grad(set_to_none=True)
            self.meta_opt.zero_grad(set_to_none=True)
            student_snapshot = self._snapshot_student_trainable_params()

            # 1. 鍑嗗楠岃瘉闆嗘暟鎹?
            with torch.no_grad():
                try:
                    # Prefer dedicated meta_val loader (clean, disjoint from train)
                    _meta_loader = getattr(self, 'loader_mv', None) or self.loader_v
                    val_iter = getattr(self, '_val_iter_for_meta', None)
                    if val_iter is None:
                        val_iter = iter(_meta_loader)
                        self._val_iter_for_meta = val_iter
                    val_batch = next(val_iter)
                except StopIteration:
                    _meta_loader = getattr(self, 'loader_mv', None) or self.loader_v
                    self._val_iter_for_meta = iter(_meta_loader)
                    val_batch = next(self._val_iter_for_meta)

                if len(val_batch) == 4:
                    v_val, a_val, y_val, _ = val_batch
                else:
                    v_val, a_val, y_val = val_batch

                v_val = v_val.to(self.device).float()
                a_val = a_val.to(self.device).float()
                y_val = y_val.argmax(dim=1).to(self.device) if y_val.ndim == 2 else y_val.to(self.device)

                # Use explicit None checks rather than hasattr: the failure path in
                # OursMLPRStrategy.compute_sample_weights() sets these to None on error,
                # so hasattr alone would still allow a stale-cache meta-update.
                _last_labeled = getattr(self, '_last_labeled_batch', None)
                _last_unlabeled = getattr(self, '_last_unlabeled_batch', None)
                _last_feats = getattr(self, '_last_w_features', None)
                _last_teacher_prob = getattr(self, '_last_teacher_prob', None)
                _last_ssl_cfg = getattr(self, '_last_ssl_loss_cfg', None)
                if (
                    _last_labeled is None
                    or _last_unlabeled is None
                    or _last_feats is None
                    or _last_teacher_prob is None
                    or _last_ssl_cfg is None
                ):
                    if was_training:
                        self.model.train()
                    return

                v_l, a_l, y_l = _last_labeled
                v_tr, a_tr, y_tr = _last_unlabeled
                w_features = _last_feats

            # 2. 鎵ц鍏冨涔犳洿鏂帮紙闅旂鐨勭幆澧冿級
            try:
                # create_graph=True (in sgd_fast_weights) requires second-order
                # gradients through attention.  Only the MATH SDP backend supports
                # backward-of-backward; flash and efficient-attention do NOT and
                # raise "derivative for _scaled_dot_product_efficient_attention
                # _backward is not implemented".  Disable both non-math backends.
                _sdp_ctx = _sdp_math_only_ctx()
                with _sdp_ctx:
                    w_mask = getattr(self, '_last_w_mask', None)
                    meta_logs = self._simple_meta_step(
                        v_l=v_l, a_l=a_l, y_l=y_l,
                        v_tr=v_tr, a_tr=a_tr, yhat_tr=y_tr,
                        teacher_prob_tr=_last_teacher_prob,
                        w_features=w_features,
                        v_val=v_val, a_val=a_val, y_val=y_val, mask_tr=w_mask,
                        alpha_ce=float(_last_ssl_cfg.get("alpha_ce", 1.0)),
                        alpha_kl=float(_last_ssl_cfg.get("alpha_kl", 0.0)),
                        temperature=float(_last_ssl_cfg.get("temperature", 1.0)),
                        # `_last_ssl_loss_cfg["lambda_u"]` is cached by the
                        # SSL strategy from the most recent training step and
                        # already reflects λ_u(t).  Fallback to λ_u_max is only
                        # hit when the meta-step fires before any SSL step.
                        lambda_u=float(_last_ssl_cfg.get("lambda_u", self.lambda_u_max)),
                    )

                # Save inner/outer losses to extra_logs for visualization
                _skipped = meta_logs.get("skipped", False)
                if _skipped:
                    self.meta_skip_count += 1
                    self._epoch_meta_skip += 1
                else:
                    self.meta_update_count += 1
                    self._epoch_meta_exec += 1
                if self.is_main:
                    if _skipped:
                        print(
                            f"[MLPR] step={step_count} SKIPPED (all pseudo masked; "
                            f"ep_skip={self._epoch_meta_skip}, ep_exec={self._epoch_meta_exec})"
                        )
                    else:
                        self._extra_logs['mlpr_inner'].append([int(step_count), float(meta_logs['meta_train_loss'])])
                        self._extra_logs['mlpr_outer'].append([int(step_count), float(meta_logs['meta_val_loss'])])
                        self._extra_logs['mlpr_w_mean'].append([int(step_count), float(meta_logs['w_mean'])])
                        print(
                            f"[MLPR] outer-step #{self.meta_update_count} at step={step_count} "
                            f"(train={meta_logs['meta_train_loss']:.4f}, val={meta_logs['meta_val_loss']:.4f}, "
                            f"w_mean={meta_logs['w_mean']:.4f}, ep_exec={self._epoch_meta_exec}, ep_skip={self._epoch_meta_skip})"
                        )
                if not _skipped and step_count % 100 == 0:
                    if self.writer is not None:
                        self.writer.add_scalar('Meta/Val_Loss_Approx', meta_logs["meta_val_loss"], step_count)
                    if self.writer is not None:
                        self.writer.add_scalar('Meta/Train_Loss_Approx', meta_logs["meta_train_loss"], step_count)
                    if self.writer is not None:
                        self.writer.add_scalar('Meta/W_Mean', meta_logs["w_mean"], step_count)
                    if self.writer is not None:
                        self.writer.add_scalar('Meta/W_Std', meta_logs["w_std"], step_count)
                    if self.writer is not None:
                        self.writer.add_scalar('Meta/W_Min', meta_logs["w_min"], step_count)
                    if self.writer is not None:
                        self.writer.add_scalar('Meta/W_Max', meta_logs["w_max"], step_count)

            finally:
                # 馃敶 鍏抽敭淇3锛氬畬鍏ㄦ竻鐞?
                self._restore_student_trainable_params(
                    student_snapshot,
                    context=f"meta_step(step={step_count})",
                )
                self.model.zero_grad(set_to_none=True)
                self.opt.zero_grad(set_to_none=True)
                self.meta_opt.zero_grad(set_to_none=True)

                # 娓呯悊CUDA缂撳瓨
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # 馃敶 鍏抽敭淇4锛氭仮澶嶈缁冩ā寮?
                if was_training:
                    self.model.train()

        except Exception as e:
            self.meta_fail_count += 1
            if self.meta_fail_count < 10:
                import traceback as _tb
                print(f"[MLPR] meta update FAILED (#{self.meta_fail_count}): {e}")
                _tb.print_exc()
            elif self.meta_fail_count % 50 == 0:
                print(f"[MLPR] meta update still failing (#{self.meta_fail_count}): {e}")

            # Emergency cleanup
            self._restore_student_trainable_params(
                student_snapshot,
                context=f"meta_step_exception(step={step_count})",
            )
            self.model.zero_grad(set_to_none=True)
            self.opt.zero_grad(set_to_none=True)
            if self.meta_opt is not None:
                self.meta_opt.zero_grad(set_to_none=True)

            # Restore training mode -- was_training is always set as the very first
            # line of the outer try, before anything that could raise.  This path is
            # hit when an exception fires BEFORE the inner try/finally (e.g. in the
            # with torch.no_grad() block), so the inner finally that normally calls
            # self.model.train() never ran.
            try:
                if was_training:
                    self.model.train()
            except NameError:
                pass  # impossible in practice, but defensive

    def _simple_meta_step(
            self,
            v_l: torch.Tensor,
            a_l: torch.Tensor,
            y_l: torch.Tensor,
            v_tr: torch.Tensor,
            a_tr: torch.Tensor,
            yhat_tr: torch.Tensor,
            teacher_prob_tr: torch.Tensor,
            w_features: torch.Tensor,
            v_val: torch.Tensor,
            a_val: torch.Tensor,
            y_val: torch.Tensor,
            mask_tr=None,
            alpha_ce: float = 1.0,
            alpha_kl: float = 0.0,
            temperature: float = 1.0,
            lambda_u: float = 1.0) -> Dict[str, float]:
        """First-order bi-level meta update using an isolated shadow copy of the student.

        Root cause of the 'backward through graph a second time' crash:
        functional_call/_reparametrize_module temporarily replaces live
        student_model._parameters[n] with fast_state tensors (grad_fn=SubBackward0).
        On certain PyTorch builds the restoration in the finally block fails to
        reinstate the original nn.Parameter objects, leaving the live model with
        non-leaf parameters.  The next supervised backward then tries to use a
        computation graph whose saved tensors were already freed by L_val.backward().

        Fix: deep-copy the student into a throw-away shadow before passing it to
        meta_step_first_order_from_features.  functional_call can corrupt meta_shadow
        all it wants; the live model is never touched.  meta_net still receives the
        correct outer-loop gradient through the chain:
          L_val → fast_state → grad(L_tr, det_params) → L_tr → w_tr → meta_net
        because meta_net and w_features are NOT part of the shadow copy.

        mask_tr: SSL pseudo-label mask [B] -- masked samples get zero weight.
        """
        import copy
        meta_shadow = copy.deepcopy(self._student_model())
        try:
            result = meta_step_first_order_from_features(
                student_model=meta_shadow,
                meta_net=self.meta,
                meta_opt=self.meta_opt,
                w_features=w_features,
                v_l=v_l,
                a_l=a_l,
                y_l=y_l,
                v_tr=v_tr,
                a_tr=a_tr,
                yhat_tr=yhat_tr,
                teacher_prob_tr=teacher_prob_tr,
                v_val=v_val,
                a_val=a_val,
                y_val=y_val,
                lr_inner=self._mlpr_inner_lr,
                mask_tr=mask_tr,
                alpha_ce=alpha_ce,
                alpha_kl=alpha_kl,
                temperature=temperature,
                lambda_u=lambda_u,
            )
        finally:
            del meta_shadow  # release immediately; do not keep the dirty copy alive

        # Post-call sanity: verify the live student parameters are still leaf tensors.
        # If this fires, something else is touching live params during the meta path.
        if __debug__:
            _live = self._student_model()
            _bad = [(n, p.grad_fn) for n, p in _live.named_parameters() if p.grad_fn is not None]
            if _bad:
                _names = ', '.join(n for n, _ in _bad[:5])
                raise RuntimeError(
                    f"[MLPR] live student parameters contaminated after meta step "
                    f"({len(_bad)} non-leaf params, e.g. {_names}). "
                    f"First grad_fn: {_bad[0][1]}"
                )

        return result

    def _save_loss_history(self):
        with open(self.out_dir / 'loss_history.json', 'w') as f:
            json.dump(self.loss_history, f, indent=2)

    def _save_epoch_metrics_csv(self):
        if not self.epoch_records:
            return

        import csv

        out_path = self.out_dir / 'epoch_metrics.csv'
        fieldnames = [
            'epoch',
            'seed',
            'train_total_loss',
            'train_sup_loss',
            'train_cava_loss',
            'train_cava_align',
            'train_cava_edge',
            'train_cava_prior',
            'train_cava_gate_loss',
            'train_pseudo_loss',
            'train_ssl_mask_ratio',
            'train_gate_mean',
            'train_gate_std',
            'train_r_align_raw',
            'train_delta_mean',
            'train_accepted_pseudo_count',
            'train_meta_exec_ep',
            'train_meta_skip_ep',
            'train_meta_update_count',
            'val_acc_student',
            'val_f1_student',
            'val_pred_majority_class_student',
            'val_pred_majority_ratio_student',
            'val_acc_teacher',
            'val_f1_teacher',
            'val_pred_majority_class_teacher',
            'val_pred_majority_ratio_teacher',
            'learning_rate',
            'ema_decay',
            'best_f1_so_far',
            'is_best_epoch',
            'no_improve',
            'epoch_time_s',
        ]
        with open(out_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.epoch_records)

    def _record_epoch_metrics(
        self,
        epoch: int,
        tr_metrics: Dict[str, float],
        val_res: Dict[str, Dict[str, float]],
        current_lr: float,
        ema_epoch: Optional[float],
        is_best_epoch: bool,
        epoch_time_s: float = 0.0,
    ) -> None:
        row = {
            'epoch': int(epoch),
            'seed': int(self.cfg.get("seed", 42)),
            'train_total_loss': float(tr_metrics.get('total', 0.0)),
            'train_sup_loss': float(tr_metrics.get('sup', 0.0)),
            'train_cava_loss': float(tr_metrics.get('cava_loss', 0.0)),
            'train_cava_align': float(tr_metrics.get('cava_align', 0.0)),
            'train_cava_edge': float(tr_metrics.get('cava_edge', 0.0)),
            'train_cava_prior': float(tr_metrics.get('cava_prior', 0.0)),
            'train_cava_gate_loss': float(tr_metrics.get('cava_gate_loss', 0.0)),
            'train_pseudo_loss': float(tr_metrics.get('pseudo_loss', 0.0)),
            'train_ssl_mask_ratio': float(tr_metrics.get('ssl_mask_ratio', 0.0)),
            'train_gate_mean': float(tr_metrics.get('gate_mean', 0.0)),
            'train_gate_std': float(tr_metrics.get('gate_std', 0.0)),
            'train_r_align_raw': float(tr_metrics.get('r_align_raw', 0.0)),
            'train_delta_mean': float(tr_metrics.get('delta_mean', 0.0)),
            'train_accepted_pseudo_count': float(tr_metrics.get('accepted_pseudo_count', 0.0)),
            'train_meta_exec_ep': int(tr_metrics.get('meta_exec_ep', 0)),
            'train_meta_skip_ep': int(tr_metrics.get('meta_skip_ep', 0)),
            'train_meta_update_count': int(tr_metrics.get('meta_update_count', 0)),
            'val_acc_student': float(val_res['student'].get('acc', 0.0)),
            'val_f1_student': float(val_res['student'].get('f1_macro', 0.0)),
            'val_pred_majority_class_student': int(val_res['student'].get('pred_majority_class', -1)),
            'val_pred_majority_ratio_student': float(val_res['student'].get('pred_majority_ratio', 0.0)),
            'val_acc_teacher': float(val_res['teacher'].get('acc', 0.0)),
            'val_f1_teacher': float(val_res['teacher'].get('f1_macro', 0.0)),
            'val_pred_majority_class_teacher': int(val_res['teacher'].get('pred_majority_class', -1)),
            'val_pred_majority_ratio_teacher': float(val_res['teacher'].get('pred_majority_ratio', 0.0)),
            'learning_rate': float(current_lr),
            'ema_decay': "" if ema_epoch is None else float(ema_epoch),
            'best_f1_so_far': float(self.best_f1),
            'is_best_epoch': int(bool(is_best_epoch)),
            'no_improve': int(self.no_improve),
            'epoch_time_s': float(epoch_time_s),
        }
        self.epoch_records.append(row)
        self._save_epoch_metrics_csv()

    def _save_extra_logs(self):
        """Save extended diagnostic data for paper figure generation."""
        try:
            out_path = self.out_dir / 'extra_logs.json'
            with open(out_path, 'w') as f:
                json.dump(self._extra_logs, f, indent=2)
        except Exception as exc:
            print(f'[WARN] Could not save extra_logs: {exc}')

    def _plot_all_visualizations(self):
        """鎭㈠鎵€鏈夊彲瑙嗗寲鍔熻兘"""
        viz_dir = self.out_dir / 'visualizations'
        self._plot_main_losses(viz_dir / 'main_losses.png')
        self._plot_cava_details(viz_dir / 'cava_details.png')
        self._plot_validation_metrics(viz_dir / 'validation_metrics.png')
        self._plot_training_dynamics(viz_dir / 'training_dynamics.png')
        if len(self.step_losses['total_loss']) > 0:
            self._plot_smooth_step_curves(viz_dir / 'smooth_step_losses.png')

    def _plot_main_losses(self, save_path):
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('Training Loss Curves Overview', fontsize=18, fontweight='bold', y=0.995)
        epochs = range(1, len(self.loss_history['total_loss']) + 1)

        axes[0, 0].plot(epochs, self.loss_history['total_loss'], 'b-', label='Total Loss')
        axes[0, 0].set_title('Total Loss');
        axes[0, 0].legend()

        axes[0, 1].plot(epochs, self.loss_history['sup_loss'], 'g-', label='Supervised Loss')
        axes[0, 1].set_title('Supervised Loss');
        axes[0, 1].legend()

        axes[0, 2].plot(epochs, self.loss_history['cava_loss'], 'r-', label='CAVA Loss')
        axes[0, 2].set_title('CAVA Loss');
        axes[0, 2].legend()

        axes[1, 0].plot(epochs, self.loss_history['cava_align'], 'orange', label='Align Loss')
        axes[1, 0].plot(epochs, self.loss_history['cava_edge'], 'purple', label='Edge Loss')
        axes[1, 0].set_title('CAVA Components');
        axes[1, 0].legend()

        axes[1, 1].plot(epochs, self.loss_history['pseudo_loss'], 'cyan', label='Pseudo Loss')
        axes[1, 1].set_title('Pseudo Label Loss');
        axes[1, 1].legend()

        if len(self.loss_history['gate_mean']) > 0:
            axes[1, 2].plot(epochs, self.loss_history['gate_mean'], 'magenta', label='Gate Mean')
            axes[1, 2].set_title('Causal Gate');
            axes[1, 2].legend()

        plt.tight_layout();
        plt.savefig(save_path);
        plt.close()

    def _plot_cava_details(self, save_path):
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('CAVA Detailed Analysis', fontsize=16)
        epochs = range(1, len(self.loss_history['total_loss']) + 1)

        axes[0, 0].plot(epochs, self.loss_history['cava_align'], 'orange', marker='o')
        axes[0, 0].set_title('InfoNCE Alignment Loss')

        axes[0, 1].plot(epochs, self.loss_history['cava_edge'], 'purple', marker='s')
        axes[0, 1].set_title('Edge Hinge Loss')

        if len(self.loss_history['gate_std']) > 0:
            mean_vals = np.array(self.loss_history['gate_mean'])
            std_vals = np.array(self.loss_history['gate_std'])
            axes[1, 0].plot(epochs, mean_vals, 'b-', label='Mean')
            axes[1, 0].fill_between(epochs, mean_vals - std_vals, mean_vals + std_vals, alpha=0.3, label='卤1 Std')
            axes[1, 0].set_title('Causal Gate Statistics')
            axes[1, 0].legend()

        axes[1, 1].plot(epochs, self.loss_history['cava_loss'], 'r-')
        axes[1, 1].set_title('Total CAVA Loss')

        plt.tight_layout();
        plt.savefig(save_path);
        plt.close()

    def _plot_validation_metrics(self, save_path):
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        epochs = range(1, len(self.loss_history['val_f1_student']) + 1)

        axes[0].plot(epochs, self.loss_history['val_f1_student'], 'b-', marker='o', label='Student')
        axes[0].plot(epochs, self.loss_history['val_f1_teacher'], 'r--', marker='s', label='Teacher')
        axes[0].set_title('F1 Score (Macro)');
        axes[0].legend()

        if len(self.loss_history['val_acc_student']) > 0:
            axes[1].plot(epochs, self.loss_history['val_acc_student'], 'b-', marker='o', label='Student')
            axes[1].plot(epochs, self.loss_history['val_acc_teacher'], 'r--', marker='s', label='Teacher')
            axes[1].set_title('Accuracy');
            axes[1].legend()

        plt.tight_layout();
        plt.savefig(save_path);
        plt.close()

    def _plot_training_dynamics(self, save_path):
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        epochs = range(1, len(self.loss_history['learning_rate']) + 1)

        axes[0, 0].plot(epochs, self.loss_history['learning_rate'], 'g-')
        axes[0, 0].set_title('Learning Rate');
        axes[0, 0].set_yscale('log')

        if len(self.loss_history['ssl_mask_ratio']) > 0:
            axes[0, 1].plot(epochs, self.loss_history['ssl_mask_ratio'], 'c-')
            axes[0, 1].set_title('SSL Mask Ratio')

        plt.tight_layout();
        plt.savefig(save_path);
        plt.close()

    def _plot_smooth_step_curves(self, save_path):
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        window = 50

        def smooth(d): return np.convolve(d, np.ones(window) / window, mode='valid') if len(d) > window else d

        if len(self.step_losses['total_loss']) > 0:
            axes[0, 0].plot(smooth(self.step_losses['total_loss']), 'b-', alpha=0.8)
            axes[0, 0].set_title('Total Loss (Step)')

            axes[0, 1].plot(smooth(self.step_losses['sup_loss']), 'g-', alpha=0.8)
            axes[0, 1].set_title('Supervised Loss (Step)')

            axes[1, 0].plot(smooth(self.step_losses['cava_loss']), 'r-', alpha=0.8)
            axes[1, 0].set_title('CAVA Loss (Step)')

            axes[1, 1].plot(smooth(self.step_losses['pseudo_loss']), 'c-', alpha=0.8)
            axes[1, 1].set_title('Pseudo Loss (Step)')

        plt.tight_layout();
        plt.savefig(save_path);
        plt.close()

    def _forward_model(self, model, v, a, *, return_aux=True, use_amp=True):
        if use_amp and self.amp_enabled:
            with amp_autocast(self.device_type, enabled=True):
                return model(v, a, return_aux=return_aux)
        return model(v, a, return_aux=return_aux)

    def _safe_forward(self, v, a, use_amp=True):
        try:
            return self._forward_model(self.model, v, a, return_aux=True, use_amp=use_amp)
        except RuntimeError as e:
            if "NaN" in str(e):
                self.nan_count += 1
                self._reset_scaler_if_needed()
                return None
            raise e

    def _forward(self, v, a, use_amp=True):
        return self._safe_forward(v, a, use_amp=use_amp)

    @torch.no_grad()
    def _validate(self, epoch: int):
        def _eval(m):
            m.eval()
            all_y, all_p = [], []
            for b in self.loader_v:
                if len(b) == 4:
                    v, a, y, _ = b
                else:
                    v, a, y = b
                v, a = v.to(self.device), a.to(self.device)
                y = y.argmax(dim=1) if y.ndim == 2 else y
                out = self._forward_model(m, v, a, return_aux=False, use_amp=False)
                logits = out['clip_logits'] if isinstance(out, dict) else out
                all_p.append(F.softmax(logits, dim=1).cpu().numpy())
                all_y.append(y.cpu().numpy())

            if len(all_y) == 0:
                return {
                    "acc": 0.0,
                    "f1_macro": 0.0,
                    "pred_majority_class": -1,
                    "pred_majority_ratio": 0.0,
                }

            y_true = np.concatenate(all_y)
            y_prob = np.concatenate(all_p)
            y_pred = y_prob.argmax(1)
            from sklearn.metrics import accuracy_score, f1_score
            pred_hist = np.bincount(y_pred, minlength=self.C)
            pred_majority_class = int(pred_hist.argmax()) if pred_hist.size > 0 else -1
            pred_majority_ratio = float(pred_hist[pred_majority_class] / max(1, len(y_pred))) if pred_majority_class >= 0 else 0.0
            return {
                "acc": accuracy_score(y_true, y_pred),
                "f1_macro": f1_score(y_true, y_pred, average='macro'),
                "pred_majority_class": pred_majority_class,
                "pred_majority_ratio": pred_majority_ratio,
            }

        # Validation runs on rank-0 only; use the underlying module to avoid
        # DDP buffer-sync collectives while other ranks wait for rank-0.
        stu = _eval(self._student_model())
        tea = _eval(self.teacher) if self.teacher else {"acc": 0, "f1_macro": 0}

        self.loss_history['val_f1_student'].append(stu['f1_macro'])
        self.loss_history['val_f1_teacher'].append(tea['f1_macro'])
        self.loss_history['val_acc_student'].append(stu['acc'])
        self.loss_history['val_acc_teacher'].append(tea['acc'])

        if self.writer is not None:
            self.writer.add_scalar('Val/F1_Student', stu['f1_macro'], epoch)
            self.writer.add_scalar('Val/F1_Teacher', tea['f1_macro'], epoch)
            self.writer.add_scalar('Val/Acc_Student', stu['acc'], epoch)

        return {"student": stu, "teacher": tea}

    def train(self):
        print("\n" + "=" * 60)
        print("馃幆 Starting Training (v3.8 Graph Safe - Full Source)...")
        print("=" * 60 + "\n")

        _epoch_times: list = []   # wall-clock seconds per epoch (train only)
        for epoch in range(self.start_epoch, self.epochs + 1):
            # DDP: shuffle must advance differently on each rank each epoch
            if self.ddp_mode and hasattr(self, '_sampler_l') and self._sampler_l is not None:
                self._sampler_l.set_epoch(epoch)
            # Unlabeled sampler must also advance so DDP shuffle differs each epoch
            if self.ddp_mode and getattr(self, '_sampler_u', None) is not None:
                self._sampler_u.set_epoch(epoch)
            # meta_val sampler: advance epoch so MLPR outer-val sees different batches
            if self.ddp_mode and getattr(self, '_sampler_mv', None) is not None:
                self._sampler_mv.set_epoch(epoch)
            # Reset meta-val iterator each epoch so it cycles through the full set
            self._val_iter_for_meta = None

            import time as _time
            _t_epoch_start = _time.perf_counter()
            tr_metrics = self._train_epoch(epoch)
            _epoch_wall = _time.perf_counter() - _t_epoch_start
            _epoch_times.append(_epoch_wall)
            if self.is_main:
                _mean_t = sum(_epoch_times) / len(_epoch_times)
                print(f"[TIME] Epoch {epoch} train time: {_epoch_wall:.1f}s | "
                      f"mean={_mean_t:.1f}s | "
                      f"ETA {_mean_t*(self.epochs-epoch):.0f}s")

            # Sync before validation so all ranks have finished the epoch
            if self.ddp_mode:
                dist.barrier()

            # Validation and checkpointing only on rank-0
            if self.is_main:
                val_res = self._validate(epoch)
                self.scheduler.step()
                self.loss_history['total_loss'].append(float(tr_metrics['total']))
                self.loss_history['sup_loss'].append(float(tr_metrics.get('sup', 0.0)))
                self.loss_history['cava_loss'].append(float(tr_metrics.get('cava_loss', 0.0)))
                self.loss_history['cava_align'].append(float(tr_metrics.get('cava_align', 0.0)))
                self.loss_history['cava_edge'].append(float(tr_metrics.get('cava_edge', 0.0)))
                self.loss_history['cava_prior'].append(float(tr_metrics.get('cava_prior', 0.0)))
                self.loss_history['cava_gate_loss'].append(float(tr_metrics.get('cava_gate_loss', 0.0)))
                self.loss_history['pseudo_loss'].append(float(tr_metrics.get('pseudo_loss', 0.0)))
                self.loss_history['ssl_mask_ratio'].append(float(tr_metrics.get('ssl_mask_ratio', 0.0)))
                self.loss_history['gate_mean'].append(float(tr_metrics.get('gate_mean', 0.0)))
                self.loss_history['gate_std'].append(float(tr_metrics.get('gate_std', 0.0)))

                # MLPR / CAVA raw diagnostics
                if self.writer:
                    self.writer.add_scalar('Train/r_align_raw', tr_metrics.get('r_align_raw', 0.0), epoch)
                    self.writer.add_scalar('Train/delta_mean', tr_metrics.get('delta_mean', 0.0), epoch)
                    self.writer.add_scalar('Train/accepted_pseudo_count', tr_metrics.get('accepted_pseudo_count', 0.0), epoch)
                    self.writer.add_scalar('CAVA/prior_loss', tr_metrics.get('cava_prior', 0.0), epoch)
                    self.writer.add_scalar('CAVA/gate_loss', tr_metrics.get('cava_gate_loss', 0.0), epoch)
                    self.writer.add_scalar('MLPR/meta_exec_ep', tr_metrics.get('meta_exec_ep', 0), epoch)
                    self.writer.add_scalar('MLPR/meta_skip_ep', tr_metrics.get('meta_skip_ep', 0), epoch)
                    self.writer.add_scalar('MLPR/weight_gen_fail_cumulative',
                                           self._mlpr_weight_gen_fail_count, epoch)
                if self.is_main:
                    _wgf = self._mlpr_weight_gen_fail_count
                    _wgf_warn = f"  *** MLPR weight-gen failures: {_wgf} ***" if _wgf > 0 else ""
                    print(
                        f"[Epoch {epoch}] r_align_raw={tr_metrics.get('r_align_raw', 0.0):.4f} "
                        f"delta_mean={tr_metrics.get('delta_mean', 0.0):.3f} "
                        f"pseudo_accept={tr_metrics.get('accepted_pseudo_count', 0.0):.0f} "
                        f"meta_exec={tr_metrics.get('meta_exec_ep', 0)} "
                        f"meta_skip={tr_metrics.get('meta_skip_ep', 0)}"
                        f"{_wgf_warn}"
                    )

                ema_epoch = None
                if self.teacher is not None:
                    ema_epoch = compute_ema_decay(
                        epoch=epoch,
                        ema_decay_base=self.ema_decay_base,
                        ema_decay_init=self.ema_decay_init,
                        warmup_epochs=self.ssl_warmup_epochs,
                    )
                    self.loss_history['ema_decay'].append(float(ema_epoch))
                    if self.writer:
                        self.writer.add_scalar('SSL/ema_decay_epoch', float(ema_epoch), epoch)
                    print(
                        f"[EMA] epoch={epoch} decay={ema_epoch:.6f} "
                        f"(init={self.ema_decay_init:.6f}, base={self.ema_decay_base:.6f})"
                    )

                current_lr = self.opt.param_groups[0]['lr']
                self.loss_history['learning_rate'].append(current_lr)
                if self.writer:
                    self.writer.add_scalar('Train/learning_rate', current_lr, epoch)
                    self.writer.add_scalar('Train/Loss', tr_metrics['total'], epoch)

                # Flush epoch accumulators into extra_logs
                wbin_epoch = {}
                for _bi in range(5):
                    _acc = self._epoch_wbin_accum[_bi]
                    wbin_epoch[f'bin_{_bi}'] = {
                        'lo': _bi*0.2, 'hi': (_bi+1)*0.2,
                        'correctness': _acc['sum_c']/max(1,_acc['n']),
                        'count': _acc['n']
                    }
                self._extra_logs['weight_bins'].append([epoch, wbin_epoch])
                if self._epoch_wsample:
                    self._extra_logs['weight_samples'].append([epoch, self._epoch_wsample[:2000]])
                if self._epoch_delta_accum:
                    _arr = np.array(self._epoch_delta_accum)
                    # Use the configured delay range so bins are not wasted on empty space.
                    # Fall back to data-driven range when CAVA is disabled or misconfigured.
                    _hist_lo = float(self.cava_cfg.get('delta_low_frames', _arr.min() - 1))
                    _hist_hi = float(self.cava_cfg.get('delta_high_frames', _arr.max() + 1))
                    if _hist_hi <= _hist_lo:
                        _hist_lo, _hist_hi = float(_arr.min()) - 0.5, float(_arr.max()) + 0.5
                    _h, _e = np.histogram(_arr, bins=20, range=(_hist_lo, _hist_hi))
                    self._extra_logs['cava_delta'].append([epoch, {
                        'mean': float(_arr.mean()), 'std': float(_arr.std()),
                        'p10': float(np.percentile(_arr, 10)),
                        'p50': float(np.percentile(_arr, 50)),
                        'p90': float(np.percentile(_arr, 90)),
                        'hist_counts': _h.tolist(), 'hist_edges': _e.tolist(),
                        # Full CAVA loss decomposition — needed to distinguish "edge is
                        # truly inactive" from "prior/gate absorb most of the penalty"
                        'loss_prior': float(tr_metrics.get('cava_prior', 0.0)),
                        'loss_gate': float(tr_metrics.get('cava_gate_loss', 0.0)),
                        'loss_align': float(tr_metrics.get('cava_align', 0.0)),
                        'loss_edge': float(tr_metrics.get('cava_edge', 0.0)),
                    }])
                f1_stu = val_res["student"]["f1_macro"]
                is_best_epoch = False
                if f1_stu > getattr(self, "best_f1", -1.0):
                    self.best_f1 = f1_stu
                    torch.save(
                        {"epoch": epoch, "state_dict": self._state_dict_for_save(), "best_f1": self.best_f1},
                        self.out_dir / 'checkpoints' / 'best_f1.pth'
                    )
                    self.no_improve = 0
                    is_best_epoch = True
                    print(f"[Rank0] New best Student F1: {self.best_f1:.4f}")
                else:
                    self.no_improve += 1

                torch.save(
                    {"epoch": epoch, "state_dict": self._state_dict_for_save()},
                    self.out_dir / 'checkpoints' / 'latest.pth'
                )
                self._record_epoch_metrics(
                    epoch=epoch,
                    tr_metrics=tr_metrics,
                    val_res=val_res,
                    current_lr=current_lr,
                    ema_epoch=ema_epoch,
                    is_best_epoch=is_best_epoch,
                    epoch_time_s=_epoch_wall,
                )
                stu_major_idx = int(val_res["student"].get("pred_majority_class", -1))
                stu_major_ratio = float(val_res["student"].get("pred_majority_ratio", 0.0))
                if 0 <= stu_major_idx < len(self.class_names):
                    stu_major_label = self.class_names[stu_major_idx]
                else:
                    stu_major_label = str(stu_major_idx)
                print(f"[Epoch {epoch}/{self.epochs}] "
                      f"Loss={tr_metrics['total']:.4f} "
                      f"(sup={tr_metrics['sup']:.3f} "
                      f"cava={tr_metrics['cava_loss']:.3f} "
                      f"pseudo={tr_metrics['pseudo_loss']:.3f} "
                      f"mask={tr_metrics['ssl_mask_ratio']:.2f}) | "
                      f"Val Stu F1={f1_stu:.4f} | Val Tea F1={val_res['teacher']['f1_macro']:.4f} | "
                      f"StuMajor={stu_major_label}({stu_major_ratio:.2f})")
            else:
                # Non-main ranks still step the scheduler to keep LR in sync
                self.scheduler.step()

            # Early-stop decision must be broadcast to all ranks
            patience = int(self.cfg.get("training", {}).get("early_stop_patience", 10))
            min_stop_epoch = int(self.cfg.get("training", {}).get("early_stop_min_epochs", 0))
            should_stop = (
                patience > 0 and
                epoch >= min_stop_epoch and
                self.no_improve >= patience
            )
            if self.ddp_mode:
                stop_flag = torch.tensor(
                    int(should_stop),
                    device=self.device
                )
                dist.broadcast(stop_flag, src=0)
                if stop_flag.item():
                    if self.is_main:
                        print(f"[Rank0] Early stopping triggered.")
                    break
            else:
                if should_stop:
                    print(f"Early stopping triggered after {patience} stagnant epochs.")
                    break

        if self.is_main:
            if self.writer:
                self.writer.close()
            self._save_loss_history()
            self._save_extra_logs()
            self._plot_all_visualizations()
        if self.ddp_mode:
            dist.barrier()
        if self.is_main:
            print("\nTraining complete!")

    def _train_epoch(self, epoch: int):
        if epoch >= self.amp_disable_epoch and self.amp_enabled:
            self.amp_enabled = False
            self.scaler = AmpGradScaler(self.device_type, enabled=False)

        self.model.train()
        if self.teacher: self.teacher.eval()

        # Reset per-epoch diagnostic accumulators
        self._epoch_wbin_accum = [{'sum_c': 0.0, 'n': 0} for _ in range(5)]
        self._epoch_delta_accum = []
        self._epoch_wsample = []
        self._epoch_meta_exec = 0    # outer steps actually executed this epoch
        self._epoch_meta_skip = 0    # outer steps skipped (all-masked) this epoch
        epoch_losses = {k: 0.0 for k in
                        ['sup_loss', 'cava_loss', 'cava_align', 'cava_edge',
                         'cava_prior', 'cava_gate_loss',
                         'pseudo_loss', 'total_loss', 'ssl_mask_ratio', 'gate_mean',
                         'gate_std', 'r_align_raw', 'accepted_pseudo_count', 'meta_update_count_ep']}
        tot = 0.0;
        nb = 0

        ssl_active = self.use_ssl and (epoch > self.ssl_warmup_epochs)
        u_iter = iter(self.loader_u) if ssl_active else None

        if self.is_main and epoch == self.ssl_warmup_epochs + 1:
            print(f"[SSL] Warmup done — SSL now active (thresh={self.ssl_final_thresh:.2f}, "
                  f"ema_decay={self.ema_decay_base:.4f}, "
                  f"λ_u schedule={self.lambda_u_schedule}, "
                  f"λ_u_max={self.lambda_u_max:.2f}, "
                  f"ramp_epochs={self.lambda_u_ramp_epochs})")
        if self.is_main and not ssl_active and epoch <= self.ssl_warmup_epochs:
            print(f"[SSL] Warmup epoch {epoch}/{self.ssl_warmup_epochs} — SSL inactive")
        if self.is_main and ssl_active and self.lambda_u_schedule != "constant":
            print(f"[SSL] λ_u(epoch={epoch}) = {self._lambda_u_at(epoch):.4f} "
                  f"(max {self.lambda_u_max:.2f}, schedule={self.lambda_u_schedule})")

        pbar = tqdm(self.loader_l, desc=f"Epoch {epoch}/{self.epochs}", disable=not self.is_main)
        self.opt.zero_grad(set_to_none=True)
        num_batches = len(self.loader_l)
        for bi, b in enumerate(pbar):
            if isinstance(b, (list, tuple)) and len(b) == 4:
                v, a, y, _ = b
            else:
                v, a, y = b

            v, a = v.to(self.device), a.to(self.device)
            y = y.argmax(dim=1).to(self.device) if y.ndim == 2 else y.to(self.device)

            # ---------------- 1. Supervised Forward ----------------
            with amp_autocast(self.device_type, enabled=self.amp_enabled):
                out = self._safe_forward(v, a, use_amp=self.amp_enabled)
                if out is None: continue

                if torch.isnan(out['clip_logits']).any():
                    self.nan_count += 1;
                    self._reset_scaler_if_needed();
                    continue

                sup_loss = self.criterion(out['clip_logits'], y)

                if sup_loss is None or torch.isnan(sup_loss):
                    self.nan_count += 1;
                    self._reset_scaler_if_needed();
                    continue

                cava_loss = torch.tensor(0.0, device=self.device)
                if self.cava_enabled:
                    try:
                        g = out.get("causal_gate", None)
                        self.cava_loss_fn.update_cfg(self.cava_cfg)
                        c_logs = self.cava_loss_fn(out)
                        cava_loss = c_logs["loss_total"]
                        epoch_losses['cava_align'] += c_logs["loss_align"].detach().item()
                        # r_align is the unweighted InfoNCE value; use it to judge alignment
                        # independent of how lambda_align is tuned (loss_align = beta * r_align)
                        epoch_losses['r_align_raw'] += c_logs["r_align"].detach().item()
                        epoch_losses['cava_edge']  += c_logs["loss_edge"].detach().item()
                        epoch_losses['cava_prior'] += c_logs["loss_prior"].detach().item()
                        epoch_losses['cava_gate_loss'] += c_logs["loss_gate"].detach().item()
                        if g is not None:
                            epoch_losses['gate_mean'] += g.mean().item()
                            epoch_losses['gate_std'] += g.std().item()
                    except Exception as _cava_exc:
                        import traceback as _tb
                        self._cava_fail_count = getattr(self, '_cava_fail_count', 0) + 1
                        if self._cava_fail_count == 1:
                            _tb.print_exc()
                            print("[CAVA] Loss computation FAILED on first call — "
                                  "CAVA will be zeroed this step. Fix the root cause.")
                        elif self._cava_fail_count % 50 == 0:
                            print(f"[CAVA] Loss still failing (#{self._cava_fail_count}): {_cava_exc}")

                    # Delta accumulation is kept outside the broad except so that
                    # a Tensor truthiness error here does not silence CAVA loss bugs.
                    if self.is_main and self.cava_enabled:
                        # Use explicit None checks — `out.get(key) or fallback` raises
                        # RuntimeError when the value is a multi-element Tensor.
                        _df = out.get('delay_frames_cont', None)
                        if _df is None:
                            _df = out.get('delay_frames', None)
                        if _df is None:
                            _df = out.get('pred_delay', None)
                        if _df is not None:
                            self._epoch_delta_accum.extend(_df.detach().cpu().reshape(-1).tolist())

                epoch_losses['sup_loss'] += sup_loss.item()
                epoch_losses['cava_loss'] += cava_loss.item()

                if self.mlpr_enabled:
                    with torch.no_grad():
                        self._last_labeled_batch = (v.detach(), a.detach(), y.detach())

                sup_total = sup_loss + cava_loss
                total_loss = sup_total
                do_step = ((bi + 1) % self.grad_accum_steps == 0) or ((bi + 1) == num_batches)
                sup_loss_scaled = sup_total / float(self.grad_accum_steps)

                # Backprop the supervised branch BEFORE the unlabeled forward so
                # graph1 is freed before graph2 is built (avoids in-place version
                # mismatch that surfaced in epoch 6).
                #
                # DDP note: in DDP mode with two separate backward calls, the
                # gradient reduction hook fires twice on the same parameter buckets
                # → "backward through graph a second time" crash.
                # Fix: use model.no_sync() for the supervised backward when SSL is
                # active so that DDP gradient sync is deferred to the unlabeled
                # backward (which carries both accumulated gradients together).
                # When SSL is inactive the supervised backward is the only one and
                # must trigger sync normally.
                _sup_sync_ctx = (
                    self.model.no_sync()
                    if (self.ddp_mode and ssl_active and hasattr(self.model, "no_sync"))
                    else contextlib.nullcontext()
                )
                with _sup_sync_ctx:
                    if self.scaler.is_enabled():
                        self.scaler.scale(sup_loss_scaled).backward()
                    else:
                        sup_loss_scaled.backward()

                # ---------------- 2. SSL / MLPR Forward ----------------
                pseudo_loss = torch.tensor(0.0, device=self.device)
                if ssl_active:
                    try:
                        bu = next(u_iter)
                    except StopIteration:
                        u_iter = iter(self.loader_u)
                        bu = next(u_iter)

                    if len(bu) == 4:
                        vu, au, y_u_true, ids_u = bu
                    else:
                        vu, au, y_u_true = bu
                        ids_u = None
                    vu, au = vu.to(self.device), au.to(self.device)

                    with torch.no_grad():
                        tout = self.teacher(vu, au)
                        t_prob = F.softmax(tout["clip_logits"], dim=1)

                        # Distribution alignment (only when strategy requests it)
                        if self.ssl_strategy.use_dist_align and self._use_dist_align:
                            q = t_prob.mean(dim=0).clamp(min=1e-8)
                            p_target = self._pi / (q + 1e-8)
                            p_target = p_target / p_target.sum()
                            t_prob = t_prob * p_target.unsqueeze(0)
                            t_prob = t_prob / t_prob.sum(dim=1, keepdim=True)

                        t_idx, mask = self.ssl_strategy.build_pseudo_targets(t_prob)

                        # One-time diagnostic: print teacher confidence on first SSL step
                        if self.is_main and bi == 0 and epoch == self.ssl_warmup_epochs + 1:
                            t_max = t_prob.max(dim=1).values
                            print(f"[SSL-diag] teacher max_prob: "
                                  f"mean={t_max.mean():.3f} "
                                  f"p50={t_max.median():.3f} "
                                  f"p90={t_max.quantile(0.9):.3f} "
                                  f"thresh={self.ssl_final_thresh:.2f} "
                                  f"mask_rate={(mask > 0.5).float().mean():.2f}")

                    sout_u = self.model(vu, au)
                    s_logits_u = sout_u["clip_logits"]

                    if torch.isnan(s_logits_u).any():
                        self.nan_count += 1
                        self._reset_scaler_if_needed()
                        continue

                    alpha_ce, alpha_kl = self._ssl_loss_mix(epoch)
                    # λ_u(t) = λ_u_max · r(t); see `_lambda_u_at` for r(t).
                    lambda_u_t = self._lambda_u_at(epoch)
                    w_eff = self.ssl_strategy.compute_sample_weights(
                        t_prob, mask,
                        student_out=sout_u,
                        ids_u=ids_u,
                        labeled_batch=(v, a, y),
                        unlabeled_batch=(vu, au),
                        alpha_ce=alpha_ce,
                        alpha_kl=alpha_kl,
                        temperature=self.ssl_temp,
                        lambda_u=lambda_u_t,
                    )

                    pseudo_loss = self.ssl_strategy.compute_unsup_loss(
                        s_logits_u,
                        t_idx,
                        w_eff,
                        teacher_prob=t_prob,
                        alpha_ce=alpha_ce,
                        alpha_kl=alpha_kl,
                        temperature=self.ssl_temp,
                    )
                    # Track weight bins vs pseudo-label correctness
                    if self.is_main and y_u_true is not None:
                        with torch.no_grad():
                            y_u_int = (y_u_true.argmax(dim=1) if y_u_true.ndim == 2
                                       else y_u_true).to(self.device)
                            correct = (t_idx == y_u_int).float().cpu()
                            w_cpu = w_eff.detach().cpu()
                            # Sample up to 256 weights per step for violin plots
                            self._epoch_wsample.extend(w_cpu[:256].tolist())
                            for _bi in range(5):
                                _lo, _hi = _bi * 0.2, (_bi + 1) * 0.2 if _bi < 4 else 2.0
                                _sel = (w_cpu >= _lo) & (w_cpu < _hi)
                                if _sel.sum() > 0:
                                    self._epoch_wbin_accum[_bi]['sum_c'] += correct[_sel].sum().item()
                                    self._epoch_wbin_accum[_bi]['n'] += int(_sel.sum().item())
                    epoch_losses["ssl_mask_ratio"] += (
                        (mask > 0.5).float().mean().item()
                    )
                    epoch_losses["accepted_pseudo_count"] += float((mask > 0.5).sum().item())
                    self.ssl_strategy.update_method_state(
                        teacher_prob=t_prob, pseudo_labels=t_idx, mask=mask
                    )

                    # Use the scheduled λ_u(t) computed above so the total
                    # loss matches  L_total = L_sup + λ_DAVA·L_DAVA + λ_u(t)·E[w_φ·L_u].
                    unsup_term = lambda_u_t * pseudo_loss
                    total_loss = sup_total + unsup_term
                    unsup_loss_scaled = unsup_term / float(self.grad_accum_steps)
                    # Guard: only backward if the loss has a grad_fn.
                    # When the entire unlabeled batch is below threshold the strategy
                    # returns student_logits.sum()*0 (differentiable zero), so
                    # grad_fn is always set — this catches any remaining edge-cases.
                    if unsup_loss_scaled.grad_fn is not None:
                        if self.scaler.is_enabled():
                            self.scaler.scale(unsup_loss_scaled).backward()
                        else:
                            unsup_loss_scaled.backward()

                epoch_losses['pseudo_loss'] += pseudo_loss.item()
                epoch_losses['total_loss'] += total_loss.item()

                # ---------------- 3. Optimization with Guard ----------------
                if do_step:
                    if self.scaler.is_enabled():
                        self.scaler.unscale_(self.opt)
                        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                        if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                            self.nan_count += 1
                            self._reset_scaler_if_needed()
                            continue
                        self.scaler.step(self.opt)
                        self.scaler.update()
                    else:
                        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                        if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                            self.nan_count += 1
                            self.opt.zero_grad()
                            continue
                        self.opt.step()

                self.consecutive_nan = 0
                if do_step:
                    self.total_steps += 1
                    self._ema_update(epoch)
                    if (self.total_steps % 100 == 0) and hasattr(self, "_last_ema_decay"):
                        if self.writer is not None:
                            self.writer.add_scalar("SSL/ema_decay", float(self._last_ema_decay), self.total_steps)
                        print(f"[EMA] step={self.total_steps} epoch={epoch} decay={self._last_ema_decay:.6f}")

                    # Strategy-level post-step hook (MLPR meta-update, etc.)
                    self.ssl_strategy.after_optimizer_step(self, self.total_steps)

                    self.opt.zero_grad(set_to_none=True)

                tot += total_loss.item()
                nb += 1
                pbar.set_postfix(loss=f"{total_loss.item():.4f}", nan=self.nan_count)

        # Counts should not be averaged — save before dividing
        _accepted_pseudo_total = epoch_losses.pop('accepted_pseudo_count')
        epoch_losses.pop('meta_update_count_ep')  # placeholder; real count via _epoch_meta_exec
        for k in epoch_losses:
            epoch_losses[k] /= max(1, nb)
        # delta_mean from delay accumulator (raw per-sample values, not divided by nb)
        _delta_mean = (float(sum(self._epoch_delta_accum) / len(self._epoch_delta_accum))
                       if self._epoch_delta_accum else 0.0)
        return {
            "total": epoch_losses['total_loss'],
            "sup": epoch_losses['sup_loss'],
            "cava_loss": epoch_losses['cava_loss'],
            "cava_align": epoch_losses['cava_align'],
            "cava_edge": epoch_losses['cava_edge'],
            "cava_prior": epoch_losses['cava_prior'],
            "cava_gate_loss": epoch_losses['cava_gate_loss'],
            "pseudo_loss": epoch_losses['pseudo_loss'],
            "ssl_mask_ratio": epoch_losses['ssl_mask_ratio'],
            "gate_mean": epoch_losses['gate_mean'],
            "gate_std": epoch_losses['gate_std'],
            # Raw CAVA / MLPR diagnostics
            "r_align_raw": epoch_losses['r_align_raw'],
            "delta_mean": _delta_mean,
            "accepted_pseudo_count": _accepted_pseudo_total,
            # Per-epoch meta update counts (exec = outer step taken; skip = all-masked)
            "meta_exec_ep": self._epoch_meta_exec,
            "meta_skip_ep": self._epoch_meta_skip,
            "meta_update_count": self.meta_update_count,  # cumulative
        }


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--output', type=str, default='./outputs/train_v3')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--seed', type=int, default=None,
                        help='Override the seed in the config file')
    args = parser.parse_args()

    with open(args.config, 'r') as f: cfg = yaml.safe_load(f)
    if args.seed is not None:
        cfg['seed'] = args.seed
    trainer = StrongTrainer(cfg, args.output, args.checkpoint)
    trainer.train()

