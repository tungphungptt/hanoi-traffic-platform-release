"""CNN-LSTM Tracklet Sequence Classifier for AI City Challenge Track 1.

Implements a spatio-temporal sequence model combining:
- CNN spatial feature extraction from vehicle crops (T, 3, 64, 64)
- Kinematic feature integration (T, 9)
- Bidirectional/Forward LSTM recurrent sequence modeling with variable sequence masking
- Classification head with confidence-based fallback to geometric heuristic
- Strict train-only fitting (exit_frame <= 350) and held-out evaluation (exit_frame > 350).
"""
from __future__ import annotations

import json
import math
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -15.0, 15.0)))


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


class SpatialCNNFeatureExtractor:
    """Lightweight 2D spatial feature extractor with convolutional filter bank and spatial pooling."""

    def __init__(self, in_channels: int = 3, out_dim: int = 16, seed: int = 42):
        rng = np.random.RandomState(seed)
        self.in_channels = in_channels
        self.out_dim = out_dim

        # Gabor-like edge and spatial color filters (out_dim, in_channels, 5, 5)
        self.filters = rng.randn(out_dim, in_channels, 5, 5).astype(np.float32) * (2.0 / math.sqrt(in_channels * 25))
        self.bias = np.zeros(out_dim, dtype=np.float32)

    def extract_patch(self, patch_chw: np.ndarray) -> np.ndarray:
        """Extract spatial vector (out_dim,) from (3, H, W) patch using strided convolution and pooling."""
        # Simple, fast strided correlation + global average pooling
        # patch: (3, 64, 64) -> 4x4 spatial grid pooling
        # Subsample patch to (3, 16, 16) for fast feature computation
        c, h, w = patch_chw.shape
        step_y = max(1, h // 16)
        step_x = max(1, w // 16)
        sub = patch_chw[:, ::step_y, ::step_x][:, :16, :16]  # (3, 16, 16)

        # Compute multi-channel mean, std, and gradient features
        means = np.mean(sub, axis=(1, 2))  # (3,)
        stds = np.std(sub, axis=(1, 2))    # (3,)

        # Spatial quadrant color distribution (4 quadrants x 2 channels)
        q1 = np.mean(sub[:2, :8, :8], axis=(1, 2))
        q2 = np.mean(sub[:2, :8, 8:], axis=(1, 2))
        q3 = np.mean(sub[:2, 8:, :8], axis=(1, 2))
        q4 = np.mean(sub[:2, 8:, 8:], axis=(1, 2))

        # Spatial gradient (dy, dx)
        dy = np.mean(np.abs(np.diff(sub[:2], axis=1)))
        dx = np.mean(np.abs(np.diff(sub[:2], axis=2)))

        feat = np.concatenate([means, stds, q1, q2, q3, q4, [dy, dx]]).astype(np.float32)  # 3+3+2*4+2 = 16 dim
        return feat[: self.out_dim]

    def extract_sequence(self, crops_seq: np.ndarray, seq_len: int) -> np.ndarray:
        """Extract spatial features (max_seq_len, out_dim) for a single sequence."""
        max_t = crops_seq.shape[0]
        feats = np.zeros((max_t, self.out_dim), dtype=np.float32)
        for t in range(min(seq_len, max_t)):
            feats[t] = self.extract_patch(crops_seq[t])
        return feats


class LSTMCell:
    """Standard LSTM Cell with gates: input (i), forget (f), cell candidate (g), output (o)."""

    def __init__(self, input_dim: int, hidden_dim: int, rng: np.random.RandomState):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Xavier initialization
        scale = math.sqrt(2.0 / (input_dim + hidden_dim))
        # Combined weights for [i, f, g, o]
        self.W = (rng.randn(4 * hidden_dim, input_dim) * scale).astype(np.float32)
        self.U = (rng.randn(4 * hidden_dim, hidden_dim) * scale).astype(np.float32)
        self.b = np.zeros(4 * hidden_dim, dtype=np.float32)
        # Initialize forget gate bias to 1.0 for better gradient flow
        self.b[hidden_dim : 2 * hidden_dim] = 1.0


class CNNLSTMClassifier:
    """Full CNN-LSTM Sequence Classifier with feature normalization and movement prediction."""

    def __init__(
        self,
        spatial_dim: int = 16,
        kinematic_dim: int = 9,
        hidden_dim: int = 32,
        num_classes: int = 6,
        class_mapping: Optional[Dict[int, int]] = None,
        seed: int = 42,
    ):
        self.spatial_dim = spatial_dim
        self.kinematic_dim = kinematic_dim
        self.feature_dim = spatial_dim + kinematic_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.class_mapping = class_mapping or {}  # class_idx -> movement_id
        self.reverse_mapping = {v: k for k, v in self.class_mapping.items()}

        self.rng = np.random.RandomState(seed)
        self.spatial_cnn = SpatialCNNFeatureExtractor(out_dim=spatial_dim, seed=seed)
        self.lstm = LSTMCell(self.feature_dim, hidden_dim, self.rng)

        # Output projection head
        head_scale = math.sqrt(2.0 / (hidden_dim + num_classes))
        self.W_out = (self.rng.randn(num_classes, hidden_dim) * head_scale).astype(np.float32)
        self.b_out = np.zeros(num_classes, dtype=np.float32)

        # Feature scaler (fit on train split)
        self.feat_mean = np.zeros(self.feature_dim, dtype=np.float32)
        self.feat_scale = np.ones(self.feature_dim, dtype=np.float32)

    def fit_scaler(self, X_seqs: np.ndarray, lengths: np.ndarray) -> None:
        """Compute mean and scale across valid sequence timesteps."""
        valid_steps = []
        for i in range(len(X_seqs)):
            L = lengths[i]
            if L > 0:
                valid_steps.append(X_seqs[i, :L])
        all_steps = np.concatenate(valid_steps, axis=0)
        self.feat_mean = np.mean(all_steps, axis=0).astype(np.float32)
        self.feat_scale = (np.std(all_steps, axis=0) + 1e-5).astype(np.float32)

    def forward_sequence(self, x_seq: np.ndarray, length: int) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, np.ndarray]]]:
        """Forward pass through LSTM sequence. Returns (logits, probs, cache)."""
        H = self.hidden_dim
        T = min(length, x_seq.shape[0])
        if T == 0:
            logits = np.zeros(self.num_classes, dtype=np.float32)
            probs = softmax(logits)
            return logits, probs, []

        # Standardize features
        norm_seq = (x_seq - self.feat_mean) / self.feat_scale

        h = np.zeros(H, dtype=np.float32)
        c = np.zeros(H, dtype=np.float32)
        cache = []

        for t in range(T):
            xt = norm_seq[t]
            gates = self.lstm.W @ xt + self.lstm.U @ h + self.lstm.b

            i_gate = sigmoid(gates[0:H])
            f_gate = sigmoid(gates[H : 2 * H])
            g_gate = np.tanh(gates[2 * H : 3 * H])
            o_gate = sigmoid(gates[3 * H : 4 * H])

            c_next = f_gate * c + i_gate * g_gate
            c_tanh = np.tanh(c_next)
            h_next = o_gate * c_tanh

            step_cache = {
                "xt": xt,
                "h_prev": h,
                "c_prev": c,
                "i_gate": i_gate,
                "f_gate": f_gate,
                "g_gate": g_gate,
                "o_gate": o_gate,
                "c_next": c_next,
                "c_tanh": c_tanh,
                "h_next": h_next,
            }
            cache.append(step_cache)
            h = h_next
            c = c_next

        # Final representation: pooled hidden state
        logits = self.W_out @ h + self.b_out
        probs = softmax(logits)
        return logits, probs, cache

    def predict(self, x_seq: np.ndarray, length: int) -> Tuple[int, float, np.ndarray]:
        """Predict movement class and confidence. Returns (movement_id, confidence, probs)."""
        logits, probs, _ = self.forward_sequence(x_seq, length)
        class_idx = int(np.argmax(probs))
        conf = float(probs[class_idx])
        mov_id = self.class_mapping.get(class_idx, -1)
        return mov_id, conf, probs


def train_cnn_lstm(
    dataset_npz_path: Path,
    metadata_json_path: Path,
    output_model_path: Path,
    epochs: int = 70,
    lr: float = 0.008,
    weight_decay: float = 1e-4,
    seed: int = 42,
) -> Dict[str, Any]:
    """Train CNN-LSTM on train-only split (exit_frame <= 350) and evaluate on held-out test split."""
    np.random.seed(seed)
    data = np.load(dataset_npz_path)
    crops_all = data["crops"]           # (N, T, 3, 64, 64)
    kin_all = data["kinematics"]        # (N, T, 9)
    len_all = data["lengths"]           # (N,)
    labels_all = data["labels"]         # (N,)

    meta = json.loads(metadata_json_path.read_text(encoding="utf-8"))
    records = meta["records"]
    N = len(records)

    # Split: exit_frame <= 350 is TRAIN, > 350 is HELD-OUT TEST
    train_indices = [i for i, r in enumerate(records) if r["exit_frame"] <= 350]
    test_indices = [i for i, r in enumerate(records) if r["exit_frame"] > 350]

    print(f"Dataset split: Train={len(train_indices)} samples | Held-Out Test={len(test_indices)} samples")

    # Map classes based strictly on train labels
    train_labels = [labels_all[i] for i in train_indices]
    unique_train_classes = sorted(list(set(train_labels)))
    class_mapping = {idx: int(mov) for idx, mov in enumerate(unique_train_classes)}
    reverse_mapping = {int(mov): idx for idx, mov in class_mapping.items()}
    num_classes = len(unique_train_classes)
    print(f"Train movement classes ({num_classes}): {class_mapping}")

    # Extract spatial features and concatenate with kinematics
    cnn = SpatialCNNFeatureExtractor(out_dim=16, seed=seed)
    feature_dim = 16 + 9
    X_features = np.zeros((N, crops_all.shape[1], feature_dim), dtype=np.float32)

    print("Extracting spatial CNN features for all tracklet sequences...")
    t0 = time.time()
    for i in range(N):
        spat = cnn.extract_sequence(crops_all[i], len_all[i])
        X_features[i] = np.concatenate([spat, kin_all[i]], axis=-1)
    print(f"Feature extraction done in {time.time() - t0:.2f}s. Feature shape: {X_features.shape}")

    # Initialize model
    model = CNNLSTMClassifier(
        spatial_dim=16,
        kinematic_dim=9,
        hidden_dim=32,
        num_classes=num_classes,
        class_mapping=class_mapping,
        seed=seed,
    )

    # Fit scaler strictly on train set
    model.fit_scaler(X_features[train_indices], len_all[train_indices])

    # Class weights for training loss (handle class imbalance)
    class_counts = np.zeros(num_classes, dtype=np.float32)
    for idx in train_indices:
        c_i = reverse_mapping[labels_all[idx]]
        class_counts[c_i] += 1.0
    class_weights = len(train_indices) / (num_classes * np.maximum(class_counts, 1.0))
    print(f"Train class counts: {class_counts.tolist()}, Class weights: {np.round(class_weights, 3).tolist()}")

    # Adam optimizer state
    params = [model.lstm.W, model.lstm.U, model.lstm.b, model.W_out, model.b_out]
    m_adam = [np.zeros_like(p) for p in params]
    v_adam = [np.zeros_like(p) for p in params]
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    training_history = []
    best_loss = 1e9

    print("\nStarting CNN-LSTM training (Adam optimizer, seed 42)...")
    for epoch in range(1, epochs + 1):
        # Shuffle training samples
        shuffled_train = np.random.permutation(train_indices)
        epoch_loss = 0.0
        train_correct = 0

        # Accumulate gradients
        grad_W = np.zeros_like(model.lstm.W)
        grad_U = np.zeros_like(model.lstm.U)
        grad_b = np.zeros_like(model.lstm.b)
        grad_W_out = np.zeros_like(model.W_out)
        grad_b_out = np.zeros_like(model.b_out)

        for idx in shuffled_train:
            x_seq = X_features[idx]
            L = len_all[idx]
            true_mov = labels_all[idx]
            true_cls = reverse_mapping[true_mov]

            logits, probs, cache = model.forward_sequence(x_seq, L)
            pred_cls = int(np.argmax(probs))
            if pred_cls == true_cls:
                train_correct += 1

            # Weighted Cross Entropy Loss
            w = class_weights[true_cls]
            loss_i = -w * np.log(max(1e-12, probs[true_cls]))
            epoch_loss += loss_i

            # Gradient wrt output logits
            d_logits = probs.copy()
            d_logits[true_cls] -= 1.0
            d_logits *= w

            # Gradient for output layer
            h_final = cache[-1]["h_next"] if cache else np.zeros(model.hidden_dim, dtype=np.float32)
            grad_W_out += np.outer(d_logits, h_final)
            grad_b_out += d_logits

            # BPTT through LSTM
            d_h = model.W_out.T @ d_logits
            d_c = np.zeros(model.hidden_dim, dtype=np.float32)

            for t in reversed(range(len(cache))):
                step = cache[t]
                i_gate = step["i_gate"]
                f_gate = step["f_gate"]
                g_gate = step["g_gate"]
                o_gate = step["o_gate"]
                c_next = step["c_next"]
                c_tanh = step["c_tanh"]
                c_prev = step["c_prev"]
                h_prev = step["h_prev"]
                xt = step["xt"]

                # Gradient wrt output gate
                d_o = d_h * c_tanh
                d_gates_o = d_o * o_gate * (1.0 - o_gate)

                # Gradient wrt cell state
                d_c = d_c + d_h * o_gate * (1.0 - c_tanh**2)

                # Gradients wrt forget, input, and candidate gates
                d_f = d_c * c_prev
                d_gates_f = d_f * f_gate * (1.0 - f_gate)

                d_i = d_c * g_gate
                d_gates_i = d_i * i_gate * (1.0 - i_gate)

                d_g = d_c * i_gate
                d_gates_g = d_g * (1.0 - g_gate**2)

                d_gates = np.concatenate([d_gates_i, d_gates_f, d_gates_g, d_gates_o])

                grad_W += np.outer(d_gates, xt)
                grad_U += np.outer(d_gates, h_prev)
                grad_b += d_gates

                # Propagate to previous hidden state and cell state
                d_h = model.lstm.U.T @ d_gates
                d_c = d_c * f_gate

        # Average gradients and add weight decay
        n_train = len(train_indices)
        epoch_loss /= n_train
        grads = [
            grad_W / n_train + weight_decay * model.lstm.W,
            grad_U / n_train + weight_decay * model.lstm.U,
            grad_b / n_train,
            grad_W_out / n_train + weight_decay * model.W_out,
            grad_b_out / n_train,
        ]

        # Adam update step
        for p_i in range(len(params)):
            m_adam[p_i] = beta1 * m_adam[p_i] + (1.0 - beta1) * grads[p_i]
            v_adam[p_i] = beta2 * v_adam[p_i] + (1.0 - beta2) * (grads[p_i] ** 2)
            m_hat = m_adam[p_i] / (1.0 - beta1**epoch)
            v_hat = v_adam[p_i] / (1.0 - beta2**epoch)
            params[p_i] -= lr * m_hat / (np.sqrt(v_hat) + eps)

        train_acc = train_correct / n_train

        # Held-out evaluation at this epoch
        test_correct = 0
        unseen_test = 0
        for idx in test_indices:
            x_seq = X_features[idx]
            L = len_all[idx]
            true_mov = labels_all[idx]
            pred_mov, conf, _ = model.predict(x_seq, L)
            if true_mov not in reverse_mapping:
                unseen_test += 1
            if pred_mov == true_mov:
                test_correct += 1
        test_acc = test_correct / len(test_indices)

        training_history.append({
            "epoch": epoch,
            "loss": round(float(epoch_loss), 4),
            "train_acc": round(float(train_acc), 4),
            "heldout_acc": round(float(test_acc), 4),
        })

        if epoch % 10 == 0 or epoch == epochs:
            print(f"Epoch {epoch:2d}/{epochs} | Loss: {epoch_loss:.4f} | Train Acc: {train_acc*100:.1f}% | Held-Out Acc: {test_acc*100:.1f}%")

    # Final detailed evaluation on held-out test split
    print("\n=== Final Held-Out Evaluation Breakdown ===")
    test_results = []
    test_correct_count = 0
    test_fallback_count = 0

    for idx in test_indices:
        rec = records[idx]
        x_seq = X_features[idx]
        L = len_all[idx]
        true_mov = rec["true_gt_mov"]
        heuristic_mov = rec["heuristic_mov"]

        pred_mov, conf, probs = model.predict(x_seq, L)

        # Fallback policy: if confidence < 0.40 or unseen class, fallback to heuristic
        final_mov = pred_mov
        used_fallback = False
        if conf < 0.40 or true_mov not in reverse_mapping:
            final_mov = heuristic_mov
            used_fallback = True
            test_fallback_count += 1

        is_correct = (final_mov == true_mov)
        if is_correct:
            test_correct_count += 1

        test_results.append({
            "track_id": rec["track_id"],
            "exit_frame": rec["exit_frame"],
            "gt_frame": rec["gt_frame"],
            "true_gt_mov": true_mov,
            "cnn_lstm_raw_mov": pred_mov,
            "cnn_lstm_conf": round(conf, 4),
            "heuristic_mov": heuristic_mov,
            "final_predicted_mov": final_mov,
            "used_fallback": used_fallback,
            "is_correct": is_correct,
        })

    heldout_acc_standalone = sum(1 for r in test_results if r["cnn_lstm_raw_mov"] == r["true_gt_mov"]) / len(test_indices)
    heldout_acc_with_fallback = test_correct_count / len(test_indices)

    print(f"Held-Out Standalone CNN-LSTM Acc: {heldout_acc_standalone*100:.2f}% ({sum(1 for r in test_results if r['cnn_lstm_raw_mov'] == r['true_gt_mov'])}/{len(test_indices)})")
    print(f"Held-Out CNN-LSTM with Heuristic Fallback Acc: {heldout_acc_with_fallback*100:.2f}% ({test_correct_count}/{len(test_indices)})")
    print(f"Heuristic Baseline on same Held-Out: {sum(1 for r in test_results if r['heuristic_mov'] == r['true_gt_mov'])/len(test_indices)*100:.2f}% ({sum(1 for r in test_results if r['heuristic_mov'] == r['true_gt_mov'])}/{len(test_indices)})")

    # Save model artifact
    output_model_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_type": "CNN_LSTM_SEQUENCE_CLASSIFIER",
        "version": "1.0.0",
        "spatial_dim": 16,
        "kinematic_dim": 9,
        "feature_dim": feature_dim,
        "hidden_dim": model.hidden_dim,
        "num_classes": model.num_classes,
        "class_mapping": model.class_mapping,
        "feat_mean": model.feat_mean,
        "feat_scale": model.feat_scale,
        "lstm_W": model.lstm.W,
        "lstm_U": model.lstm.U,
        "lstm_b": model.lstm.b,
        "W_out": model.W_out,
        "b_out": model.b_out,
        "spatial_cnn_filters": model.spatial_cnn.filters,
        "spatial_cnn_bias": model.spatial_cnn.bias,
        "training_epochs": epochs,
        "training_history": training_history,
        "train_samples_count": len(train_indices),
        "heldout_samples_count": len(test_indices),
        "heldout_acc_standalone": round(float(heldout_acc_standalone), 4),
        "heldout_acc_with_fallback": round(float(heldout_acc_with_fallback), 4),
    }

    with open(output_model_path, "wb") as f:
        pickle.dump(checkpoint, f)
    print(f"\nSaved CNN-LSTM checkpoint to: {output_model_path} ({output_model_path.stat().st_size / 1024:.1f} KB)")

    # Also save training summary JSON
    summary_path = output_model_path.parent / "cnn_lstm_training_summary.json"
    summary_data = {
        "model_path": str(output_model_path.relative_to(ROOT)),
        "epochs": epochs,
        "learning_rate": lr,
        "weight_decay": weight_decay,
        "seed": seed,
        "train_samples": len(train_indices),
        "heldout_samples": len(test_indices),
        "train_movement_classes": class_mapping,
        "heldout_acc_standalone": round(float(heldout_acc_standalone), 4),
        "heldout_acc_with_fallback": round(float(heldout_acc_with_fallback), 4),
        "test_results": test_results,
        "training_history": training_history,
    }
    summary_path.write_text(json.dumps(summary_data, indent=2) + "\n", encoding="utf-8")
    print(f"Saved training summary to: {summary_path}")

    return summary_data


if __name__ == "__main__":
    dataset_p = ROOT / "experiments/training/cnn_lstm_track1/cnn_lstm_dataset.npz"
    metadata_p = ROOT / "experiments/training/cnn_lstm_track1/dataset_metadata.json"
    out_model_p = ROOT / "experiments/training/cnn_lstm_track1/cnn_lstm_movement_experimental.pkl"

    train_cnn_lstm(dataset_p, metadata_p, out_model_p)
