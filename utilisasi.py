import pandas as pd, numpy as np
import matplotlib.pyplot as plt
import json, torch, logging, os, cv2, stat, time, seaborn as sns
from collections import defaultdict
from pathlib import Path
from sklearn.metrics import confusion_matrix, auc
from typing import List, Dict, Optional, Tuple
from glob import glob


class LoggerManager:
    _loggers = {}
    def __init__(self, log_file: Path):
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)

        logger_name = str(log_file.resolve())
        if logger_name in LoggerManager._loggers:
            self.logger = LoggerManager._loggers[logger_name]
        else:
            self.logger = logging.getLogger(logger_name)
            self.logger.setLevel(logging.INFO)
            self.logger.propagate = False  # Cegah log ganda ke root logger

            file_handler = logging.FileHandler(log_file, mode='a')
            file_handler.setLevel(logging.INFO)
            file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)

            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            console_formatter = logging.Formatter('%(levelname)s: %(message)s')
            console_handler.setFormatter(console_formatter)
            self.logger.addHandler(console_handler)

            LoggerManager._loggers[logger_name] = self.logger

    def info(self, message: str):
        self.logger.info(message)

    def error(self, message: str):
        self.logger.error(message)

    def debug(self, message: str):
        self.logger.debug(message)

class CSVRemaker:
    def __init__(self, csv_path, olah_path, log_path, num_folds, data, model_type, size):
        self.csv_path = csv_path
        self.num_folds = num_folds
        self.data = data
        self.model_type = model_type
        self.size = size
        self.output_path = olah_path / self.model_type / f'ukuran_{self.size}' / "hasil_csv"
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.log_path =  log_path / "log_csv.txt"
        self.logger = LoggerManager(log_file=self.log_path)
        self.ensure_directory_exists(self.output_path)
        self.columns = [
            "train/box_loss", "train/cls_loss", "train/dfl_loss",
            "val/box_loss", "val/cls_loss", "val/dfl_loss",
            "metrics/precision(B)", "metrics/recall(B)",
            "metrics/mAP50(B)", "metrics/mAP50-95(B)"
        ]

    def ensure_directory_exists(self, directory_path):
        """Memastikan direktori ada dan memiliki izin yang tepat."""
        try:
            # Coba buat direktori dengan izin penuh
            directory_path.mkdir(parents=True, exist_ok=True)
            
            # Atur izin penuh untuk direktori (Windows)
            if os.name == 'nt':  # Windows
                os.chmod(directory_path, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
                
            return True
        except PermissionError as e:
            # Coba pendekatan alternatif jika izin ditolak
            self.logger.info(f"Izin ditolak saat mencoba membuat direktori {directory_path}.")
            self.logger.info(f"Mencoba metode alternatif...")
            
            # Opsi alternatif: buat direktori di lokasi berbeda
            alt_path = Path(os.path.expanduser("~")) / "Documents" / f"temp_output_{int(time.time())}"
            alt_path.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Dibuat direktori alternatif: {alt_path}")
            
            # Update path
            self.output_path = alt_path
            return False
        except Exception as e:
            self.logger.error(f"Error membuat direktori {directory_path}: {e}")
            raise

    def load_data(self, num_folds, data):
        csv_path = self.csv_path / f"training_{num_folds}_{data}" / "train" / "results_train.csv"
        if not csv_path.exists():
            self.logger.error(f"File CSV tidak ditemukan: {csv_path}")
            return None
        try:
            df = pd.read_csv(csv_path)
            self.logger.info(f"Data berhasil dimuat dari {csv_path}")
            
            # Tambahkan kolom 'epoch' jika tidak ada
            if "epoch" not in df.columns:
                self.logger.info(f"Menambahkan kolom 'epoch' di {csv_path}")
                df.insert(0, "epoch", range(1, len(df) + 1))  # Menambahkan epoch dengan urutan 1,2,3,...
                # Simpan ulang CSV dengan kolom epoch
                df.to_csv(csv_path, index=False)
                self.logger.info(f"CSV diperbarui dengan kolom 'epoch' di {csv_path}")
            
            return df
        except Exception as e:
            self.logger.error(f"Gagal memuat data dari {csv_path}: {e}")
            return None
    
    def save_fold_metrics(self):
        """Menyimpan metrik dari setiap fold ke dalam folder fold_{fold}."""
        for fold in range(1, self.num_folds + 1):
            df = self.load_data(num_folds=fold, data=self.data)
            if df is None:
                continue

            # Pastikan semua kolom ada di dataframe
            for col in self.columns:
                if col not in df.columns:
                    self.logger.info(f"Kolom {col} tidak ditemukan di fold {fold}, menambahkan nilai default.")
                    df[col] = 0  # Default 0 jika tidak ada nilai
            
            # Pastikan kolom epoch ada
            if "epoch" not in df.columns:
                df.insert(0, 'epoch', range(1, len(df) + 1))  

            fold_output_path = self.output_path / f"fold_{fold}"
            try:
                self.ensure_directory_exists(fold_output_path)

                for col in self.columns:
                    try:
                        output_file = fold_output_path / f"fold_{fold}_{col.replace('/', '_')}.csv"
                        df[["epoch", col]].to_csv(output_file, index=False)
                        self.logger.info(f"File CSV Fold-{fold} disimpan di path: {output_file}")
                    except Exception as e:
                        self.logger.error(f"Gagal menyimpan file {col} untuk fold {fold}: {e}")
            except Exception as e:
                self.logger.error(f"Gagal membuat direktori output untuk fold {fold}: {e}")
                continue

    def save_combined_metrics(self):
        """Menggabungkan metrik dari semua fold dan menyimpannya di folder combined/."""
        combined_dfs = {col: pd.DataFrame() for col in self.columns}
        combined_output_path = self.output_path / "combined"

        try:
            self.ensure_directory_exists(combined_output_path)

            for fold in range(1, self.num_folds + 1):
                for col in self.columns:
                    fold_file = self.output_path / f"fold_{fold}" / f"fold_{fold}_{col.replace('/', '_')}.csv"
                    if fold_file.exists():
                        self.logger.info(f"Sukses menyimpan file Fold-{fold} yang disimpan di path: {fold_file}") 
                        try:
                            df = pd.read_csv(fold_file)
                            if "epoch" in df.columns:
                                if combined_dfs[col].empty:
                                    combined_dfs[col] = df.copy()
                                    combined_dfs[col].rename(columns={col: f"Fold_{fold}"}, inplace=True)
                                else:
                                    combined_dfs[col][f"Fold_{fold}"] = df[col].copy()
                        except Exception as e:
                            self.logger.error(f"Gagal membaca file {fold_file}: {e}")

                    else:
                        self.logger.error(f"File tidak ditemukan: {fold_file}")

            # Simpan file gabungan
            for col, combined_df in combined_dfs.items():
                if not combined_df.empty:
                    try:
                        combined_file = combined_output_path / f"combined_{col.replace('/', '_')}.csv"
                        combined_df.to_csv(combined_file, index=False)
                        self.logger.info(f"File gabungan disimpan: {combined_file}")

                        # Simpan file mean/epoch
                        fold_cols = [c for c in combined_df.columns if c.startswith("Fold_")]
                        if fold_cols:
                            mean_df = pd.DataFrame()
                            mean_df["epoch"] = combined_df["epoch"]
                            mean_df["mean"] = combined_df[fold_cols].mean(axis=1)
                            mean_file = combined_output_path / f"mean_{col.replace('/', '_')}.csv"
                            mean_df.to_csv(mean_file, index=False)
                            self.logger.info(f"File rata-rata per epoch disimpan: {mean_file}")
                    except Exception as e:
                        self.logger.error(f"Gagal menyimpan file gabungan {col}: {e}")
        except Exception as e:
            self.logger.error(f"Gagal membuat direktori output gabungan: {e}")

class CSVPlotter:
    def __init__(self, olah_path, log_path, num_folds, data, model_type, size):
        self.num_folds = num_folds
        self.data = data
        self.model_type = model_type
        self.size = size
        self.file_path = olah_path / model_type / f'ukuran_{size}' / "hasil_csv"
        self.output_path = olah_path / model_type / f'ukuran_{size}' / "grafik"
        self.log_path = log_path / "log_plotting.txt"
        self.logger = LoggerManager(log_file=self.log_path)
        self.results_comp = self.output_path / "comparison"
        self.results_indt = self.output_path / f"independent/fold_{num_folds}"

        for path in [self.file_path, self.output_path, self.results_comp, self.results_indt]:
            path.mkdir(parents=True, exist_ok=True)

        self.variables = [
            "train/box_loss", "train/cls_loss", "train/dfl_loss", 
            "val/box_loss", "val/cls_loss", "val/dfl_loss", 
            "metrics/precision(B)", "metrics/recall(B)", 
            "metrics/mAP50(B)", "metrics/mAP50-95(B)"
        ]

    def _load_csv(self, filepath):
        if not filepath.exists():
            self.logger.error(f"File tidak ditemukan: {filepath}")
            return None
        try:
            return pd.read_csv(filepath)
        except Exception as e:
            self.logger.error(f"Gagal membaca file {filepath.name}: {e}")
            return None

    def _plot_curve(self, x, ys_dict, title, ylabel, save_path, use_style_variation=False):
        plt.figure(figsize=(8, 6))
        markers = ['o', 's', 'D', '^', 'v', '*', 'x', '+', 'p', 'h']
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

        for i, (label, y) in enumerate(ys_dict.items()):
            marker = markers[i % len(markers)] if use_style_variation else 'o'
            color = colors[i % len(colors)] if use_style_variation else 'b'
            plt.plot(x, y, marker=marker, linestyle='-', color=color, label=label)

        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.savefig(save_path, dpi=300)
        plt.close()
        self.logger.info(f"Plot disimpan: {save_path}")

    def plot_variable(self, target_folds, data):
        """
        Membuat plot untuk fold tertentu (atau semua jika target_folds=None).
        :param target_folds: int atau list of int (misalnya 3 atau [2, 3])
        """
        if target_folds is None:
            folds_to_process = range(1, self.num_folds + 1)
        elif isinstance(target_folds, int):
            folds_to_process = [target_folds]
        elif isinstance(target_folds, (list, tuple)):
            folds_to_process = target_folds
        else:
            self.logger.error(f"Parameter target_folds tidak valid: {target_folds}")
            return

        for fold in folds_to_process:
            for var in self.variables:
                filename = var.replace("/", "_")
                filepath = self.file_path / f"fold_{fold}" / f"fold_{fold}_{filename}.csv"
                df = self._load_csv(filepath)
                if df is None or "epoch" not in df or var not in df:
                    continue
                self._plot_curve(
                    df["epoch"], {var: df[var]},
                    title=f"{var} - Fold {fold}-{data}",
                    ylabel=var,
                    save_path=self.results_indt / f"{fold}_fold_Ind_{filename}.png"
                )

    def plot_combined(self, data):
        for var in self.variables:
            filename = var.replace("/", "_")
            filepath = self.file_path / "combined" / f"combined_{filename}.csv"
            df = self._load_csv(filepath)
            if df is None or "epoch" not in df:
                continue
            ys_dict = {col: df[col] for col in df.columns if col != "epoch"}
            self._plot_curve(
                df["epoch"], ys_dict,
                f"Combined Fold - {data.upper()} - {var}", var,
                self.results_comp / f"Cbd_{filename}.png",
                use_style_variation=True
            )

    def plot_mean_curves(self, data):
        for var in self.variables:
            filename = var.replace("/", "_")
            filepath = self.file_path / "combined" / f"mean_{filename}.csv"
            df = self._load_csv(filepath)
            if df is None or "epoch" not in df or "mean" not in df:
                continue
            self._plot_curve(
                df["epoch"], {f"Mean-{var}": df["mean"]},
                f"Mean Folds per Epoch - {data.upper()} - {var}", var,
                self.results_comp / f"Mean_{filename}.png"
            )

    def plot_all(self, target_folds, data):
        self.plot_variable(target_folds, data)
        self.plot_combined(data)
        self.plot_mean_curves(data)
        self.logger.info("Semua grafik berhasil dibuat dan disimpan.")

class ValidationDetector:
    def __init__(self, predictions_json, label_dir, olah_path, log_path, fold, data, model_type, size):
        self.predictions_json = predictions_json / "train" / "predictions.json"
        self.label_dir = label_dir
        self.fold = fold
        self.data = data
        self.model_type = model_type
        self.size = size
        self.image_width, self.image_height = 640, 640
        self.log_path = log_path / "log_val_det.txt"
        self.logger = LoggerManager(log_file=self.log_path)

        self.output_path = olah_path / model_type / f"ukuran_{size}" / "hasil_csv"
        self.output_json = olah_path / model_type / f"ukuran_{size}" / "json_final"
        self.output_gt = olah_path / model_type / f"ukuran_{size}" / "json_gt_deteksi"
        self.output_cm_d = olah_path / model_type / f"ukuran_{size}" / "cm_deteksi"
        self.output_json.mkdir(parents=True, exist_ok=True)
        self.output_gt.mkdir(parents=True, exist_ok=True)
        self.output_cm_d.mkdir(parents=True, exist_ok=True)

        self.valid_labels = {
            0: "bercak cokelat",
            1: "bercak cokelat tipis",
            2: "blas daun",
            3: "lepuh daun",
            4: "hawar daun bakteri",
            5: "sehat"
        }
        self.label_mapping = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5}

    def compute_iou(self, box1, box2):
        x1_min, y1_min = box1[0], box1[1]
        x1_max = x1_min + box1[2]
        y1_max = y1_min + box1[3]
        x2_min, y2_min = box2[0], box2[1]
        x2_max = x2_min + box2[2]
        y2_max = y2_min + box2[3]

        inter_xmin = max(x1_min, x2_min)
        inter_ymin = max(y1_min, y2_min)
        inter_xmax = min(x1_max, x2_max)
        inter_ymax = min(y1_max, y2_max)

        inter_area = max(0, inter_xmax - inter_xmin) * max(0, inter_ymax - inter_ymin)
        area1 = box1[2] * box1[3]
        area2 = box2[2] * box2[3]
        union_area = area1 + area2 - inter_area

        return inter_area / union_area if union_area > 0 else 0.0

    def _nms(self, pred_boxes, scores, classes, iou_threshold):
        """Non-Maximum Suppression (NMS) untuk satu gambar"""
        if not pred_boxes:
            return [], [], []

        boxes = np.array(pred_boxes)
        scores = np.array(scores)
        classes = np.array(classes)

        indices = scores.argsort()[::-1]
        keep_boxes, keep_scores, keep_classes = [], [], []

        while len(indices) > 0:
            current = indices[0]
            keep_boxes.append(pred_boxes[current])
            keep_scores.append(scores[current])
            keep_classes.append(classes[current])

            rest_indices = indices[1:]
            rest_boxes = [pred_boxes[i] for i in rest_indices]
            ious = [self.compute_iou(pred_boxes[current], box) for box in rest_boxes ]
            indices = [i for i, iou in zip(rest_indices, ious) if iou < iou_threshold]

        return keep_boxes, keep_scores, keep_classes
    
    def _load_gt(self):
        gt_json_path = self.output_gt / f"fold_{self.fold}_labels.json"
        if gt_json_path.exists():
            with open(gt_json_path, "r") as f:
                return json.load(f)

        ground_truth = {}
        if not os.path.exists(self.label_dir):
            raise FileNotFoundError(f"Label directory not found: {self.label_dir}")

        for label_file in os.listdir(self.label_dir):
            image_id = os.path.splitext(label_file)[0].strip()
            with open(os.path.join(self.label_dir, label_file), "r") as f:
                classes, bboxes = [], []
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    cls = int(parts[0])
                    xc, yc, w, h = map(float, parts[1:5])
                    x_min = (xc - w / 2) * self.image_width
                    y_min = (yc - h / 2) * self.image_height
                    w_px = w * self.image_width
                    h_px = h * self.image_height
                    coco_bbox = [x_min, y_min, w_px, h_px]
                    classes.append(cls)
                    bboxes.append(coco_bbox)

            ground_truth[image_id] = {"true_classes": classes, "true_bboxes": bboxes}

        with open(gt_json_path, "w") as f:
            json.dump(ground_truth, f, indent=4)

        return ground_truth

    def generate_matched_json(self, iou_threshold :float, score_threshold : float):
        # 1. Load raw predictions
        with open(self.predictions_json, "r") as f:
            raw_preds = json.load(f)

        # 2. Map ke format internal
        predictions = [
            {
                "image_id": str(item["image_id"]).strip(),
                "pred_label": self.label_mapping.get(item["category_id"], item["category_id"]),
                "bbox": item["bbox"],
                "score": item.get("score", 1.0)
            }
            for item in raw_preds
        ]

        # 3. Load ground truth
        ground_truth = self._load_gt()

        # 4. Kelompokkan prediksi per image & apply confidence threshold
        pred_by_image = defaultdict(list)
        for p in predictions:
            if p["score"] >= score_threshold:
                pred_by_image[p["image_id"]].append(p)

        # 5. Proses matching + NMS + statistik
        matched_data = {}
        total_gt = total_pred = total_matched_gt = unmatched_gts = unmatched_preds = 0

        # Hitung GT per image, lakukan NMS, hitung TP
        for image_id, gt in ground_truth.items():
            true_classes = gt["true_classes"]
            true_bboxes = gt["true_bboxes"]
            total_gt += len(true_bboxes)

            preds = pred_by_image.get(image_id, [])
            if not preds:
                unmatched_gts += 1
                self.logger.info(f"[UNMATCHED GT] {image_id}: {len(true_bboxes)} GT bbox, no predictions.")
                matched_data[image_id] = {
                    "true_classes": true_classes,
                    "true_bboxes": true_bboxes,
                    "pred_classes": [],
                    "pred_bboxes": [],
                    "scores": [],
                }
                continue

            # Extract lists
            boxes  = [p["bbox"] for p in preds]
            scores = [p["score"] for p in preds]
            labs   = [p["pred_label"] for p in preds]

            # NMS
            nms_boxes, nms_scores_np, nms_labels_np = self._nms(boxes, scores, labs, iou_threshold)
            # Cast scores & labels ke Python float/int
            nms_scores = [float(s) for s in nms_scores_np]
            nms_labels = [int(l)   for l in nms_labels_np]
            total_pred += len(nms_boxes)

            # Hitung GT yang terdeteksi (TP)
            matched_idxs = set()
            for gt_idx, gt_box in enumerate(true_bboxes):
                for pred_box in nms_boxes:
                    if self.compute_iou(gt_box, pred_box) >= iou_threshold:
                        matched_idxs.add(gt_idx)
                        break
            total_matched_gt += len(matched_idxs)

            matched_data[image_id] = {
                "true_classes": true_classes,
                "true_bboxes": true_bboxes,
                "pred_classes": nms_labels,
                "pred_bboxes": nms_boxes,
                "scores": nms_scores,
            }

        # Prediksi tanpa GT
        for image_id in pred_by_image:
            if image_id not in ground_truth:
                unmatched_preds += 1
                self.logger.error(f"[UNMATCHED PREDICTION] image_id '{image_id}' not in GT.")

        # Logging statistik
        total_missed = total_gt - total_matched_gt
        self.logger.info("\n[STATISTIK MATCHED JSON - VALIDASI]")
        self.logger.info(f"Total GT BBoxes       : {total_gt}")
        self.logger.info(f"Total Prediksi BBoxes : {total_pred}")
        self.logger.info(f"GT yang Terdeteksi    : {total_matched_gt}")
        self.logger.info(f"GT Tidak Terdeteksi   : {total_missed}")
        self.logger.info(f"Total Image matched   : {len(matched_data)}")
        self.logger.info(f"Image tanpa prediksi  : {unmatched_gts}")
        self.logger.info(f"Prediksi di luar GT   : {unmatched_preds}")

        # 6. Simpan matched_data ke JSON (tanpa statistik)
        out_json = self.output_json / f"fold_{self.fold}_matched.json"
        with open(out_json, "w") as f:
            json.dump(matched_data, f, indent=4)

        self.logger.info(f"\nFile Matched Json tersimpan di path: {out_json}")
        return out_json
    
    def generate_confusion_matrix(self, iou_threshold, score_threshold):
        matched_path = self.output_json / f"fold_{self.fold}_matched.json"
        with open(matched_path, "r") as f:
            matched_data = json.load(f)

        y_true, y_pred = [], []
        matched_pairs = set()

        for image_id, data in matched_data.items():
            gt_classes = data.get("true_classes", [])
            gt_bboxes = data.get("true_bboxes", [])
            pred_classes = data.get("pred_classes", [])
            pred_bboxes = data.get("pred_bboxes", [])
            scores = data.get("scores", [])

            used_pred = set()

            for i, (gt_cls, gt_box) in enumerate(zip(gt_classes, gt_bboxes)):
                best_iou = 0
                best_j = -1
                for j, (pred_cls, pred_box, score) in enumerate(zip(pred_classes, pred_bboxes, scores)):
                    if j in used_pred or score < score_threshold:
                        continue
                    iou = self.compute_iou(gt_box, pred_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_j = j
                if best_iou >= iou_threshold and best_j != -1:
                    y_true.append(gt_cls)
                    y_pred.append(pred_classes[best_j])
                    used_pred.add(best_j)
                    matched_pairs.add((image_id, best_j))
                else:
                    y_true.append(gt_cls)
                    y_pred.append("no pred")

            for j, (pred_cls, score) in enumerate(zip(pred_classes, scores)):
                if j not in used_pred and score >= score_threshold:
                    y_true.append("no gt")
                    y_pred.append(pred_cls)

        # Confusion matrix
        label_ids = list(self.valid_labels.keys())
        display_names = [self.valid_labels[i] for i in label_ids] + ["no pred", "no gt"]
        cm_labels = label_ids + ["no pred", "no gt"]

        cm = confusion_matrix(y_true, y_pred, labels=cm_labels)

        # Plot
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm.T, annot=True, fmt='d', cmap='Greens',
                    xticklabels=display_names, yticklabels=display_names,
                    cbar=True, square=True, linewidths=.5)
        plt.title(f"Confusion Matrix Multi-Kelas Data Validasi - Fold-{self.fold}-{self.data.upper()}", fontsize=13)
        plt.xlabel("True Labels", fontsize=12)
        plt.ylabel("Predicted Labels", fontsize=12)
        plt.xticks(rotation=90, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout(pad=2.5)

        output_cm = self.output_cm_d / f"fold_{self.fold}_cm.png"
        plt.savefig(output_cm, format="png", dpi=300)
        self.logger.info(f"[INFO] Confusion Matrix saved to: {output_cm}")

class PredictionDetector:
    def __init__(self, label_hasil, label_awal, olah_path, log_path, size, model_type, data, fold, engine):
        self.label_hasil = label_hasil
        self.label_awal = label_awal / "labels"
        self.size = size
        self.model_type = model_type
        self.engine = engine
        self.data = data
        self.fold = fold
        self.log_path = log_path / "log_uji_det.txt"
        self.logger = LoggerManager(log_file=self.log_path)
        self.output_json_hasil = olah_path / model_type / f"ukuran_{size}" / "json_hasil_uji"
        self.output_json_awal = olah_path / model_type / f"ukuran_{size}" /"json_gt_deteksi"
        self.json_matched = olah_path / model_type / f"ukuran_{size}" / "json_matched_uji"
        self.cm_uji = olah_path / model_type / f"ukuran_{size}" / "cm_deteksi_uji"
        self.output_json_hasil.mkdir(parents=True, exist_ok=True)
        self.json_matched.mkdir(parents=True, exist_ok=True)
        self.cm_uji.mkdir(parents=True, exist_ok=True)
        self.image_width, self.image_height = 640, 640
        self.valid_labels = {
            0: "bercak cokelat",
            1: "bercak cokelat tipis",
            2: "blas daun",
            3: "lepuh daun",
            4: "hawar daun bakteri",
            5: "sehat"
        }
        self.label_mapping = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5}

    def compute_iou(self, box1, box2):
        x1_min, y1_min = box1[0], box1[1]
        x1_max = x1_min + box1[2]
        y1_max = y1_min + box1[3]
        x2_min, y2_min = box2[0], box2[1]
        x2_max = x2_min + box2[2]
        y2_max = y2_min + box2[3]

        inter_xmin = max(x1_min, x2_min)
        inter_ymin = max(y1_min, y2_min)
        inter_xmax = min(x1_max, x2_max)
        inter_ymax = min(y1_max, y2_max)

        inter_area = max(0, inter_xmax - inter_xmin) * max(0, inter_ymax - inter_ymin)
        area1 = box1[2] * box1[3]
        area2 = box2[2] * box2[3]
        union_area = area1 + area2 - inter_area

        return inter_area / union_area if union_area > 0 else 0.0
    
    def compute_ap(self, recall, precision):
        """Menghitung area under PR curve (COCO style)."""
        return auc(recall, precision)
    
    def _nms(self, pred_boxes, scores, classes, iou_threshold):
        """Non-Maximum Suppression (NMS) untuk satu gambar"""
        if not pred_boxes:
            return [], [], []

        boxes = np.array(pred_boxes)
        scores = np.array(scores)
        classes = np.array(classes)

        indices = scores.argsort()[::-1]
        keep_boxes, keep_scores, keep_classes = [], [], []

        while len(indices) > 0:
            current = indices[0]
            keep_boxes.append(pred_boxes[current])
            keep_scores.append(scores[current])
            keep_classes.append(classes[current])

            rest_indices = indices[1:]
            rest_boxes = [pred_boxes[i] for i in rest_indices]
            ious = [self.compute_iou(pred_boxes[current], box) for box in rest_boxes ]
            indices = [i for i, iou in zip(rest_indices, ious) if iou < iou_threshold]

        return keep_boxes, keep_scores, keep_classes

    def _load_gt(self):
        gt_json_path = self.output_json_awal / f"uji_awal_labels.json"
        if gt_json_path.exists():
            with open(gt_json_path, "r") as f:
                return json.load(f)

        ground_truth = {}
        if not os.path.exists(self.label_awal):
            raise FileNotFoundError(f"Label directory not found: {self.label_awal}")

        for label_file in os.listdir(self.label_awal):
            image_id = os.path.splitext(label_file)[0].strip()
            with open(os.path.join(self.label_awal, label_file), "r") as f:
                classes, bboxes = [], []
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    cls = int(parts[0])
                    xc, yc, w, h = map(float, parts[1:5])
                    x_min = (xc - w / 2) * self.image_width
                    y_min = (yc - h / 2) * self.image_height
                    w_px = w * self.image_width
                    h_px = h * self.image_height
                    coco_bbox = [x_min, y_min, w_px, h_px]
                    classes.append(cls)
                    bboxes.append(coco_bbox)

            ground_truth[image_id] = {"true_classes": classes, "true_bboxes": bboxes}

        with open(gt_json_path, "w") as f:
            json.dump(ground_truth, f, indent=4)

        return ground_truth
    
    def _load_pred(self):
        pred_json_path = self.output_json_hasil / f"{self.engine}_fold_{self.fold}_hasil_uji_labels.json"
        if pred_json_path.exists():
            with open(pred_json_path, "r") as f:
                return json.load(f)
        
        self.logger.info(f"\nPath Label Data Uji yang dijalankan yakni : {self.label_hasil} \n")

        if not os.path.exists(self.label_hasil):
            raise FileNotFoundError(f"Label directory not found: {self.label_hasil}")

        predictions = {}
        for label_file in os.listdir(self.label_hasil):
            image_id = os.path.splitext(label_file)[0].strip()
            classes, bboxes, scores = [], [], []
            with open(os.path.join(self.label_hasil, label_file), "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 6:
                        continue
                    cls = int(parts[0])
                    xc, yc, w, h, score = map(float, parts[1:])

                    # Konversi ke format [x_min, y_min, width, height] (COCO style)
                    x_min = (xc - w / 2) * self.image_width
                    y_min = (yc - h / 2) * self.image_height
                    w_px = w * self.image_width
                    h_px = h * self.image_height
                    coco_bbox = [x_min, y_min, w_px, h_px]
                    classes.append(cls)
                    bboxes.append(coco_bbox)
                    scores.append(score)

            predictions[image_id] = {
                "pred_classes": classes, 
                "pred_bboxes": bboxes, 
                "scores": scores}

        with open(pred_json_path, "w") as f:
            json.dump(predictions, f, indent=4)

        self.logger.info(f"\nfile JSON Prediksi dari Data Uji path: {pred_json_path} berhasil didapatkan !")

        return predictions
    
    def _load_matched(self, output_path, score_threshold, iou_threshold):
        gt_data = self._load_gt()
        pred_data = self._load_pred()

        matched = {}
        total_gt = total_pred = total_matched_gt = 0

        for image_id in sorted(set(gt_data) | set(pred_data)):
            gt_info    = gt_data.get(image_id, {})
            true_cls   = gt_info.get("true_classes", [])
            true_boxes = gt_info.get("true_bboxes", [])
            total_gt  += len(true_boxes)

            pred_info  = pred_data.get(image_id, {})
            raw_boxes  = pred_info.get("pred_bboxes", [])
            raw_scores = pred_info.get("scores", [])
            raw_cls    = pred_info.get("pred_classes", [])

            # 1) Confidence filtering
            filtered = [
                (b, s, c) for b, s, c in zip(raw_boxes, raw_scores, raw_cls)
                if s >= score_threshold
            ]

            # 2) NMS
            if filtered:
                boxes, scores, cls = zip(*filtered)
                nms_boxes, nms_scores_np, nms_cls_np = self._nms(boxes, scores, cls, iou_threshold)
                # Cast to Python types
                nms_scores = [float(s) for s in nms_scores_np]
                nms_cls    = [int(c)   for c in nms_cls_np]
            else:
                nms_boxes, nms_scores, nms_cls = [], [], []

            total_pred += len(nms_boxes)

            # 3) Hitung GT yang terdeteksi
            matched_idxs = set()
            for idx, gt_box in enumerate(true_boxes):
                for pbox in nms_boxes:
                    if self.compute_iou(gt_box, pbox) >= iou_threshold:
                        matched_idxs.add(idx)
                        break
            total_matched_gt += len(matched_idxs)

            matched[image_id] = {
                "true_classes": true_cls,
                "true_bboxes":  true_boxes,
                "pred_classes": nms_cls,
                "pred_bboxes":  nms_boxes,
                "scores":       nms_scores,
            }

        # 4) Tulis ke JSON
        with open(output_path, "w") as f:
            json.dump(matched, f, indent=4)

        # 5) Logging statistik
        total_missed = total_gt - total_matched_gt
        self.logger.info("\n[STATISTIK MATCHED JSON]")
        self.logger.info(f"Total GT BBoxes       : {total_gt}")
        self.logger.info(f"Total Prediksi BBoxes : {total_pred}")
        self.logger.info(f"GT yang Terdeteksi    : {total_matched_gt}")
        self.logger.info(f"GT Tidak Terdeteksi   : {total_missed}")
        self.logger.info(f"File JSON gabungan tersimpan di path: {output_path}.\n")

        return output_path

    def generate_json(self, score_threshold, iou_threshold, output_path):
        self.logger.info(f"\nGenerating matched JSON dengan threshold confidence score = {score_threshold} dan IoU = {iou_threshold}...")

        # Muat GT dan Prediksi
        self._load_gt()
        self._load_pred()

        # Lakukan pencocokan dan simpan hasil
        matched = self._load_matched(
            score_threshold=score_threshold,
            iou_threshold=iou_threshold,
            output_path=output_path)
        self.logger.info("\nFile JSON untuk Label Data Awal dan Data Hasil sudah dibuat !!")

        return matched

    def generate_cm(self, matched_json_path, iou_threshold, score_threshold):
        with open(matched_json_path, "r") as f:
            matched_data = json.load(f)

        matched_pairs = set()
        y_true, y_pred = [], []

        # Gunakan prediksi terbaik per gambar jika diaktifkan
        for image_id, data in matched_data.items():
            gt_classes = data.get("true_classes", [])
            gt_bboxes = data.get("true_bboxes", [])
            pred_classes = data.get("pred_classes", [])
            pred_bboxes = data.get("pred_bboxes", [])
            scores = data.get("scores", [])

            used_pred = set()

            for i, (gt_cls, gt_box) in enumerate(zip(gt_classes, gt_bboxes)):
                best_iou = 0
                best_j = -1
                for j, (pred_cls, pred_box, score) in enumerate(zip(pred_classes, pred_bboxes, scores)):
                    if j in used_pred or score < score_threshold:
                        continue
                    iou = self.compute_iou(gt_box, pred_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_j = j
                if best_iou >= iou_threshold and best_j != -1:
                    y_true.append(gt_cls)
                    y_pred.append(pred_classes[best_j])
                    used_pred.add(best_j)
                    matched_pairs.add((image_id, best_j))
                else:
                    y_true.append(gt_cls)
                    y_pred.append("no pred")

            # Tambahkan false positive dari prediksi tidak match (score valid)
            for j, (pred_cls, score) in enumerate(zip(pred_classes, scores)):
                if j not in used_pred and score >= score_threshold:
                    y_true.append("no gt")
                    y_pred.append(pred_cls)

        # Susun label untuk confusion matrix
        label_ids = list(self.valid_labels.keys())
        display_names = [self.valid_labels[i] for i in label_ids] + ["no pred", "no gt"]
        cm_labels = label_ids + ["no pred", "no gt"]

        cm = confusion_matrix(y_true, y_pred, labels=cm_labels)

        # Visualisasi confusion matrix
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm.T, annot=True, fmt='d', cmap='Reds',
                    xticklabels=display_names, yticklabels=display_names,
                    cbar=True, square=True, linewidths=.5)

        plt.title(f"Confusion Matrix Multi-Kelas - Fold-{self.fold} Data Uji {self.data.upper()}-{self.engine.upper()}", fontsize=13)
        plt.xlabel("True Labels", fontsize=12)
        plt.ylabel("Predicted Labels", fontsize=12)
        plt.xticks(rotation=90, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout(pad=2.5)

        output_cm = self.cm_uji / f"fold_{self.fold}_cm_uji_{self.engine}.png"
        plt.savefig(output_cm, format="png", dpi=300)
        self.logger.info(f"\nConfusion Matrix tersimpan di path: {output_cm}")
    
    def log_evaluation_metrics(self, path_matched: str, conf_threshold: float, iou_thresholds: list):
        if isinstance(iou_thresholds, float):
            iou_thresholds = [iou_thresholds]

        with open(path_matched) as f:
            data = json.load(f)

        pred_by_class_iou = defaultdict(lambda: defaultdict(list))
        n_gt_per_class = defaultdict(lambda: defaultdict(int))
        all_classes = set()

        for _, item in data.items():
            gt_classes = item["true_classes"]
            gt_bboxes = item["true_bboxes"]
            pred_classes = item["pred_classes"]
            pred_bboxes = item["pred_bboxes"]
            scores = item["scores"]

            for tc in gt_classes:
                all_classes.add(tc) # untuk memastikan semua GT masuk ke set class

            for iou_thresh in iou_thresholds:
                used_gt = set()
                for i, (pc, pb, score) in enumerate(zip(pred_classes, pred_bboxes, scores)):
                    if score < conf_threshold:
                        continue

                    matched = False
                    for j, (tc, tb) in enumerate(zip(gt_classes, gt_bboxes)):
                        iou = self.compute_iou(tb, pb)
                        if iou >= iou_thresh and j not in used_gt and tc == pc:
                            pred_by_class_iou[pc][iou_thresh].append((score, 1))  # TP
                            used_gt.add(j)
                            matched = True
                            break

                    if not matched:
                        pred_by_class_iou[pc][iou_thresh].append((score, 0))  # FP

                # Count FN per class
                for j, tc in enumerate(gt_classes):
                    if j not in used_gt:
                        n_gt_per_class[tc][iou_thresh] += 1

        # Precision, Recall, mAP50
        if 0.50 in iou_thresholds:
            total_precision, total_recall, ap_values = [], [], []
            for cls in all_classes:
                preds = pred_by_class_iou[cls][0.50]
                if not preds:
                    continue

                preds.sort(key=lambda x: -x[0])
                tp_fp = np.array([p[1] for p in preds])
                tp_cum = np.cumsum(tp_fp)
                fp_cum = np.cumsum(1 - tp_fp)

                n_gt = n_gt_per_class[cls][0.50]
                if n_gt == 0:
                    continue

                recall = tp_cum / (n_gt + 1e-6)
                precision = tp_cum / (tp_cum + fp_cum + 1e-6)

                recall = np.clip(recall, 0, 1)
                precision = np.clip(precision, 0, 1)

                final_precision = precision[-1] if len(precision) else 0.0
                final_recall = recall[-1] if len(recall) else 0.0
                ap = self.compute_ap(recall, precision)

                total_precision.append(final_precision)
                total_recall.append(final_recall)
                ap_values.append(ap)

            mean_precision = np.mean(total_precision) if total_precision else 0.0
            mean_recall = np.mean(total_recall) if total_recall else 0.0
            map50 = np.mean(ap_values) if ap_values else 0.0

            self.logger.info(f"\nMetriks Evaluasi Deteksi Model {self.engine}-{self.model_type} YOLOv8{self.size} Jenis Data {self.data.upper()} pada Fold-{self.fold} : ")
            self.logger.info(f"[IoU 0.50] Precision: {mean_precision:.2f}, Recall: {mean_recall:.2f}, mAP50: {map50:.2f}")

        # mAP50-95 (mean AP over all IoU thresholds)
        all_aps = []
        for iou_thresh in iou_thresholds:
            ap_list = []
            for cls in all_classes:
                preds = pred_by_class_iou[cls][iou_thresh]
                if not preds:
                    continue

                preds.sort(key=lambda x: -x[0])
                tp_fp = np.array([p[1] for p in preds])
                tp_cum = np.cumsum(tp_fp)
                fp_cum = np.cumsum(1 - tp_fp)

                n_gt = n_gt_per_class[cls][iou_thresh]
                if n_gt == 0:
                    continue

                recall = tp_cum / (n_gt + 1e-6)
                precision = tp_cum / (tp_cum + fp_cum + 1e-6)
                recall = np.clip(recall, 0, 1)
                precision = np.clip(precision, 0, 1)

                ap = self.compute_ap(recall, precision)
                ap_list.append(ap)

            if ap_list:
                all_aps.append(np.mean(ap_list))

        if all_aps:
            map5095 = np.mean(all_aps)
            self.logger.info(f"[EVALUASI] mAP50-95: {map5095:.2f}")
            self.logger.info(f"\n\nMetrik Evaluasi Deteksi model {self.engine} untuk Fold-{self.fold} didapatkan ! \n")

class VisualisasiDatasetAwal:
    def __init__(self, class_mapping: Dict[int, str], hide_labels: bool, logger):
        self.class_mapping = class_mapping
        self.hide_labels = hide_labels
        self.logger = logger
        self.colors = {  # Normalized colors for better visibility
            0: (1, 0, 0),    # Red
            1: (1, 0.65, 0),  # Orange
            2: (0, 0, 1),    # Blue
            3: (1, 1, 0),    # Yellow
            4: (0.5, 0, 0.5),# Purple
            5: (0, 1, 1)     # Cyan
        }
    
    def _normalize_image(self, image: torch.Tensor) -> np.ndarray:
        """Convert image tensor to NumPy format and normalize to [0,1]."""
        if isinstance(image, torch.Tensor):
            image = image.permute(1, 2, 0).cpu().numpy()
        return image.astype(np.float32) / 255.0 if image.dtype == np.uint8 else np.clip(image, 0, 1)

    def plot_gallery(self, images: List[torch.Tensor], boxes: List[torch.Tensor], 
                    labels: List[torch.Tensor], output_path: str, rows: int = 2, cols: int = 2, title: Optional[str] = None):
        """Plot images with bounding boxes."""
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))
        if title:
            fig.suptitle(title, fontsize=10)
        axes = axes.flatten()

        for idx, ax in enumerate(axes):
            if idx < len(images):
                img, bboxes, lbls = images[idx], boxes[idx], labels[idx]
                img = self._normalize_image(img)
                img_height, img_width = img.shape[:2] # ambil ukuran gambar height dan width
                ax.imshow(img)

                # hanya menampilkan bboxes tanpa label di pojok kiri atas
                for box, label in zip(bboxes, lbls):
                    if len(box) != 4:
                        continue
                    
                    # Konversi dari [x_center, y_center, width, height] ke [x_min, y_min, x_max, y_max]
                    x_center, y_center, w, h = box
                    if w == 0 or h == 0:
                        continue

                    x_min = int((x_center - w / 2) * img_width)
                    y_min = int((y_center - h / 2) * img_height)
                    x_max = int((x_center + w / 2) * img_width)
                    y_max = int((y_center + h / 2) * img_height)

                    # Pastikan bounding box tidak keluar dari batas gambar
                    x_min, y_min = max(0, x_min), max(0, y_min)
                    x_max, y_max = min(img_width, x_max), min(img_height, y_max)

                    class_label = self.class_mapping.get(int(label), "unknown")
                    if class_label == "unknown":
                        continue # jangan tampilkan label yang tidak dikenali

                    color = self.colors.get(int(label), (1,1,1)) # default ke putih jika label tidak ditemukan
                    ax.add_patch(plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, 
                                            fill=False, edgecolor=color, linewidth=2))

                    # Tampilkan label hanya di dalam bounding box jika tidak disembunyikan
                    if not self.hide_labels:
                        label_x = x_min
                        label_y = max(y_min - 5,5) # hindari nilai negatif dan overlap dengan bounding box
                        ax.text(label_x, label_y, class_label, color='black', fontsize=8, weight="bold",
                            bbox=dict(facecolor=color, alpha=0.8, pad=0.5),
                            horizontalalignment='left', verticalalignment='top')
                ax.axis('off')
            else:
                ax.axis('off')  # Nonaktifkan subplot kosong
        plt.tight_layout()
        plt.subplots_adjust(left=0, right=1, top=0.95 if title else 1, bottom=0)
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()

    def visualize_samples(self, dataset, indices: List[int], num_samples: int, output_path: str, title: Optional[str] = None):
        """Extract and visualize samples from dataset."""
        selected_indices = np.random.choice(indices, min(num_samples, len(indices)), replace=False)
        images, boxes, labels = zip(*[dataset[i][:3] for i in selected_indices])
        self.plot_gallery(images, boxes, labels, output_path=output_path, title=title)

class VisualisasiDatasetPred:
    def __init__(self, image_dir, pred_dir, anotasi_dir, output_dir, data:str, model_type:str, size:str, fold):
        self.image_folder = image_dir # path gambar set uji
        self.pred_dir = pred_dir # path hasil gambar prediksi
        self.anotasi_dir = anotasi_dir # path anotasi gambar set uji
        self.output_dir = output_dir # path hasil gambar komparasi
        self.data = data # jenis data yang digunakan
        self.model_type = model_type # tipe model yang digunakan
        self.size = size # ukuran model yang digunakan
        self.fold = fold # fold dari k-fold cross validation
        os.makedirs(self.output_dir, exist_ok=True)  # Buat folder output jika belum ada
        self.image_files = self._get_matching_files()

        #Class Mapping sesuai kelasnya
        self.class_mapping = {
            0: "bercak cokelat",
            1: "bercak cokelat tipis",
            2: "blas daun",
            3: "lepuh daun",
            4: "hawar daun bakteri",
            5: "sehat"
        }

        # Color mapping anotasi
        self.color_mapping = {
            0: (0, 0, 255),       # biru
            1: (0, 255, 255),     # cyan
            2: (255, 255, 255),   # putih
            3: (0, 0, 128),       # navy
            4: (64, 224, 208),    # turquoise
            5: (255, 0, 255)      # magenta
        }
        
    def _get_matching_files(self)-> List[Tuple[str, str, str]]:
        """Mencari semua gambar mentahan dan mencocokkannya dengan hasil prediksi serta anotasi."""
        supported_formats = ['*.png', '*.jpg', '*.jpeg', '*.JPG','*.PNG','*.JPEG', '*.bmp', '*.tif', '*.tiff']
    
        # Gabungkan semua ekstensi yang didukung
        image_paths = []
        for fmt in supported_formats:
            image_paths.extend(glob(os.path.join(self.image_folder, fmt)))

        matching_files = []

        for img_path in image_paths:
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            pred_path = os.path.join(self.pred_dir, f"{base_name}.jpg")
            annot_path = os.path.join(self.anotasi_dir, f"{base_name}.txt")

            if os.path.exists(pred_path) and os.path.exists(annot_path):
                matching_files.append((img_path, pred_path, annot_path))

        return matching_files

    def _load_annotations(self, annotation_path, image_shape):
        """Membaca file anotasi dan mengonversinya ke bounding box dalam koordinat piksel."""
        h, w, _ = image_shape
        bboxes = []

        with open(annotation_path, "r") as file:
            lines = file.readlines()

        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            class_id, x_center, y_center, width, height = map(float, parts)
            
            # Konversi YOLO ke koordinat piksel
            x_min = int((x_center - width / 2) * w)
            y_min = int((y_center - height / 2) * h)
            x_max = int((x_center + width / 2) * w) 
            y_max = int((y_center + height / 2) * h)
            
            bboxes.append((int(class_id), x_min, y_min, x_max, y_max))

        return bboxes

    def _draw_bboxes(self, image, bboxes, color_mapping):
        """Menambahkan bounding boxes ke gambar."""
        img_copy = image.copy()
        for class_id, x_min, y_min, x_max, y_max in bboxes:
            color = color_mapping.get(class_id, (255,255,0)) # warna default kuning kalau tidak ada
            label = self.class_mapping.get(class_id, "unknown")
            cv2.rectangle(img_copy, (x_min, y_min), (x_max, y_max), color, 2)
            cv2.putText(img_copy, label, (x_min, max(y_min - 5, 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
            
        return img_copy
    
    def _resize_if_needed(self, img, target_shape):
        if img.shape[:2] != target_shape:
            return cv2.resize(img, (target_shape[1], target_shape[0]))
        return img

    def _normalize_image(self, image):
        if isinstance(image, torch.Tensor):
            image = image.permute(1, 2, 0).cpu().numpy()
        
        if image.dtype == np.uint8:
            return image.astype(np.float32) / 255.0
        else:
            return np.clip(image, 0, 1)
        
    def plot_gallery(self, images: List[torch.Tensor], boxes: List[torch.Tensor], 
                    labels: List[torch.Tensor], output_path: str, rows: int = 2, cols: int = 2, title: Optional[str] = None):
        """Plot images with bounding boxes."""
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))
        if title:
            fig.suptitle(title, fontsize=10)
        axes = axes.flatten()

        for idx, ax in enumerate(axes):
            if idx < len(images):
                img, bboxes, lbls = images[idx], boxes[idx], labels[idx]
                img = self._normalize_image(img)
                img_height, img_width = img.shape[:2] # ambil ukuran gambar height dan width
                ax.imshow(img)

                # hanya menampilkan bboxes tanpa label di pojok kiri atas
                for box, label in zip(bboxes, lbls):
                    if len(box) != 4:
                        continue
                    
                    # Konversi dari [x_center, y_center, width, height] ke [x_min, y_min, x_max, y_max]
                    x_center, y_center, w, h = box
                    if w == 0 or h == 0:
                        continue

                    x_min = int((x_center - w / 2) * img_width)
                    y_min = int((y_center - h / 2) * img_height)
                    x_max = int((x_center + w / 2) * img_width)
                    y_max = int((y_center + h / 2) * img_height)

                    # Pastikan bounding box tidak keluar dari batas gambar
                    x_min, y_min = max(0, x_min), max(0, y_min)
                    x_max, y_max = min(img_width, x_max), min(img_height, y_max)

                    class_label = self.class_mapping.get(int(label), "unknown")
                    if class_label == "unknown":
                        continue # jangan tampilkan label yang tidak dikenali

                    color = self.colors.get(int(label), (1,1,1)) # default ke putih jika label tidak ditemukan
                    ax.add_patch(plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, 
                                            fill=False, edgecolor=color, linewidth=2))

                    # Tampilkan label hanya di dalam bounding box jika tidak disembunyikan
                    if not self.hide_labels:
                        label_x = x_min
                        label_y = max(y_min - 5,5) # hindari nilai negatif dan overlap dengan bounding box
                        ax.text(label_x, label_y, class_label, color='black', fontsize=8, weight="bold",
                            bbox=dict(facecolor=color, alpha=0.8, pad=0.5),
                            horizontalalignment='left', verticalalignment='top')
                ax.axis('off')
            else:
                ax.axis('off')  # Nonaktifkan subplot kosong
        plt.tight_layout()
        plt.subplots_adjust(left=0, right=1, top=0.95 if title else 1, bottom=0)
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()

    def visualize_side_by_side(self):
        """ Visualisasi 2 gambar berdampingan: mentahan+anotasi vs prediksi """
        for img_path, pred_path, annot_path in self.image_files:
            img = cv2.imread(img_path)
            pred_img = cv2.imread(pred_path)

            if img is None or pred_img is None:
                print(f"Error loading {img_path} atau {pred_path}")
                continue

            pred_shape = pred_img.shape[:2]
            img = self._resize_if_needed(img, pred_shape)

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pred_rgb = cv2.cvtColor(pred_img, cv2.COLOR_BGR2RGB)

            # Load and draw bounding boxes
            bboxes = self._load_annotations(annot_path, img.shape)
            img_with_annots = self._draw_bboxes(img_rgb, bboxes, self.color_mapping)

            fig, axes = plt.subplots(1, 2, figsize=(16, 8))

            axes[0].imshow(img_with_annots)
            axes[0].set_title("Gambar Mentahan + Anotasi", fontsize=12)
            axes[0].axis("off")

            axes[1].imshow(pred_rgb)
            axes[1].set_title("Hasil Prediksi", fontsize=12)
            axes[1].axis("off")

            # Tulis nama file di bawah gambar
            fig.text(0.5, 0.02, f"{os.path.basename(img_path)}", ha='center', fontsize=12)

            base_name = os.path.splitext(os.path.basename(img_path))[0]
            output_path = os.path.join(self.output_dir, f"comp_{self.data}_{self.model_type}_{self.size}_{self.fold}_{base_name}.png")
            plt.savefig(output_path, bbox_inches="tight", dpi=300)
            plt.close(fig)
            print(f"\nGambar disimpan di path: \n{output_path}.\n")

    def visualize_overlay(self):
        """ Visualisasi overlay: anotasi dan prediksi digabung dalam 1 gambar """
        for img_path, pred_path, annot_path in self.image_files:
            img = cv2.imread(img_path)
            pred_img = cv2.imread(pred_path)

            if img is None or pred_img is None:
                print(f"Error loading {img_path} atau {pred_path}")
                continue

            pred_shape = pred_img.shape[:2]
            img = self._resize_if_needed(img, pred_shape)

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pred_rgb = cv2.cvtColor(pred_img, cv2.COLOR_BGR2RGB)

            bboxes = self._load_annotations(annot_path, img.shape)
            img_with_annots = self._draw_bboxes(img_rgb, bboxes, self.color_mapping)

            # Overlay hasil prediksi di atas gambar mentahan+anotasi
            overlay_img = cv2.addWeighted(img_with_annots, 0.7, pred_rgb, 0.3, 0)

            plt.figure(figsize=(8, 8))
            plt.imshow(overlay_img)
            plt.title(f"Overlay Anotasi vs Prediksi - {os.path.basename(img_path)}", fontsize=12)
            plt.axis("off")

            base_name = os.path.splitext(os.path.basename(img_path))[0]
            output_path = os.path.join(self.output_dir, f"overlay_{self.data}_{self.model_type}_{self.size}_{self.fold}_{base_name}.png")
            plt.savefig(output_path, bbox_inches="tight", dpi=300)
            plt.close()
            print(f"Saved: {output_path}")

MAIN_PATH = Path("D:/Riset Skripsi/script riset/deteksi_citra/")
DATASET_PATH = Path(f"D:/Riset Skripsi/dataset_skripsi/")
OLAH_CONFIG = {
    'n_folds':5,
    'conf_score':0.35,
    'iou_map50': 0.5,
    'iou_map5095' : [round(x,2) for x in np.arange(0.5, 1.0, 0.05)]
}

def running_det_uji(detector: PredictionDetector, olah_config: dict):
    logger = detector.logger
    engine_label = detector.engine.lower()
    matched_name = f"{engine_label}_fold_{detector.fold}_uji_matched.json"
    matched_path = detector.json_matched / matched_name

    # Generate matched JSON jika belum ada
    if not matched_path.exists():
        logger.info(f"[INFO] Membuat file matched.json untuk {engine_label.upper()} karena belum ada.")
        matched_path = detector.generate_json(
            score_threshold=olah_config["conf_score"],
            iou_threshold=olah_config["iou_map50"],
            output_path=matched_path
        )
    else:
        logger.info(f"[INFO] File matched.json {engine_label.upper()} ditemukan. Melewati pembuatan ulang.")

    # Generate confusion matrix jika belum dibuat
    if input(f"\nApakah confusion matrix untuk {engine_label.upper()} model sudah dibuat? (y/n): ").strip().lower() == 'n':
        detector.generate_cm(
            matched_json_path=matched_path,
            iou_threshold=olah_config['iou_map50'],
            score_threshold=olah_config['conf_score']
        )
    else:
        logger.info(f"[INFO] Confusion Matrix {engine_label.upper()} sudah tersedia. Lanjut evaluasi...\n")

    # Logging metrik evaluasi
    logger.info(f"\n[LOGGING] Metrik Evaluasi Deteksi ({engine_label.upper()})")
    detector.log_evaluation_metrics(
        path_matched=matched_path,
        conf_threshold=olah_config["conf_score"],
        iou_thresholds=olah_config["iou_map5095"]
    ) 

def main():
    # Running Program
    print("\n=== Memulai Pembuatan Grafik Variabel-Variabel dan Confusion Matrix ===\n")
    
    # Pilih jenis data
    data = input(f"Masukkan jenis data citra yang ingin diolah hasilnya (nonbg/bg/mix): ").strip()
    # Pastikan input valid
    if data not in {"nonbg", "bg", "mix"}:
        raise ValueError(f"Jenis data '{data}' tidak valid. Harus 'nonbg', 'bg', atau 'mix'.")
    
    # memilih salah satu jenis model (biasa/kustom)
    model_type = input("Masukkan hasil model yang ingin kamu gunakan (biasa/kustom): ").strip()
    if model_type not in {"biasa", "kustom"}:
        raise ValueError("Maaf, jenis model yang tersedia hanya 'biasa' atau 'kustom'.")

    # Tentukan fold yang ingin dijalankan
    fold_mode = input("Ingin menjalankan semua fold atau fold tertentu saja? (all/nomor): ").strip()
    if fold_mode == "all":
        fold_range = range(1, OLAH_CONFIG['n_folds'] + 1)
    elif fold_mode.isdigit() and 1 <= int(fold_mode) <= OLAH_CONFIG['n_folds']:
        fold_range = [int(fold_mode)]
    else:
        raise ValueError("Input fold tidak valid. Masukkan 'all' atau angka fold yang valid.")
    
    for fold in fold_range:
        torch.cuda.empty_cache()
        print(f"\n>>>>> Memproses fold {fold} <<<<<< \n")

        # Pilih path hasil ukuran model fitting berdasarkan input
        size = input("\nMasukkan size hasil model (n/s/m/l/xl untuk biasa, s/m/l untuk kustom): ").strip()
        valid_sizes = {"n", "s", "m", "l", "xl"} if model_type == "biasa" else {"s", "m", "l"}
        if size not in valid_sizes:
            raise ValueError(f"Size model '{size}' tidak valid untuk model {model_type}.")

        # Path setup
        proses_path = MAIN_PATH / data / "olah_data"
        result_path = MAIN_PATH / data / f"hasil_{data}" / model_type / f"ukuran_{size}"
        log_path = MAIN_PATH / data / "logging" / "olah_data"
        log_root_path = log_path / f"olah_data_{data}.txt"
        json_path = result_path / f"validating_{fold}_{data}"
        hasil_uji_path = result_path / f"testing_{fold}" / "predict"
        trt_uji_path = result_path / f"testing_trt{fold}" / "predict"
        labels_path = DATASET_PATH / f"dataset_{data}/fold/fold_{fold}/val/labels"
        labels_uji_path = DATASET_PATH / f"dataset_{data}/fold/test"

        log_path.mkdir(parents=True, exist_ok=True)
        proses_path.mkdir(parents=True, exist_ok=True)

        logger = LoggerManager(log_file=log_root_path)
        
        try:
            # === CSV Remake ===
            if input("Buat file CSV untuk utilisasi? (y/n): ").strip().lower() == 'y':
                logger.info("[CSV] Memulai pemindahan data CSV...")
                csv_remaker = CSVRemaker(result_path, proses_path, log_path, fold, data, model_type, size)
                csv_remaker.save_fold_metrics()
                csv_remaker.save_combined_metrics()
                logger.info("[CSV] Selesai menyimpan file CSV utilisasi.")

            # === Plotting Grafik ===
            if input("Lakukan plotting grafik? (y/n): ").strip().lower() == 'y':
                logger.info("[PLOT] Memulai proses plotting...")
                plotter = CSVPlotter(proses_path, log_path, fold, data, model_type, size)
                plotter.plot_all(target_folds=fold, data=data)
                logger.info("[PLOT] Grafik selesai dibuat.")

            # === Confusion Matrix Validasi ===
            if input("\nApakah kamu ingin buat confusion matrix data validasi? (y/n): ").strip().lower() == 'y':
                logger.info("[VALIDASI] Memulai evaluasi prediksi validasi...")

                detector_val = ValidationDetector(
                    predictions_json=json_path,
                    label_dir=labels_path,
                    olah_path=proses_path,
                    log_path=log_path,
                    fold=fold,
                    data=data,
                    model_type=model_type,
                    size=size
                )

                matched_path = detector_val.output_json / f"fold_{fold}_matched.json"

                if not matched_path.exists():
                    logger.info("[VALIDASI] File matched.json belum ada. Dibuat sekarang.")
                    matched_path = detector_val.generate_matched_json(
                        iou_threshold=OLAH_CONFIG["iou_map50"],
                        score_threshold=OLAH_CONFIG["conf_score"]
                    )
                else:
                    logger.info("[VALIDASI] File matched.json ditemukan. Lewati pembuatan ulang.")

                detector_val.generate_confusion_matrix(
                    iou_threshold=OLAH_CONFIG["iou_map50"],
                    score_threshold=OLAH_CONFIG["conf_score"]
                )

            # === Confusion Matrix Uji (YOLO, TRT, atau Semua) ===
            if input("\nApakah ingin membuat confusion matrix data uji? (y/n): ").strip().lower() == 'y':
                engine_mode = input("Evaluasi hasil dari YOLO, TRT, atau keduanya? (yolo/trt/all): ").strip().lower()
                if engine_mode not in {"yolo", "trt", "all"}:
                    raise ValueError("Pilihan engine harus: yolo, trt, atau all")

                engine_paths = {
                    "yolo": hasil_uji_path / "labels",
                    "trt": trt_uji_path / "labels"
                }

                for engine in ["yolo", "trt"]:
                    if engine_mode in {engine, "all"}:
                        detector = PredictionDetector(
                            label_hasil=engine_paths[engine],
                            label_awal=labels_uji_path,
                            olah_path=proses_path,
                            log_path=log_path,
                            size=size,
                            model_type=model_type,
                            data=data,
                            fold=fold,
                            engine=engine
                        )
                        running_det_uji(detector, olah_config=OLAH_CONFIG)

        except Exception as e:
            logger.error(f"\nError processing fold {fold}:")
            logger.error(f"Error details: {str(e)}")
            logger.error(f"Fold {fold} failed: {str(e)}\n")
            continue

if __name__ == "__main__":
    main()
