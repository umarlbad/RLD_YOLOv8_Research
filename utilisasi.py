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

class MetricsUtils:
    @staticmethod
    def compute_iou(box1, box2):
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
    
    @staticmethod
    def compute_ap(recall: np.ndarray, precision: np.ndarray) -> float:
        mrec = np.concatenate(([0.0], recall, [1.0]))
        mpre = np.concatenate(([0.0], precision, [0.0]))
        for i in range(len(mpre) - 2, -1, -1):
            mpre[i] = max(mpre[i], mpre[i + 1])
        recall_levels = np.linspace(0, 1, 101)
        ap = 0.0
        for r in recall_levels:
            idx = np.searchsorted(mrec, r, side='left')
            ap += mpre[idx]
        return ap / len(recall_levels)
    
    @staticmethod
    def nms(pred_boxes, scores, classes, iou_threshold):
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
            ious = [MetricsUtils.compute_iou(pred_boxes[current], box) for box in rest_boxes]
            indices = [i for i, iou in zip(rest_indices, ious) if iou < iou_threshold]

        return keep_boxes, keep_scores, keep_classes

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
            "metrics/precision(B)", "metrics/recall(B)", "metrics/f1_score(B)",
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
            self.logger.info(f"\nData berhasil dimuat dari {csv_path}")
            
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
        eps = 1e-6
        for fold in range(1, self.num_folds + 1):
            df = self.load_data(num_folds=fold, data=self.data)
            if df is None:
                continue

            # Pastikan kolom ada
            for col in self.columns:
                key = col
                if col == "metrics/f1_score":
                    continue  # akan dibuat
                if key not in df.columns:
                    df[key] = 0
            # Hitung f1-score jika precision & recall tersedia
            if "metrics/precision(B)" in df.columns and "metrics/recall(B)" in df.columns:
                df["metrics/f1_score"] = 2 * df["metrics/precision(B)"] * df["metrics/recall(B)"] / \
                                        (df["metrics/precision(B)"] + df["metrics/recall(B)"] + eps)
            else:
                df["metrics/f1_score"] = 0

            if "epoch" not in df.columns:
                df.insert(0, 'epoch', range(1, len(df) + 1))

            fold_output = self.output_path / f"fold_{fold}"
            self.ensure_directory_exists(fold_output)
            for col in self.columns:
                out_file = fold_output / f"fold_{fold}_{col.replace('/', '_')}.csv"
                df[["epoch", col]].to_csv(out_file, index=False)
                self.logger.info(f"File CSV untuk Fold-{fold} {col} tersimpan di path: {out_file}")

    def save_combined_metrics(self):
        combined = {col: pd.DataFrame() for col in self.columns}
        combined_output_path = self.output_path / "combined"
        self.ensure_directory_exists(combined_output_path)

        # 1). Kumpulkan semua fold dan buat combined per kolom
        for fold in range(1, self.num_folds + 1):
            for col in self.columns:
                fn = col.replace('/', '_')
                fold_file = self.output_path / f"fold_{fold}" / f"fold_{fold}_{fn}.csv"
                if not fold_file.exists():
                    self.logger.error(f"Missing file: {fold_file}")
                    continue
                df = pd.read_csv(fold_file)
                if combined[col].empty:
                    combined[col] = df.rename(columns={col: f"Fold_{fold}"})
                else:
                    combined[col][f"Fold_{fold}"] = df[col]

        # 2). Simpan file gabungan dan rata-rata per kolom untuk tiap kolom
        for col, df_comb in combined.items():
            if df_comb.empty:
                continue
            fn = col.replace('/', '_')  # **PENTING:** hitung ulang fn di sini
            combined_file = combined_output_path / f"combined_{fn}.csv"
            df_comb.to_csv(combined_file, index=False)
            self.logger.info(f"File Kombinasi CSV {col} tersimpan di path: {combined_file}")

            # rata-rata per epoch
            fold_cols = [c for c in df_comb.columns if c.startswith("Fold_")]
            if fold_cols:
                mean_df = pd.DataFrame({
                    "epoch": df_comb["epoch"],
                    "mean": df_comb[fold_cols].mean(axis=1)
                })
                mean_file = combined_output_path / f"mean_{fn}.csv"
                mean_df.to_csv(mean_file, index=False)
                self.logger.info(f"File Mean Kombinasi CSV {col} tersimpan di path: {mean_file}")

    def log_validation_metrics(self):
        """
        Logging untuk metrik evaluasi deteksi pada data validasi:
        1) Ringkasan loss pada epoch terakhir (train, val, avg) untuk setiap loss type
        2) Per-fold:
        > Metrik pada epoch terakhir (Precision, Recall, F1, mAP50, mAP50-95, Fitness)
        >Rata-rata per-epoch (Precision, Recall, F1, mAP50, mAP50-95, Fitness)
        3) Agregat rata-rata semua fold (berdasarkan rerata mean per-fold)
        """
        eps = 1e-6

        # 1) Ringkasan loss epoch terakhir (train, val, avg) untuk setiap loss type
        combined_output = self.output_path / "combined"
        loss_types = ["box_loss", "cls_loss", "dfl_loss"]

        for loss in loss_types:
            train_file = combined_output / f"mean_train_{loss}.csv"
            val_file   = combined_output / f"mean_val_{loss}.csv"

            if not train_file.exists() or not val_file.exists():
                self.logger.warning(f"[Loss Summary] File mean untuk '{loss}' tidak lengkap, skip.")
                continue

            # Baca DataFrame (kolom: ["epoch","mean"])
            df_train_loss = pd.read_csv(train_file)
            df_val_loss   = pd.read_csv(val_file)

            if df_train_loss.empty or df_val_loss.empty:
                self.logger.warning(f"[Loss Summary] Data mean_{loss} ada tapi kosong, skip.")
                continue

            # 1a). Ambil baris terakhir (epoch terakhir)
            last_train_loss = float(df_train_loss["mean"].iloc[-1])
            last_val_loss   = float(df_val_loss["mean"].iloc[-1])
            last_avg_loss   = (last_train_loss + last_val_loss) / 2.0

            self.logger.info(
                f"[Loss Last Epoch] pada Fold-{self.num_folds} {loss.replace('_', ' ').upper():<9}>> "
                f"Train: {last_train_loss:.4f}, Val: {last_val_loss:.4f}, "
                f"Avg: {last_avg_loss:.4f} \n"
            )

            # 1b). Mean terhadap semua epoch
            mean_train_all = df_train_loss["mean"].mean()
            mean_val_all   = df_val_loss["mean"].mean()
            mean_avg_all   = (mean_train_all + mean_val_all) / 2.0

            self.logger.info(
                f"[Loss Mean semua Epoch] pada fold-{self.num_folds} {loss.replace('_', ' ').upper():<9}>> "
                f"Train: {mean_train_all:.4f}, Val: {mean_val_all:.4f}, "
                f"Avg: {mean_avg_all:.4f} \n"
            )

        # 2). Metrik deteksi per-fold
        # container untuk agregasi semua fold
        precisions, recalls = [], []
        f1_scores, map50s, map5095s = [], [], []
        fitnesses = []

        for fold in range(1, self.num_folds + 1):
            df = self.load_data(num_folds=fold, data=self.data)
            if df is None or df.empty:
                self.logger.error(f"[Fold-{fold}] Gagal memuat data untuk logging metrik.")
                continue

            # cek kolom penting
            required = [
                "metrics/precision(B)", "metrics/recall(B)",
                "metrics/mAP50(B)", "metrics/mAP50-95(B)"
            ]
            missing = [c for c in required if c not in df.columns]
            if missing:
                self.logger.error(f"[Fold {fold}] Kolom metrik hilang: {missing}")
                continue

            # ambil series per-epoch
            prec_series   = df["metrics/precision(B)"]
            rec_series    = df["metrics/recall(B)"]
            map50_series  = df["metrics/mAP50(B)"]
            map5095_series= df["metrics/mAP50-95(B)"]

            # mengambil metrik pada epoch terakhir
            last = df.iloc[-1]
            last_precision = float(last["metrics/precision(B)"])
            last_recall = float(last["metrics/recall(B)"])
            last_map50 = float(last["metrics/mAP50(B)"])
            last_map5095 = float(last["metrics/mAP50-95(B)"])
            last_f1 = 2 * last_precision * last_recall / (last_precision + last_recall + eps)
            last_fitness   = (0.1 * last_precision + 0.1 * last_recall + 0.1 * last_map50 + 0.7 * last_map5095)

            self.logger.info(
                f"\n[Fold {fold}] pada Epoch Terakhir Data Validasi: \n"
                f"Presisi: {last_precision:.4f}, Recall: {last_recall:.4f}"
                f"\nF1-Score: {last_f1:.4f}, mAP50: {last_map50:.4f}"
                f"\nmAP50-95: {last_map5095:.4f}, Fitness: {last_fitness:.4f}"
            )
            # rata-rata simple untuk precision, recall, mAP50, mAP50-95
            avg_precision = prec_series.mean()
            avg_recall    = rec_series.mean()
            avg_map50     = map50_series.mean()
            avg_map5095   = map5095_series.mean()

            # F1 per-epoch >> mean
            f1_per_epoch = 2 * prec_series * rec_series / (prec_series + rec_series + eps)
            avg_f1 = f1_per_epoch.mean()

            # Fitness per-epoch >> mean
            fitness_per_epoch = (
                0.1 * prec_series +
                0.1 * rec_series +
                0.1 * map50_series +
                0.7 * map5095_series
            )
            avg_fitness = fitness_per_epoch.mean()

            # simpan untuk agregasi akhir
            precisions.append(avg_precision)
            recalls.append(avg_recall)
            f1_scores.append(avg_f1)
            map50s.append(avg_map50)
            map5095s.append(avg_map5095)
            fitnesses.append(avg_fitness)

            # log per fold
            self.logger.info(
                f"\n[Fold {fold}] Hasil Mean Metriks dengan Data {self.data.upper()} Validasi: \n"
                f"Presisi: {avg_precision:.4f}, Recall: {avg_recall:.4f}"
                f"\nF1_Score: {avg_f1:.4f}, mAP50: {avg_map50:.4f}, "
                f"\nmAP50-95: {avg_map5095:.4f}, Fitness: {avg_fitness:.4f}"
            )

        # logging metrik aggregat untuk semua fold
        if precisions:
            agg_p     = np.mean(precisions)
            agg_r     = np.mean(recalls)
            agg_m50   = np.mean(map50s)
            agg_m5095 = np.mean(map5095s)
            agg_f1    = np.mean(f1_scores)
            agg_ftn   = np.mean(fitnesses)

            self.logger.info(
                f"\n[Aggregate] Mean Data {self.data.upper()} Latih untuk Semua Fold: \n"
                f"Presisi: {agg_p:.4f}, Recall: {agg_r:.4f} "
                f"\nF1: {agg_f1:.4f}, mAP50: {agg_m50:.4f}, "
                f"\nmAP50-95: {agg_m5095:.4f}, Fitness: {agg_ftn:.4f}"
            )

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
            "metrics/precision(B)", "metrics/recall(B)", "metrics/f1_score(B)",
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

    def _plot_curve(self, x, ys_dict, title:str, ylabel:str, save_path, use_style_variation=False):
        plt.figure(figsize=(12, 10))
        markers = ['o', 's', 'D', '^', 'v', '*', 'x', '+', 'p', 'h']
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

        for i, (label, y) in enumerate(ys_dict.items()):
            marker = markers[i % len(markers)] if use_style_variation else 'o'
            color = colors[i % len(colors)] if use_style_variation else 'b'
            plt.plot(x, y, marker=marker, linestyle='-', color=color, label=label)

        plt.xlabel("Epoch", fontsize=15)
        plt.ylabel(ylabel, fontsize=15)
        plt.title(title, fontsize=16)
        plt.legend()
        plt.grid(True)
        plt.savefig(save_path, dpi=300)
        plt.close()
        self.logger.info(f"Plot disimpan di Path: {save_path}")

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
                if df is None or "epoch" not in df or var not in df.columns:
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
                title=f"Grafik Kombinasi Data Validasi Fitting Model Fold - {data.upper()} - {var}", ylabel=var,
                save_path=self.results_comp / f"Cbd_{filename}.png",
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
                f"Grafik Data Validasi Fitting Model Mean Kombinasi Fold terhadap Epoch - {data.upper()} - {var}", var,
                self.results_comp / f"Mean_{filename}.png"
            )

    def plot_mean_losses_curves(self, data):
        """
        Plot overlay mean train vs mean val curves dalam satu plot per jenis loss.
        """
        loss_types = ["box_loss", "cls_loss", "dfl_loss"]

        for loss in loss_types:
            # Path ke CSV mean_train_{loss}.csv dan mean_val_{loss}.csv
            train_path = self.file_path / "combined" / f"mean_train_{loss}.csv"
            val_path   = self.file_path / "combined" / f"mean_val_{loss}.csv"

            df_train = self._load_csv(train_path)
            df_val   = self._load_csv(val_path)
            if df_train is None or df_val is None:
                continue
            if "epoch" not in df_train.columns or "mean" not in df_train.columns:
                continue
            if "epoch" not in df_val.columns or "mean" not in df_val.columns:
                continue

            epochs = df_train["epoch"]
            loss_train = df_train["mean"]
            loss_val   = df_val["mean"]

            plt.figure(figsize=(12, 10))
            # Overlay train & val di satu axes
            plt.plot(
                epochs,
                loss_train,
                label="Latih",
                color="tab:blue",
                marker="o",
                markersize=5,
                linewidth=1.5
            )
            plt.plot(
                epochs,
                loss_val,
                label="Validasi",
                color="tab:orange",
                linestyle="--",
                marker="x",
                markersize=5,
                linewidth=1.5
            )

            plt.title(f"Mean Loss Data Latih dan Validasi -{data.upper()}-{loss.capitalize()}", fontsize=18)
            plt.xlabel("Epoch", fontsize=15)
            plt.ylabel(f"{loss}", fontsize=15)
            plt.legend()
            plt.grid(True)

            save_path = self.results_comp / f"MeanTrainVal_{loss}.png"
            plt.tight_layout()
            plt.savefig(save_path, dpi=300)
            plt.close()

            self.logger.info(f"Plot untuk Mean Train+Val pada Losses {loss} disimpan di path: {save_path} \n")
            
    def plot_all(self, target_folds, data):
        self.plot_variable(target_folds, data)
        self.plot_combined(data)
        self.plot_mean_curves(data)
        self.plot_mean_losses_curves(data)
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
            "0": "bercak cokelat",
            "1": "bercak cokelat tipis",
            "2": "blas daun",
            "3": "lepuh daun",
            "4": "hawar daun bakteri",
            "5": "sehat"
        }
        self.label_mapping = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5}
    
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

    def generate_matched_json(self, iou_threshold: float, score_threshold: float):
        """
        Buat dua JSON:
        - raw  : semua prediksi setelah NMS, tanpa threshold
        - filtered: hanya prediksi dengan score>=score_threshold
                    dan yang match GT dengan IoU>=iou_threshold
        Sertakan logging statistik untuk kedua set.
        """
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
        ground_truth = self._load_gt()  # {image_id: {true_classes, true_bboxes}}

        # 4. Kelompokkan prediksi per image (raw) — TANPA filter score
        preds_by_image_raw = defaultdict(list)
        for p in predictions:
            preds_by_image_raw[p["image_id"]].append(p)

        # siapkan struktur dan counters
        raw_matched      = {}
        filtered_matched = {}

        total_gt=total_pred_raw=total_matched_gt_raw =unmatched_gts_raw=total_pred_flt=total_matched_gt_flt=unmatched_gts_flt = 0

        # 5. Proses per image
        for image_id, gt in ground_truth.items():
            true_classes = gt["true_classes"]
            true_bboxes  = gt["true_bboxes"]
            total_gt     += len(true_bboxes)

            # --- RAW: NMS semua prediksi ---
            preds_raw = preds_by_image_raw.get(image_id, [])
            if not preds_raw:
                unmatched_gts_raw += 1
            boxes_all  = [p["bbox"]   for p in preds_raw]
            scores_all = [p["score"]  for p in preds_raw]
            labs_all   = [p["pred_label"] for p in preds_raw]

            nms_boxes_all, nms_scores_all_np, nms_labels_all_np = (
                MetricsUtils.nms(boxes_all, scores_all, labs_all, iou_threshold)
            )
            nms_scores_all = [float(s) for s in nms_scores_all_np]
            nms_labels_all = [int(l)   for l in nms_labels_all_np]
            total_pred_raw += len(nms_boxes_all)

            # hitung matched GT pada raw
            matched_idxs_raw = set()
            for gt_idx, gt_box in enumerate(true_bboxes):
                for pred_box in nms_boxes_all:
                    if MetricsUtils.compute_iou(gt_box, pred_box) >= iou_threshold:
                        matched_idxs_raw.add(gt_idx)
                        break
            total_matched_gt_raw += len(matched_idxs_raw)

            raw_matched[image_id] = {
                "true_classes": true_classes,
                "true_bboxes":  true_bboxes,
                "pred_classes": nms_labels_all,
                "pred_bboxes":  nms_boxes_all,
                "scores":       nms_scores_all
            }

            # --- FILTERED: score+IoU threshold applied ---
            filt_labels, filt_boxes, filt_scores = [], [], []
            for cls, box, sc in zip(nms_labels_all, nms_boxes_all, nms_scores_all):
                if sc < score_threshold:
                    continue
                # cek match GT
                if any(MetricsUtils.compute_iou(gt_box, box) >= iou_threshold for gt_box in true_bboxes):
                    filt_labels.append(cls)
                    filt_boxes.append(box)
                    filt_scores.append(sc)

            if not filt_labels:
                unmatched_gts_flt += 1
            total_pred_flt += len(filt_labels)

            # hitung matched GT pada filtered
            matched_idxs_flt = set()
            for gt_idx, gt_box in enumerate(true_bboxes):
                for box in filt_boxes:
                    if MetricsUtils.compute_iou(gt_box, box) >= iou_threshold:
                        matched_idxs_flt.add(gt_idx)
                        break
            total_matched_gt_flt += len(matched_idxs_flt)

            filtered_matched[image_id] = {
                "true_classes": true_classes,
                "true_bboxes":  true_bboxes,
                "pred_classes": filt_labels,
                "pred_bboxes":  filt_boxes,
                "scores":       filt_scores
            }

        # cek prediksi untuk image tanpa GT
        unmatched_preds_raw = sum(
            1 for img in preds_by_image_raw if img not in ground_truth
        )
        unmatched_preds_flt = unmatched_preds_raw  # filtered tidak menambah image baru

        # 6. Logging statistik RAW
        total_missed_raw = total_gt - total_matched_gt_raw
        self.logger.info("\n[STATISTIK MATCHED JSON RAW - VALIDASI]")
        self.logger.info(f"Total GT BBoxes         : {total_gt}")
        self.logger.info(f"Total Prediksi BBoxes   : {total_pred_raw}")
        self.logger.info(f"GT yang Terdeteksi      : {total_matched_gt_raw}")
        self.logger.info(f"GT Tidak Terdeteksi     : {total_missed_raw}")
        self.logger.info(f"Image tanpa prediksi    : {unmatched_gts_raw}")
        self.logger.info(f"Prediksi di luar GT     : {unmatched_preds_raw}")

        # 7. Logging statistik FILTERED
        total_missed_flt = total_gt - total_matched_gt_flt
        self.logger.info("\n[STATISTIK MATCHED JSON FILTERED - VALIDASI]")
        self.logger.info(f"Total GT BBoxes         : {total_gt}")
        self.logger.info(f"Total Prediksi BBoxes   : {total_pred_flt}")
        self.logger.info(f"GT yang Terdeteksi      : {total_matched_gt_flt}")
        self.logger.info(f"GT Tidak Terdeteksi     : {total_missed_flt}")
        self.logger.info(f"Image tanpa prediksi    : {unmatched_gts_flt}")
        self.logger.info(f"Prediksi di luar GT     : {unmatched_preds_flt}")

        # 8. Simpan kedua JSON
        out_raw = self.output_json / f"fold_{self.fold}_matched_raw.json"
        out_flt = self.output_json / f"fold_{self.fold}_matched_filtered.json"
        with open(out_raw, "w") as f:
            json.dump(raw_matched, f, indent=4)
        with open(out_flt, "w") as f:
            json.dump(filtered_matched, f, indent=4)

        self.logger.info(f"\nRaw matched JSON tersimpan di: {out_raw}")
        self.logger.info(f"Filtered matched JSON tersimpan di: {out_flt}")

        return out_raw, out_flt
    
    def generate_confusion_matrix(self, filt_json_path: str, iou_threshold: float):
        """
        Bangun dan simpan confusion matrix multi‐kelas
        berdasarkan file filtered JSON, lalu plot dengan seaborn.
        """
        # 1. Load filtered JSON
        with open(filt_json_path, "r") as f:
            matched_data = json.load(f)

        y_true, y_pred = [], []

        # 2. Pairing GT ↔ Pred (sisa jadi no_pred / no_gt)
        for _, data in matched_data.items():
            gt_classes = data.get("true_classes", [])
            gt_bboxes = data.get("true_bboxes", [])
            pred_classes = data.get("pred_classes", [])
            pred_bboxes = data.get("pred_bboxes", [])

            used_pred = set()

            # A) TP / FN: cocokkan tiap GT ke satu pred terbaik (bertumpu IoU)
            for i, (gt_cls, gt_box) in enumerate(zip(gt_classes, gt_bboxes)):
                best_j, best_iou = -1, 0.0
                for j, pred_box in enumerate(pred_bboxes):
                    if j in used_pred:
                        continue
                    iou = MetricsUtils.compute_iou(gt_box, pred_box)
                    if iou > best_iou:
                        best_iou, best_j = iou, j

                if best_j >= 0 and best_iou >= iou_threshold:
                    y_true.append(gt_cls)
                    y_pred.append(pred_classes[best_j])
                    used_pred.add(best_j)
                else:
                    # GT tanpa pred memuaskan → FN
                    y_true.append(gt_cls)
                    y_pred.append("background") #no_pred

            # B) FP: sisa pred yang tak terpakai
            for j, pred_cls in enumerate(pred_classes):
                if j not in used_pred:
                    y_true.append("background") #no_gt
                    y_pred.append(pred_cls)

        # 3. CAST semua label ke str agar robust di confusion_matrix
        y_true = [str(v) for v in y_true]
        y_pred = [str(v) for v in y_pred]

        # 4. Siapkan labels dan display names dari mapping (string key)
        label_ids = [str(i) for i in self.valid_labels.keys()]
        display_names = [self.valid_labels[str(i)] for i in self.valid_labels.keys()] + ["background"]
        cm_labels = label_ids + ["background"]
        
        # Debugging log
        self.logger.info(f"[generate_cm_valid] Labels: {cm_labels}")
        self.logger.info(f"[generate_cm_valid] Display: {display_names}")
        self.logger.info(f"[generate_cm_valid] y_true samples: {y_true[:15]}")
        self.logger.info(f"[generate_cm_valid] y_pred samples: {y_pred[:15]}")
        self.logger.info(f"[generate_cm_valid] Unique y_true: {set(y_true)}")
        self.logger.info(f"[generate_cm_valid] Unique y_pred: {set(y_pred)}")
        self.logger.info(f"[generate_cm_valid] Total: {len(y_true)}, {len(y_pred)}")

        # 5. Compute Confusion Matrix
        cm = confusion_matrix(y_true, y_pred, labels=cm_labels)
        self.logger.info(f"[generate_cm] Confusion Matrix array: \n{cm} \n")

        idx_obj = len(cm_labels) - 1  # Asumsi objectness selalu di index terakhir
        mask = np.zeros_like(cm, dtype=bool)
        mask[idx_obj, idx_obj] = True

        annot = cm.astype(str)
        annot[idx_obj, idx_obj] = ""  # Blank cell untuk objectness-objectness

        # 6. Plotting menggunakan seaborn
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm.T, annot=annot.T, fmt='s', cmap='Greens', mask=mask.T,
                    xticklabels=display_names, yticklabels=display_names,
                    cbar=True, square=True, linewidths=.5)
        plt.title(f"Confusion Matrix Multi-Kelas Data Validasi - Fold-{self.fold}-{self.data.upper()}", fontsize=15)
        plt.xlabel("True Labels", fontsize=13)
        plt.ylabel("Predicted Labels", fontsize=13)
        plt.xticks(rotation=90, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout(pad=2.5)

        # 7. Simpan dan Logging
        output_cm = self.output_cm_d / f"0_fold_{self.fold}_cm.png"
        plt.savefig(output_cm, format="png", dpi=300)
        plt.close()
        self.logger.info(f"[INFO] Confusion Matrix saved to: {output_cm}")

        return output_cm

    def log_metrics(self, filt_json_path: str, iou_thresholds: float, score_threshold: float):
        """
        Logging metrik evaluasi deteksi pada data validasi berdasarkan filtered matched JSON:
        - Precision, Recall, F1-score dan AP@50 untuk setiap kelas
        - mAP@50-95 (rata-rata AP pada IoU 0.50, 0.55, ..., 0.95)
        - Fitness = 0.1*Precision + 0.1*Recall + 0.1*mAP50 + 0.7*mAP50-95
        """
        # 1. Pastikan iou_thresholds berupa list (boleh float atau list)
        if isinstance(iou_thresholds, float):
            iou_thresholds = [iou_thresholds]

        # 2. Muat hasil pencocokan prediksi dan ground truth (filtered matched JSON)
        with open(filt_json_path, "r") as f:
            matched_data = json.load(f)

        # 3. Siapkan struktur penampung: 
        #    - pred_by_class: {kelas: {iou: [(score, TP/FP), ...]}}
        #    - fn_per_class:  {kelas: {iou: FN_count}}
        pred_by_class = defaultdict(lambda: defaultdict(list))
        fn_per_class  = defaultdict(lambda: defaultdict(int))
        all_classes   = set()

        # 4. Iterasi semua gambar untuk menghitung TP, FP, dan FN per kelas dan threshold IoU
        for entry in matched_data.values():
            gt_classes  = entry.get("true_classes", [])
            gt_bboxes   = entry.get("true_bboxes", [])
            pred_classes= entry.get("pred_classes", [])
            pred_bboxes = entry.get("pred_bboxes", [])
            scores      = entry.get("scores", [])

            all_classes.update(gt_classes)
            # Tandai prediksi yang sudah "dipakai" untuk matching GT
            for thr in iou_thresholds:
                used = set()
                # --- True Positive & False Positive ---
                for pc, pb, sc in zip(pred_classes, pred_bboxes, scores):
                    if sc < score_threshold:
                        continue  # prediksi di bawah confidence threshold → skip
                    matched = False
                    for j, (tc, tb) in enumerate(zip(gt_classes, gt_bboxes)):
                        if j in used: continue  # GT sudah dipakai pred lain
                        if pc == tc and MetricsUtils.compute_iou(tb, pb) >= thr:
                            # Jika prediksi benar kelas & cukup overlap → TP
                            pred_by_class[pc][thr].append((sc, 1))
                            used.add(j)
                            matched = True
                            break
                    if not matched:
                        # Jika tidak match GT mana pun → FP
                        pred_by_class[pc][thr].append((sc, 0))
                # --- False Negative (GT yang tidak terdeteksi) ---
                for j, tc in enumerate(gt_classes):
                    if j not in used:
                        fn_per_class[tc][thr] += 1

        # 5. Logging per kelas (khusus threshold IoU = 0.5)
        per_class_metrics = {}
        self.logger.info(f"\n Logging Metrik Deteksi Data Validasi {self.data.upper()} pada Fold-{self.fold}: ")
        for cls in sorted(all_classes):
            per_class_metrics[cls] = {}
            for thr in iou_thresholds:
                entries = pred_by_class[cls][thr]
                if not entries:
                    per_class_metrics[cls][thr] = {"precision": 0.0, "recall": 0.0, "f1": 0.0, "ap": 0.0}
                    continue
                # Urutkan prediksi dari confidence tertinggi
                entries.sort(key=lambda x: -x[0])
                tp_fp  = np.array([e[1] for e in entries])  # 1=TP, 0=FP
                tp_cum = np.cumsum(tp_fp)
                fp_cum = np.cumsum(1 - tp_fp)
                tp_final = tp_cum[-1]
                fn_final = fn_per_class[cls][thr]
                n_gt     = tp_final + fn_final
                if n_gt == 0:
                    per_class_metrics[cls][thr] = {"precision": 0.0, "recall": 0.0, "f1": 0.0, "ap": 0.0}
                    continue
                # Hitung metrik per kelas per threshold
                precision = tp_final / (tp_final + fp_cum[-1] + 1e-6)
                recall    = tp_final / (n_gt + 1e-6)
                f1        = 2 * precision * recall / (precision + recall + 1e-6)
                recall_curve = tp_cum / (n_gt + 1e-6)
                prec_curve   = tp_cum / (tp_cum + fp_cum + 1e-6)
                ap          = MetricsUtils.compute_ap(recall_curve, prec_curve)
                per_class_metrics[cls][thr] = {
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "ap": ap
                }

                # Logging hasil per kelas @IoU=0.5
                if thr == 0.5:
                    self.logger.info(
                        f"[VALIDASI][Fold-{self.fold}][Kelas {cls} @IoU=0.50] "
                        f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, AP@50: {ap:.4f} \n"
                    )

        # 6. Logging mean per fold (khusus threshold IoU = 0.5)
        if 0.5 in iou_thresholds:
            precisions = [per_class_metrics[cls][0.5]["precision"] for cls in all_classes]
            recalls    = [per_class_metrics[cls][0.5]["recall"]    for cls in all_classes]
            f1s        = [per_class_metrics[cls][0.5]["f1"]        for cls in all_classes]
            ap50s      = [per_class_metrics[cls][0.5]["ap"]        for cls in all_classes]
            mean_p  = float(np.mean(precisions)) if precisions else 0.0
            mean_r  = float(np.mean(recalls))    if recalls    else 0.0
            mean_f1 = float(np.mean(f1s))        if f1s        else 0.0
            mean_ap50 = float(np.mean(ap50s))    if ap50s      else 0.0
        else:
            mean_p = mean_r = mean_f1 = mean_ap50 = 0.0

        # 7. Logging mAP50-95 (rata-rata dari semua AP tiap threshold)
        aps_mean_per_thr = []
        for thr in iou_thresholds:
            aps_this_thr = [per_class_metrics[cls][thr]["ap"] for cls in all_classes]
            aps_this_thr = [a for a in aps_this_thr if a > 0]
            if aps_this_thr:
                aps_mean_per_thr.append(np.mean(aps_this_thr))
        map5095 = float(np.mean(aps_mean_per_thr)) if aps_mean_per_thr else 0.0

        # 8. Logging fitness
        fitness = 0.1 * mean_p + 0.1 * mean_r + 0.1 * mean_ap50 + 0.7 * map5095

        self.logger.info(
            f"\n[VALIDASI][Fold-{self.fold}][Summary @IoU=0.50]: "
            f"Mean Precision: {mean_p:.4f}, Mean Recall: {mean_r:.4f}, Mean F1: {mean_f1:.4f}, "
            f"Mean mAP50: {mean_ap50:.4f}, Mean mAP50-95: {map5095:.4f}, Fitness: {fitness:.4f}\n"
        )

        self.logger.info(f"\nLogging Metrik Deteksi Data Validasi {self.data.upper()} pada Fold-{self.fold} Sudah Didapatkan! \n")

class PredictionDetector:
    def __init__(self, label_hasil, label_awal, olah_path, log_path, size, model_type, data, fold, engine):
        self.label_hasil = Path(label_hasil)
        self.label_awal = Path(label_awal)
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
            '0': 'bercak cokelat',
            '1': 'bercak cokelat tipis',
            '2': 'blas daun',
            '3': 'lepuh daun',
            '4': 'hawar daun bakteri',
            '5': 'sehat'
        }
        self.label_mapping = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5}

    def _load_gt(self, force_reload=False):
        gt_json_path = self.output_json_awal / f"fold_{self.fold}_uji_awal_labels.json"
        # Jika force_reload False dan file ada, baca dari cache
        if gt_json_path.exists() and not force_reload:
            with open(gt_json_path, "r") as f:
                return json.load(f)

        self.logger.info(f"\nPath Label Data Awal Uji yang dijalankan yakni : {self.label_awal} \n")

        if not os.path.exists(self.label_awal):
            raise FileNotFoundError(f"Label directory not found: {self.label_awal}")

        ground_truth = {}
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

        self.logger.info(f"\nfile JSON Prediksi dari Data Awal Uji path yakni: {gt_json_path}")
        return ground_truth
    
    def _load_pred(self):
        pred_json_path = self.output_json_hasil / f"{self.engine}_fold_{self.fold}_hasil_uji_labels.json"
        if pred_json_path.exists():
            with open(pred_json_path, "r") as f:
                return json.load(f)
        
        self.logger.info(f"\nPath Label Data Hasil Uji yang dijalankan yakni : {self.label_hasil} \n")

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

        self.logger.info(f"\nfile JSON Prediksi dari Data Hasil Uji path yakni: {pred_json_path}")

        return predictions
    
    def generate_matched_json(self, score_threshold: float, iou_threshold: float):
        """
        Buat dua JSON:
        > matched_raw.json      : semua prediksi setelah NMS, tanpa filter apapun
        > matched_filtered.json : hanya prediksi dengan score>=score_threshold dan IoU>=iou_threshold (sesuai operating point)
        Keduanya dilengkapi logging statistik.
        """
        self.logger.info(
            f"\nGenerate matched JSON dengan score_thr={score_threshold}, IoU_thr={iou_threshold}…"
        )

        gt_data   = self._load_gt(force_reload=True)   # {image_id: {true_classes, true_bboxes}}
        pred_data = self._load_pred() # {image_id: {pred_classes, pred_bboxes, scores}}

        raw_matched      = {}
        filtered_matched = {}

        # Counters untuk statistik
        total_gt            = 0
        total_pred_raw      = 0
        total_matched_gt_raw= 0
        unmatched_gt_raw    = 0

        total_pred_flt      = 0
        total_matched_gt_flt= 0
        unmatched_gt_flt    = 0

        # Proses setiap image_id (union GT+pred)
        for image_id in sorted(set(gt_data) | set(pred_data)):
            gt_info    = gt_data.get(image_id, {})
            true_cls   = gt_info.get("true_classes", [])
            true_boxes = gt_info.get("true_bboxes", [])
            total_gt  += len(true_boxes)

            pd_info    = pred_data.get(image_id, {})
            raw_boxes  = pd_info.get("pred_bboxes", [])
            raw_scores = pd_info.get("scores", [])
            raw_cls    = pd_info.get("pred_classes", [])

            # --- RAW: Confidence filtering = OFF, pakai seluruh raw_boxes ---
            # 1) NMS pada semua raw pred
            if raw_boxes:
                nms_boxes_all, nms_scores_all_np, nms_cls_all_np = \
                    MetricsUtils.nms(raw_boxes, raw_scores, raw_cls, iou_threshold)
                nms_scores_all = [float(s) for s in nms_scores_all_np]
                nms_cls_all    = [int(c)   for c in nms_cls_all_np]
            else:
                nms_boxes_all = []
                nms_scores_all = []
                nms_cls_all    = []

            total_pred_raw += len(nms_boxes_all)

            # Hitung matched GT (raw)
            matched_idxs_raw = set()
            for idx, gt_box in enumerate(true_boxes):
                for pbox in nms_boxes_all:
                    if MetricsUtils.compute_iou(gt_box, pbox) >= iou_threshold:
                        matched_idxs_raw.add(idx)
                        break
            total_matched_gt_raw += len(matched_idxs_raw)
            if not nms_boxes_all:
                unmatched_gt_raw += 1

            raw_matched[image_id] = {
                "true_classes": true_cls,
                "true_bboxes":  true_boxes,
                "pred_classes": nms_cls_all,
                "pred_bboxes":  nms_boxes_all,
                "scores":       nms_scores_all
            }

            # --- FILTERED: Confidence + IoU thresholding ---
            filt_boxes, filt_scores, filt_cls = [], [], []
            for box, sc, cl in zip(nms_boxes_all, nms_scores_all, nms_cls_all):
                if sc < score_threshold:
                    continue
                # pastikan minimal satu GT match IoU
                if any(MetricsUtils.compute_iou(gt_box, box) >= iou_threshold for gt_box in true_boxes):
                    filt_boxes.append(box)
                    filt_scores.append(sc)
                    filt_cls.append(cl)

            total_pred_flt += len(filt_boxes)

            # Hitung matched GT (filtered)
            matched_idxs_flt = set()
            for idx, gt_box in enumerate(true_boxes):
                for box in filt_boxes:
                    if MetricsUtils.compute_iou(gt_box, box) >= iou_threshold:
                        matched_idxs_flt.add(idx)
                        break
            total_matched_gt_flt += len(matched_idxs_flt)
            if not filt_boxes:
                unmatched_gt_flt += 1

            filtered_matched[image_id] = {
                "true_classes": true_cls,
                "true_bboxes":  true_boxes,
                "pred_classes": filt_cls,
                "pred_bboxes":  filt_boxes,
                "scores":       filt_scores
            }

        # Statistik prediksi tanpa GT
        unmatched_preds_raw = sum(1 for img in raw_matched if img not in gt_data)
        unmatched_preds_flt = unmatched_preds_raw

        # Logging statistik RAW
        missed_raw = total_gt - total_matched_gt_raw
        self.logger.info("\n[STATISTIK MATCHED JSON RAW]")
        self.logger.info(f"Total GT BBoxes        : {total_gt}")
        self.logger.info(f"Total Pred BBoxes RAW  : {total_pred_raw}")
        self.logger.info(f"GT Terdeteksi RAW      : {total_matched_gt_raw}")
        self.logger.info(f"GT Tidak Terdeteksi RAW: {missed_raw}")
        self.logger.info(f"Gambar tanpa Pred RAW     : {unmatched_gt_raw}")
        self.logger.info(f"Hasil Pred tanpa GT RAW      : {unmatched_preds_raw}")

        # Logging statistik FILTERED
        missed_flt = total_gt - total_matched_gt_flt
        self.logger.info("\n[STATISTIK MATCHED JSON FILTERED]")
        self.logger.info(f"Total GT BBoxes             : {total_gt}")
        self.logger.info(f"Total Pred BBoxes FILTERED  : {total_pred_flt}")
        self.logger.info(f"GT Terdeteksi FILTERED      : {total_matched_gt_flt}")
        self.logger.info(f"GT Tidak Terdeteksi FILTERED: {missed_flt}")
        self.logger.info(f"Gambar tanpa Pred FILTERED     : {unmatched_gt_flt}")
        self.logger.info(f"Hasil Pred tanpa GT FILTERED      : {unmatched_preds_flt}")

        # 6. Simpan dua JSON
        out_raw = self.json_matched / f"{self.engine}_fold_{self.fold}_uji_matched_raw.json"
        out_flt = self.json_matched / f"{self.engine}_fold_{self.fold}_uji_matched_filtered.json"
        with open(out_raw, "w") as f:
            json.dump(raw_matched, f, indent=4)
        with open(out_flt, "w") as f:
            json.dump(filtered_matched, f, indent=4)

        self.logger.info(f"\nRaw JSON   : {out_raw}")
        self.logger.info(f"Filtered JSON: {out_flt}\n")

        return out_raw, out_flt

    def generate_cm(self, filtered_json_path, iou_threshold):
        # 1. Load JSON
        with open(filtered_json_path, "r") as f:
            matched_data = json.load(f)

        y_true, y_pred = [], []

        # 2. Matching GT ↔ Pred berdasarkan IoU
        for _, data in matched_data.items():
            gt_classes = data.get("true_classes", [])
            gt_bboxes = data.get("true_bboxes", [])
            pred_classes = data.get("pred_classes", [])
            pred_bboxes = data.get("pred_bboxes", [])

            used_pred = set()
            # A) Untuk setiap GT cari pred terbaik
            for gt_cls, gt_box in zip(gt_classes, gt_bboxes):
                best_iou, best_j = 0, -1
                for j, pred_box in enumerate(pred_bboxes):
                    if j in used_pred:
                        continue
                    iou = MetricsUtils.compute_iou(gt_box, pred_box)
                    if iou > best_iou:
                        best_iou, best_j = iou, j
                if best_iou >= iou_threshold:
                    y_true.append(gt_cls)
                    y_pred.append(pred_classes[best_j])
                    used_pred.add(best_j)
                else:
                    y_true.append(gt_cls)
                    y_pred.append("background") #no_pred
            # B) Tambahkan FP dari prediksi yang tidak dipakai
            for j, pred_cls in enumerate(pred_classes):
                if j not in used_pred:
                    y_true.append("background") #no_gt
                    y_pred.append(pred_cls)

        # CAST SEMUA LABEL KE STR!
        y_true = [str(v) for v in y_true]
        y_pred = [str(v) for v in y_pred]

        # 3. Auto-labeling untuk label unik
        unique_gt_labels = set(v for v in y_true if v not in ["background"])
        unique_pred_labels = set(v for v in y_pred if v not in ["background"])
        all_label_ids = sorted(unique_gt_labels | unique_pred_labels, key=lambda x: int(x) if x.isdigit() else x)

        # Mapping label (gunakan self.valid_labels jika ada dan cocok)
        label_map = getattr(self, "valid_labels", None)
        if not label_map or any(str(l) not in label_map for l in all_label_ids):
            label_map = {str(l): f"Class {l}" for l in all_label_ids}

        display_names = [label_map[str(i)] for i in all_label_ids] + ["background"]
        cm_labels = [str(i) for i in all_label_ids] + ["background"]

        # Debugging log
        self.logger.info(f"[generate_cm] Labels: {cm_labels}")
        self.logger.info(f"[generate_cm] Display: {display_names}")
        self.logger.info(f"[generate_cm] y_true samples: {y_true[:15]}")
        self.logger.info(f"[generate_cm] y_pred samples: {y_pred[:15]}")
        self.logger.info(f"[generate_cm] Unique y_true: {set(y_true)}")
        self.logger.info(f"[generate_cm] Unique y_pred: {set(y_pred)}")
        self.logger.info(f"[generate_cm] Total: {len(y_true)}, {len(y_pred)}")

        # 4. Compute Confusion Matrix
        cm = confusion_matrix(y_true, y_pred, labels=cm_labels)
        self.logger.info(f"[generate_cm] Confusion Matrix array: \n{cm} \n")
        idx_obj = len(cm_labels) - 1  # Asumsi objectness selalu di index terakhir
        mask = np.zeros_like(cm, dtype=bool)
        mask[idx_obj, idx_obj] = True

        annot = cm.astype(str)
        annot[idx_obj, idx_obj] = ""  # Blank cell untuk objectness-objectness

        # 5. Plot Confusion Matrix
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm.T, annot=annot.T, fmt='s', cmap='Reds', mask=mask.T,
                    xticklabels=display_names, yticklabels=display_names,
                    cbar=True, square=True, linewidths=.5)

        plt.title(f"Confusion Matrix Multi-Kelas - Fold-{self.fold} Data Uji {self.data.upper()}-{self.engine.upper()}", fontsize=15)
        plt.xlabel("True Labels", fontsize=13)
        plt.ylabel("Predicted Labels", fontsize=13)
        plt.xticks(rotation=90, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout(pad=2.5)

        output_cm = self.cm_uji / f"0_fold_{self.fold}_cm_uji_{self.engine}.png"
        plt.savefig(output_cm, format="png", dpi=300)
        plt.close()
        self.logger.info(f"\nConfusion Matrix tersimpan di path: {output_cm}")

        return output_cm
    
    def log_metrics_uji(self, path_matched: str, conf_threshold: float, iou_thresholds: list):
        
        # 1. Pastikan iou_thresholds berupa list (boleh float atau list)
        if isinstance(iou_thresholds, float):
            iou_thresholds = [iou_thresholds]

        # 2. Muat hasil pencocokan prediksi dan ground truth (filtered matched JSON)
        with open(path_matched) as f:
            data = json.load(f)

        # 3. Siapkan struktur penampung: 
        #    - pred_by_class: {kelas: {iou: [(score, TP/FP), ...]}}
        #    - fn_per_class:  {kelas: {iou: FN_count}}
        pred_by_class = defaultdict(lambda: defaultdict(list))
        fn_per_class  = defaultdict(lambda: defaultdict(int))
        all_classes   = set()

        # 4. Iterasi semua gambar untuk menghitung TP, FP, dan FN per kelas dan threshold IoU
        for item in data.values():
            gt_cls, gt_box   = item["true_classes"], item["true_bboxes"]
            pd_cls, pd_box   = item["pred_classes"],  item["pred_bboxes"]
            scores           = item["scores"]
            
            all_classes.update(gt_cls)

            # Tandai prediksi yang sudah "dipakai" untuk matching GT
            for thr in iou_thresholds:
                used = set()
                # TP/FP
                for pc, pb, sc in zip(pd_cls, pd_box, scores):
                    if sc < conf_threshold:
                        continue
                    matched = False
                    for j, (tc, tb) in enumerate(zip(gt_cls, gt_box)):
                        if j in used: continue # GT sudah dipakai pred lain
                        if pc==tc and MetricsUtils.compute_iou(tb,pb)>=thr:
                            # Jika prediksi benar kelas dan cukup untuk overlap TP
                            pred_by_class[pc][thr].append((sc,1))
                            used.add(j)
                            matched = True
                            break
                    if not matched:
                        # Jika tidak match GT manapun pada TP
                        pred_by_class[pc][thr].append((sc,0))
                # FN (GT yang tidak terdeteksi)
                for j, tc in enumerate(gt_cls):
                    if j not in used:
                        fn_per_class[tc][thr] += 1

        # 2) Precision, Recall, F1, mAP50
        mean_p = mean_r = f1 = map50 = 0.0
        if 0.50 in iou_thresholds:
            precs, recs, ap50s = [], [], []
            for cls in all_classes:
                entries = pred_by_class[cls][0.50]
                if not entries:
                    continue
                entries.sort(key=lambda x:-x[0])
                tp_fp  = np.array([e[1] for e in entries])
                tp_cum = np.cumsum(tp_fp)
                fp_cum = np.cumsum(1-tp_fp)

                # sekarang benar: total GT = TP + FN
                tp_final = tp_cum[-1]
                fn_final = fn_per_class[cls][0.50]
                n_gt      = tp_final + fn_final
                if n_gt == 0:
                    continue

                precision_i = tp_final / (tp_final + fp_cum[-1] + 1e-6)
                recall_i    = tp_final / (n_gt + 1e-6)
                f1 =2 * precision_i * recall_i / (precision_i + recall_i + 1e-6)

                precs.append(precision_i)
                recs.append(recall_i)

                # curve untuk AP
                recall_curve = tp_cum / (n_gt + 1e-6)
                prec_curve   = tp_cum / (tp_cum + fp_cum + 1e-6)
                ap50 = MetricsUtils.compute_ap(recall_curve, prec_curve)
                ap50s.append(ap50)

                self.logger.info(
                        f"[UJI][Fold-{self.fold}][Kelas {cls} @IoU=0.50] "
                        f"Precision: {precision_i:.4f}, Recall: {recall_i:.4f}, F1: {f1:.4f}, AP@50: {ap50:.4f} \n"
                    )

            if precs:
                mean_p, mean_r = float(np.mean(precs)), float(np.mean(recs))
                f1_mean = 2*mean_p*mean_r/(mean_p+mean_r+1e-6)
            if ap50s:
                map50 = float(np.mean(ap50s))

            self.logger.info(f"\n[UJI] IoU=0.50: \nPrecision: {mean_p:.4f}, Recall: {mean_r:.4f}, \nF1-score: {f1_mean:.4f}, mAP50: {map50:.4f}")

        # 3) mAP50-95
        map5095 = 0.0
        aps_all = []
        for thr in iou_thresholds:
            aps = []
            for cls in all_classes:
                entries = pred_by_class[cls][thr]
                if not entries:
                    continue
                entries.sort(key=lambda x:-x[0])
                tp_fp  = np.array([e[1] for e in entries])
                tp_cum = np.cumsum(tp_fp)
                fp_cum = np.cumsum(1-tp_fp)

                tp_final = tp_cum[-1]
                fn_final = fn_per_class[cls][thr]
                n_gt      = tp_final + fn_final
                if n_gt == 0:
                    continue

                recall_curve = tp_cum / (n_gt + 1e-6)
                prec_curve   = tp_cum / (tp_cum + fp_cum + 1e-6)
                aps.append(MetricsUtils.compute_ap(recall_curve, prec_curve))
            if aps:
                aps_all.append(np.mean(aps))
        if aps_all:
            map5095 = float(np.mean(aps_all))
            self.logger.info(f"[UJI] mAP50-95: {map5095:.4f}")

        # 4) Fitness
        fitness = 0.1*mean_p + 0.1*mean_r + 0.1*map50 + 0.7*map5095
        self.logger.info(f"[UJI] Fitness: {fitness:.4f}")
        self.logger.info(f"\nEvaluasi selesai untuk Model {self.engine.upper()} pada data-{self.data.upper()} untuk Fold-{self.fold}. \n")

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
    def __init__(self, image_dir : Path, pred_dir : Path, anotasi_dir : Path, output_dir : Path, log_path :Path, data:str, model_type:str, size:str, engine, fold):
        self.image_folder = Path(image_dir) # path gambar set uji
        self.pred_folder = Path(pred_dir) # path hasil gambar prediksi
        self.annot_folder = Path(anotasi_dir) # path anotasi gambar set uji
        self.output_dir = Path(output_dir) # path hasil gambar komparasi
        self.data = data # jenis data yang digunakan
        self.model_type = model_type # tipe model yang digunakan
        self.size = size # ukuran model yang digunakan
        self.fold = fold # fold dari k-fold cross validation
        self.engine = engine
        os.makedirs(self.output_dir, exist_ok=True)  # Buat folder output jika belum ada
        self.matches= self._get_matching_files()

        # Subfolder untuk tiap mode
        self.comp_dir = self.output_dir / 'comparison'
        self.ovl_dir  = self.output_dir / 'overlay'
        self.comp_dir.mkdir(parents=True, exist_ok=True)
        self.ovl_dir.mkdir(parents=True, exist_ok=True)

        # Logger
        self.log_path = log_path / "log_visual.txt"
        self.logger = LoggerManager(log_file=self.log_path)

        # Mapping kelas ke nama dan warna
        self.class_names = {
            0: "bercak cokelat",
            1: "bercak cokelat tipis",
            2: "blas daun",
            3: "lepuh daun",
            4: "hawar daun bakteri",
            5: "sehat"
        }
        self.color_map = {
            0: (0, 0, 255),
            1: (0, 255, 255),
            2: (255, 255, 255),
            3: (0, 0, 128),
            4: (64, 224, 208),
            5: (255, 0, 255)
        }
        
    def _get_matching_files(self) -> List[Tuple[str, str, str]]:
        """
        Temukan triplet (gbr mentah, gbr prediksi, file anotasi GT).
        Prediksi diambil dari folder pred_folder dengan ekstensi gambar standar.
        """
        exts_img = ['*.png','*.jpg','*.jpeg','*.bmp','*.tif','*.tiff']
        support_pred_ext = ['.png','.jpg','.jpeg','.bmp','.tif','.tiff']

        # Cari semua gambar mentah
        raw_paths = []
        for ext in exts_img:
            raw_paths.extend(glob(os.path.join(self.image_folder, ext)))

        matches = []
        for raw in sorted(raw_paths):
            base = os.path.splitext(os.path.basename(raw))[0]
            # cari prediksi
            pred_path = None
            for ext in support_pred_ext:
                p = os.path.join(self.pred_folder, f"{base}{ext}")
                if os.path.exists(p):
                    pred_path = p
                    break
            if pred_path is None:
                continue
            # cari anotasi GT
            annot_path = os.path.join(self.annot_folder, f"{base}.txt")
            if not os.path.exists(annot_path):
                continue
            matches.append((raw, pred_path, annot_path))
        return matches

    def _load_annotations(self, path: str, img_shape: Tuple[int,int,int]) -> List[Tuple[int,int,int,int,int]]:
        """Load YOLO .txt annotations dan konversi ke koordinat pixel."""
        h, w, _ = img_shape
        boxes = []
        with open(path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cls, x_c, y_c, bw, bh = map(float, parts)
                x_min = int((x_c - bw/2) * w)
                y_min = int((y_c - bh/2) * h)
                x_max = int((x_c + bw/2) * w)
                y_max = int((y_c + bh/2) * h)
                boxes.append((int(cls), x_min, y_min, x_max, y_max))
        return boxes

    def _draw_bboxes(self, img: np.ndarray, boxes: List[Tuple[int,int,int,int,int]]) -> np.ndarray:
        """Gambar bounding box dan label pada citra RGB."""
        out = img.copy()
        for cls, x1, y1, x2, y2 in boxes:
            color = self.color_map.get(cls, (0,255,0))
            name = self.class_names.get(cls, 'unknown')
            cv2.rectangle(out, (x1,y1), (x2,y2), color, 2)
            cv2.putText(
                out, name, (x1, max(y1-5,5)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA
            )
        return out

    def visualize_comparison(self):
        """
        Tampilkan dan simpan perbandingan: kiri=GT, kanan=Prediksi (gambar sudah dirender).
        """
        for raw, pred_img, annot in self.matches:
            img = cv2.imread(raw)
            pred = cv2.imread(pred_img)
            if img is None or pred is None:
                self.logger.error(f"Error loading {raw} atau {pred_img}")
                continue
            # samakan ukuran
            if img.shape[:2] != pred.shape[:2]:
                pred = cv2.resize(pred, (img.shape[1], img.shape[0]))

            img_rgb  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pred_rgb = cv2.cvtColor(pred, cv2.COLOR_BGR2RGB)

            # gambar GT pada salinan
            gt_boxes = self._load_annotations(annot, img.shape)
            img_gt = self._draw_bboxes(img_rgb, gt_boxes)

            fig, ax = plt.subplots(1,2,figsize=(16,8))
            ax[0].imshow(img_gt);   ax[0].set_title('Data Awal + GT'); ax[0].axis('off')
            ax[1].imshow(pred_rgb); ax[1].set_title('Hasil Prediksi'); ax[1].axis('off')

            base = os.path.splitext(os.path.basename(raw))[0]
            fname = self.comp_dir / f"comp_{self.data}_{self.model_type}_{self.engine}_{self.size}_{self.fold}_{base}.png"
            out = os.path.join(self.output_dir, fname)
            plt.tight_layout()
            plt.savefig(out, dpi=300, bbox_inches='tight')
            plt.close(fig)
            self.logger.info(f"\nFile Gambar Comparison tersimpan di path: {out}")

    def visualize_overlay(self, alpha: float = 0.5):
        """
        Tampilkan dan simpan overlay GT di atas hasil prediksi.
        """
        for raw, pred_img, annot in self.matches:
            img = cv2.imread(raw)
            pred = cv2.imread(pred_img)
            if img is None or pred is None:
                self.logger.error(f"Error loading {raw} atau {pred_img}")
                continue
            if img.shape[:2] != pred.shape[:2]:
                pred = cv2.resize(pred, (img.shape[1], img.shape[0]))

            img_rgb  = cv2.cvtColor(img,  cv2.COLOR_BGR2RGB)
            pred_rgb = cv2.cvtColor(pred, cv2.COLOR_BGR2RGB)

            gt_boxes = self._load_annotations(annot, img.shape)
            gt_img   = self._draw_bboxes(img_rgb, gt_boxes)

            # overlay
            overlay = cv2.addWeighted(gt_img, alpha, pred_rgb, 1-alpha, 0)

            fig, ax = plt.subplots(figsize=(8,8))
            ax.imshow(overlay)
            ax.set_title(f"Overlay GT vs Pred - {os.path.basename(raw)}")
            ax.axis('off')

            base = os.path.splitext(os.path.basename(raw))[0]
            fname = self.ovl_dir / f"overlay_{self.data}_{self.model_type}_{self.engine}_{self.size}_{self.fold}_{base}.png"
            out = os.path.join(self.output_dir, fname)
            plt.savefig(out, dpi=300, bbox_inches='tight')
            plt.close(fig)
            self.logger.info(f"\nFile Overlay Tersimpan di Path: {out}")

MAIN_PATH = Path("D:/Riset Skripsi/script riset/deteksi_citra/")
DATASET_PATH = Path(f"D:/Riset Skripsi/dataset_skripsi/")
OLAH_CONFIG = {
    'n_folds':5,
    'conf_score':0.35,
    'iou_map50': 0.5,
    'iou_map5095' : [round(x,2) for x in np.arange(0.5, 1.0, 0.05)]
}

def running_det_uji(detector: PredictionDetector, olah_config: dict):
    logger       = detector.logger
    engine_label = detector.engine.lower()

    # 1) Tentukan nama & path file matched raw & filtered
    matched_name_raw  = f"{engine_label}_fold_{detector.fold}_uji_matched_raw.json"
    matched_name_filt = f"{engine_label}_fold_{detector.fold}_uji_matched_filtered.json"
    path_raw  = detector.json_matched / matched_name_raw
    path_filt = detector.json_matched / matched_name_filt

    # 2) Generate matched JSON (raw + filtered) jika belum ada
    if not path_raw.exists() or not path_filt.exists():
        logger.info(f"[INFO] Generating matched JSON for {engine_label.upper()} (raw & filtered)…")
        raw_path, filt_path = detector.generate_matched_json(
            score_threshold=olah_config["conf_score"],
            iou_threshold=olah_config["iou_map50"]
        )
    else:
        logger.info(f"[INFO] Matched JSON for {engine_label.upper()} found. Skipping generation.")
        raw_path, filt_path = path_raw, path_filt

    # 3) Generate confusion matrix (filtered) jika belum dibuat
    if input(f"\nApakah confusion matrix untuk {engine_label.upper()} model sudah dibuat? (y/n): ").strip().lower() == 'n':
        detector.generate_cm(
            filtered_json_path=filt_path,
            iou_threshold=olah_config["iou_map50"]
        )
    else:
        logger.info(f"[INFO] Confusion Matrix {engine_label.upper()} sudah tersedia. Lanjut evaluasi...\n")

    # 4) Logging metrik evaluasi (gunakan raw JSON untuk mAP/PR-curve)
    logger.info(f"\n[LOGGING] Metrik Evaluasi Deteksi ({engine_label.upper()})")
    detector.log_metrics_uji(
        path_matched=raw_path,
        conf_threshold=olah_config["conf_score"],
        iou_thresholds=olah_config["iou_map5095"]
    )

    return raw_path, path_filt

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
        print(f"\n>>>>> Memproses fold-{fold} <<<<<< \n")

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
        labels_path = DATASET_PATH / f"dataset_{data}" / "fold" / f"fold_{fold}" / "val" / "labels"
        labels_uji_path = DATASET_PATH / f"dataset_{data}" / "fold" / "test"

        # Path Setup Visualisasi
        gambar_awal_path = labels_uji_path / "images" # path data gambar uji awal
        label_awal_path = labels_uji_path / "labels" # path data anotasi uji awal
        gambar_hasil_path = hasil_uji_path # path data gambar uji hasil yolo
        gambar_trt_hasil_path = trt_uji_path # path data gambar uji hasil trt
        output_path = MAIN_PATH / data / "gambar" # path output hasil visualisasi

        log_path.mkdir(parents=True, exist_ok=True)
        proses_path.mkdir(parents=True, exist_ok=True)

        logger = LoggerManager(log_file=log_root_path)
        
        try:
            # === CSV Remake ===
            if input("Buat file CSV untuk utilisasi? (y/n): ").strip().lower() == 'y':
                logger.info("\n[CSV] Memulai pemindahan data CSV... \n")
                csv_remaker = CSVRemaker(result_path, proses_path, log_path, fold, data, model_type, size)
                csv_remaker.save_fold_metrics()
                csv_remaker.save_combined_metrics()
                csv_remaker.log_validation_metrics()
                logger.info("\n[CSV] Selesai menyimpan file CSV utilisasi.")

            # === Plotting Grafik ===
            if input("Lakukan plotting grafik? (y/n): ").strip().lower() == 'y':
                logger.info("[PLOT] Memulai proses plotting...")
                plotter = CSVPlotter(proses_path, log_path, fold, data, model_type, size)
                plotter.plot_all(target_folds=fold, data=data)
                logger.info("[PLOT] Grafik selesai dibuat.")

            # === Confusion Matrix & Metrik Validasi ===
            if input("\nApakah kamu ingin buat confusion matrix dari data validasi (post-training)? (y/n): ").strip().lower() == 'y':
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

                # 1) Generate kedua JSON (raw + filtered), atau load jika sudah ada
                raw_json_path  = detector_val.output_json / f"fold_{fold}_matched_raw.json"
                filt_json_path = detector_val.output_json / f"fold_{fold}_matched_filtered.json"

                if not raw_json_path.exists() or not filt_json_path.exists():
                    logger.info("[VALIDASI] Matched JSON belum lengkap. Membuat ulang raw + filtered JSON…")
                    raw_json_path, filt_json_path = detector_val.generate_matched_json(
                        iou_threshold=OLAH_CONFIG["iou_map50"],
                        score_threshold=OLAH_CONFIG["conf_score"]
                    )
                else:
                    logger.info("[VALIDASI] Matched JSON (raw + filtered) sudah ada. Lewati pembuatan ulang.")

                # 2) Confusion matrix pakai filtered JSON
                detector_val.generate_confusion_matrix(
                    filt_json_path=filt_json_path,
                    iou_threshold=OLAH_CONFIG["iou_map50"]
                )
                # 3) Logging Metrik Deteksi Data Validasi dari Confusion Matrix (post-training)
                detector_val.log_metrics(
                    filt_json_path=filt_json_path,
                    iou_thresholds=OLAH_CONFIG["iou_map5095"],
                    score_threshold=OLAH_CONFIG["conf_score"]
                )

            # === Confusion Matrix Uji (YOLO, TRT, atau Semua) ===
            if input("\nApakah ingin membuat confusion matrix data uji? (y/n): ").strip().lower() == 'y':
                engine_mode = input("Evaluasi hasil dari YOLO, TRT, atau keduanya? (yolo/trt/all): ").strip().lower()
                if engine_mode not in {"yolo", "trt", "all"}:
                    raise ValueError("Pilihan engine harus: yolo, trt, atau all")

                engine_paths = {
                    "yolo": hasil_uji_path / "labels",
                    "trt": trt_uji_path  / "labels"
                }

                for engine in ["yolo", "trt"]:
                    if engine_mode in {engine, "all"}:
                        detector = PredictionDetector(
                            label_hasil=engine_paths[engine],
                            label_awal=label_awal_path,
                            olah_path=proses_path,
                            log_path=log_path,
                            size=size,
                            model_type=model_type,
                            data=data,
                            fold=fold,
                            engine=engine
                        )
                        running_det_uji(detector, olah_config=OLAH_CONFIG)

            if input("\nApakah ingin menampilkan visual perbandingan data awal dan data hasil? (y/n) ").strip().lower() == 'y':
                engines = []
                if engine_mode in {'yolo', 'all'}:
                    engines.append(('yolo', gambar_hasil_path))
                if engine_mode in {'trt', 'all'}:
                    engines.append(('trt', gambar_trt_hasil_path))

                for eng, pred_folder in engines:
                    logger.info(f"Menjalankan visualisasi perbandingan untuk engine {eng.upper()} ...")
                    viz = VisualisasiDatasetPred(
                        image_dir= gambar_awal_path,
                        pred_dir= pred_folder,
                        anotasi_dir= label_awal_path,
                        output_dir= output_path,
                        log_path=log_path,
                        data= data,
                        model_type=model_type,
                        size=size,
                        engine=eng,
                        fold=fold
                    )

                    viz.visualize_comparison()
                    viz.visualize_overlay()

        except Exception as e:
            logger.error(f"\nError processing fold {fold}:")
            logger.error(f"Error details: {str(e)}")
            logger.error(f"Fold {fold} failed: {str(e)}\n")
            continue

if __name__ == "__main__":
    main()
