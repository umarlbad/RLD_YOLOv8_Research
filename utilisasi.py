import pandas as pd, numpy as np
import matplotlib.pyplot as plt
import json, torch, logging, os, cv2, stat, time, csv, seaborn as sns
from pathlib import Path
from collections import defaultdict
from sklearn.metrics import confusion_matrix
from typing import List, Dict, Optional, Tuple
from glob import glob


class LoggerManager:
    def __init__(self, log_file):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        if not self.logger.handlers:
            try:
                file_handler = logging.FileHandler(log_file, mode='w')
                file_handler.setLevel(logging.INFO)
                file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
                file_handler.setFormatter(file_formatter)
                
                console_handler = logging.StreamHandler()
                console_handler.setLevel(logging.INFO)
                console_formatter = logging.Formatter('%(levelname)s: %(message)s')
                console_handler.setFormatter(console_formatter)
                
                self.logger.addHandler(file_handler)
                self.logger.addHandler(console_handler)
            except Exception as e:
                print(f"Error setting up logger: {e}")
                raise

    def info(self, message):
        self.logger.info(message)
    
    def error(self, message):
        self.logger.error(message)
    
    def debug(self, message):
        self.logger.debug(message)

class CSVRemaker:
    def __init__(self, csv_path, olah_path, num_folds, data, model_type, size):
        self.csv_path = csv_path
        self.num_folds = num_folds
        self.data = data
        self.model_type = model_type
        self.size = size
        self.output_path = olah_path / self.model_type / f'ukuran_{self.size}' / "hasil_csv"
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.log_path = self.output_path / "log_csv.txt"
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
            self.logger.debug(f"Data berhasil dimuat dari {csv_path}")
            
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
                        self.logger.debug(f"File CSV Fold-{fold} disimpan di path: {output_file}")
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
                        self.logger.debug(f"Sukses menyimpan file Fold-{fold} yang disimpan di path: {fold_file}") 
                        try:
                            df = pd.read_csv(fold_file)
                            if "epoch" in df.columns:
                                if combined_dfs[col].empty:
                                    combined_dfs[col] = df.copy()
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
                        self.logger.debug(f"File gabungan disimpan: {combined_file}")
                    except Exception as e:
                        self.logger.error(f"Gagal menyimpan file gabungan {col}: {e}")
        except Exception as e:
            self.logger.error(f"Gagal membuat direktori output gabungan: {e}")

class CSVPlotter:
    def __init__(self, olah_path, num_folds, data, model_type, size):
        self.num_folds = num_folds
        self.data = data
        self.model_type = model_type
        self.size = size
        self.file_path = olah_path / self.model_type / f'ukuran_{self.size}'/ "hasil_csv"
        self.output_path  = olah_path /  self.model_type / f'ukuran_{self.size}'/ "grafik"
        self.log_path = olah_path / "hasil_grafik" / "log_plotting.txt"
        self.logger = LoggerManager(log_file=self.log_path)
        self.results_comp = self.output_path.joinpath("comparison/")
        self.results_indt = self.output_path.joinpath(f"independent/fold_{self.num_folds}/")
        self.file_path.mkdir(parents=True, exist_ok=True)
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.results_comp.mkdir(parents=True, exist_ok=True)
        self.results_indt.mkdir(parents=True, exist_ok=True)

        self.variables = [
            "train/box_loss", "train/cls_loss", "train/dfl_loss", 
            "val/box_loss", "val/cls_loss", "val/dfl_loss", 
            "metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)", "metrics/mAP50-95(B)"
        ]
        self.var = {var: [] for var in self.variables}
        self.var["epoch"] = None
    
    def ensure_directory_exists(self, directory_path):
        """Memastikan direktori ada dan memiliki izin yang tepat."""
        try:
            # Coba buat direktori dengan izin penuh
            directory_path.mkdir(parents=True, exist_ok=True)
            
            # Atur izin penuh untuk direktori (Windows)
            if os.name == 'nt':  # Windows
                os.chmod(directory_path, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
                
            return True
        except PermissionError:
            self.logger.error(f"Izin ditolak saat membuat {directory_path}, menggunakan lokasi alternatif.")
            alt_path = Path(os.path.expanduser("~")) / "Documents" / f"temp_output_{int(time.time())}"
            alt_path.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Dibuat direktori alternatif: {alt_path}")
            self.output_path = alt_path
            return False
        except Exception as e:
            self.logger.error(f"Error membuat direktori {directory_path}: {e}")
            raise

    def collect_fold_data(self):
        fold_results = {}

        for fold in range(1, self.num_folds+1):
            fold_data = {} # Dict untuk menyimpan data per fold
            
            # Loop untuk setiap variabel dalam self.variables
            for var in self.variables:
                # Ubah "/" menjadi "_" agar sesuai dengan nama file
                var_filename = var.replace("/", "_")
                fold_path = self.file_path / f"fold_{fold}" / f"fold_{fold}_{var_filename}.csv"
                try:
                    if not fold_path.exists():
                        self.logger.error(f"File {fold_path} tidak ditemukan.")
                        fold_data[var] = None
                        continue

                    # Membaca CSV
                    df = pd.read_csv(fold_path)

                    # Pastikan kolom epoch ada hanya saat membaca file pertama kali
                    if "epoch" not in df.columns:
                        self.logger.error(f"Fold {fold}: File CSV {fold_path.name} tidak memiliki kolom 'epoch'.")
                        fold_data[var] = None
                        continue

                    fold_data["epoch"] = df["epoch"].tolist()

                    # Simpan data variabel
                    if var in df.columns:
                        fold_data[var] = df[var].tolist()
                    else:
                        fold_data[var] = None
                        self.logger.error(f"Fold {fold}: Variabel {var} tidak ditemukan dalam CSV.")
                except Exception as e:
                    self.logger.error(f"Kesalahan saat memproses {fold_path}: {e}")
                    fold_data[var] = None

            # Simpan data fold ini ke dictionary utama
            fold_results[fold] = fold_data
        
        return fold_results # mengembalikan hasil sebagai dict per fold
    
    def collect_combined_data(self):
        """
        Fungsi ini membaca file CSV gabungan dari berbagai metrik/loss dalam folder 'combined/' 
        dan menyimpannya dalam struktur data self.var.
        """
        combined_results = {}
        combined_results["epoch"] = None
        epoch_loaded = False # pastikan dideklarasikan sebelum digunakan

        combined_folder = self.file_path / "combined"

        if not combined_folder.exists():
            self.logger.error(f"Direktori {combined_folder} tidak ditemukan.")
            return {}
        
        # Loop untuk setiap variabel dalam self.variables
        for var in self.variables:
            try:
                # Ubah "/" menjadi "_" agar sesuai dengan nama file
                var_filename = var.replace("/", "_")
                file_path = combined_folder / f"combined_{var_filename}.csv"

                if not file_path.exists():
                    self.logger.error(f"File {file_path} tidak ditemukan.")
                    continue

                # Membaca CSV
                df = pd.read_csv(file_path)

                # Pastikan kolom pertama adalah 'epoch'
                if df.columns[0].lower() != "epoch":
                    self.logger.error(f"File {file_path.name} tidak memiliki kolom 'epoch' sebagai kolom pertama!")
                    continue

                # Simpan epoch hanya sekali
                if not epoch_loaded:
                    combined_results["epoch"] = df["epoch"].tolist()
                    epoch_loaded = True  # Tandai bahwa epoch telah berhasil dimuat

                # Ambil semua kolom fold (selain 'epoch')
                fold_columns = [col for col in df.columns if col != "epoch"]

                if not fold_columns:
                    self.logger.error(f"File {file_path.name} tidak memiliki data Fold yang valid. Melewati variabel {var}.")
                    continue

                # Simpan data variabel
                combined_results[var] = [df[fold_col].tolist() for fold_col in fold_columns]
                
            except Exception as e:
                self.logger.error(f"Kesalahan saat memproses {var}: {e}")
                combined_results[var] = []

        if not epoch_loaded:
            self.logger.error("Gagal memuat epoch! Pastikan file CSV memiliki kolom 'epoch'.")
            return {}

        self.var = combined_results  # Simpan hasil ke self.var
        self.logger.info("Data dari file CSV gabungan berhasil dikumpulkan.")
        
        return combined_results  # Pastikan fungsi mengembalikan dictionary yang valid

    def smooth_curve(self, values, weight=0.9):
        """Melakukan smoothing pada kurva untuk mengurangi noise."""
        smoothed = []
        last = values[0]
        for point in values:
            smoothed_val = last * weight + (1 - weight) * point
            smoothed.append(smoothed_val)
            last = smoothed_val
        return smoothed
    
    def plot_combined(self, collected_combined):
        """
        Fungsi untuk membaca data dari file CSV gabungan menggunakan `collect_combined_data()`
        dan melakukan plotting untuk setiap Fold dengan markers & colors yang berbeda.
        """
        # Ambil data terbaru dari file CSV gabungan
        collected_combined = self.collect_combined_data()

        # Pastikan collected_combined adalah dictionary yang valid
        if not isinstance(collected_combined, dict):
            self.logger.error("Data collected_combined tidak valid. Pastikan berupa dictionary.")
            return

        # Pastikan epoch tersedia
        if "epoch" not in collected_combined or not collected_combined["epoch"]:
            self.logger.error("Data epoch tidak ditemukan dalam collected_combined.")
            return

        epochs = collected_combined["epoch"]  # Ambil daftar epoch

        # Definisi markers dan colors
        markers = ['o', 's', 'D', '^', 'v']  # Marker untuk variasi visual tiap fold
        colors = ['b', 'g', 'r', 'c', 'm']  # Warna berbeda untuk tiap fold

        for var in self.variables:
            if var not in collected_combined or not collected_combined[var]:
                self.logger.error(f"Data {var} tidak tersedia atau kosong, dilewati.")
                continue

            # Pastikan data untuk variabel ini berbentuk DataFrame
            data_dict = {"epoch": epochs}
            fold_columns = [f"Fold_{i+1}" for i in range(len(collected_combined[var]))]

            for i, fold_data in enumerate(collected_combined[var]):
                data_dict[f"Fold_{i+1}"] = fold_data

            df = pd.DataFrame(data_dict)

            # Plot grafik untuk setiap Fold dengan marker dan warna yang berbeda
            plt.figure(figsize=(8, 6))
            for i, col in enumerate(fold_columns):
                plt.plot(
                    df["epoch"], df[col], 
                    marker=markers[i % len(markers)],  # Rotasi marker jika fold > jumlah marker
                    color=colors[i % len(colors)],    # Rotasi warna jika fold > jumlah warna
                    linestyle='-', label=col
                )

            # Konfigurasi plot
            plt.xlabel("Epoch")
            plt.ylabel(var)
            plt.title(f"Training Loss / Metrics per Fold - {var}")
            plt.legend()
            plt.grid(True)

            # Simpan hasil plot
            save_path = Path(self.results_comp) / f"Cbd_{var.replace('/', '_')}.png"
            plt.savefig(save_path, dpi=300)
            plt.close()

        self.logger.info("Comparison graphs have been saved successfully.")
    
    def plot_variable(self, var, collect_fold):
        # Periksa apakah data tersedia
        fold_data = collect_fold # ambil data dari function collect_fold_data

        for fold , fold_values in fold_data.items():  # Loop untuk setiap fold
            if not isinstance(fold_values, dict):
                self.logger.error(f"Fold {fold}: Data fold bukan dictionary, mungkin format salah. Dilewati.")
                continue
            
            if "epoch" not in self.var or var not in self.var:
                self.logger.error(f"Data {var} atau epoch tidak ditemukan dalam fold_results.")
                return
            
            epochs = fold_values["epoch"]  # Ambil semua epoch
            values = fold_values[var] # Ambil nilai variabel untuk fold ini

            # Pastikan values memiliki data yang valid
            if not isinstance(values, list) or all(v is None for v in values):
                self.logger.error(f"Fold {fold}: Data {var} tidak valid atau kosong, dilewati.")
                continue
                
            # Pastikan panjang epoch dan values cocok
            if len(epochs) != len(values):
                self.logger.error(f"Skipping Fold {fold}: Mismatch (epochs: {len(epochs)}, values: {len(values)})")
                continue

            # Buat figure baru untuk setiap fold agar tidak tumpang tindih
            plt.figure(figsize=(10, 5))
            plt.plot(epochs, values, marker='o', linestyle='-', label=var)

            plt.xlabel("Epoch")
            plt.ylabel(var)
            plt.title(f"Graph of {var} - Fold {fold}")
            plt.legend()
            plt.grid(True)

            # Simpan hasil plot dengan nama berdasarkan fold
            save_path = Path(self.results_indt) / f"{fold}_fold_Ind_{var.replace('/', '_')}.png"
            save_path.parent.mkdir(parents=True, exist_ok=True)  # Pastikan folder ada
            plt.savefig(save_path, dpi=300)
            plt.close()  # Tutup plot untuk fold ini agar tidak menumpuk

            self.logger.debug(f"Plot {var} untuk fold {fold} telah disimpan di {save_path}.")

    def plot_all(self):
        collect_fold = self.collect_fold_data()
        collect_comb = self.collect_combined_data()
        for var in self.variables:
            self.plot_variable(var, collect_fold)
        self.plot_combined(collect_comb)
        self.logger.debug("All Graphs have been saved successfully.")

class PredictionClassificator:
    # class yang berisi command untuk membuat confusion matrix biner (klasifikasi)
    def __init__(self, predictions_json, label_dir, olah_path, fold, data, model_type, size):
        self.predictions_json = predictions_json / "train" / "predictions.json"
        self.label_dir = label_dir
        self.fold = fold
        self.data = data # variabel untuk menampilkan / mendefinisikan jenis data
        self.model_type = model_type
        self.size = size
        self.output_json = olah_path /self.model_type / f'ukuran_{self.size}'/ "json_final" # direktori json final untuk menjadi bahan confusion matrix klasifikasi
        self.output_gt = olah_path /self.model_type / f'ukuran_{self.size}'/ "json_gt" # direktori untuk membuat file json ground truth dari file labels
        self.output_pred = olah_path /self.model_type / f'ukuran_{self.size}'/ "json_pred" # direktori untuk membuat file json prediksi yang difiltrasi dari file mentah predictions.json 
        self.output_cm = olah_path /self.model_type / f'ukuran_{self.size}'/ "cm" # direktori untuk membuat confusion matrix klasifikasi
        # Pastikan folder json_gt dan cm ada
        self.output_gt.mkdir(parents=True, exist_ok=True)
        self.output_cm.mkdir(parents=True, exist_ok=True)
        self.output_pred.mkdir(parents=True, exist_ok=True)
        self.output_json.mkdir(parents=True, exist_ok=True)
        self.matched_data = []
        # Dictionary mapping kategori prediksi ke nilai yang benar
        self.valid_labels = {
            0 : "bercak cokelat",
            1 : "bercak cokelat tipis",
            2 : "blas daun",
            3 : "lepuh daun",
            4 : "hawar daun bakteri",
            5 : "sehat"
        }
        self.label_mapping = {
            1: 0,  # bercak cokelat
            2: 1,  # bercak cokelat tipis
            3: 2,  # blas daun
            4: 3,  # lepuh daun
            5: 4,  # hawar daun bakteri
            6: 5   # sehat
        }
        self.logger = LoggerManager(log_file=self.output_json / "log_cm.txt")

    def load_gt_labels(self):
        ground_truth = {}

        if not os.path.exists(self.label_dir):
            self.logger.error(f"Direktori label {self.label_dir} tidak ditemukan.")
            return ground_truth

        for label_file in os.listdir(self.label_dir):
            try:
                image_id = os.path.splitext(label_file)[0]
                label_path = os.path.join(self.label_dir, label_file)

                with open(label_path, "r") as f:
                    gt_bboxes = []
                    gt_classes = []
                    for line in f.readlines():
                        parts = line.strip().split()
                        class_id = int(parts[0])
                        bbox = list(map(float, parts[1:5]))
                        gt_classes.append(class_id)
                        gt_bboxes.append(bbox)

                    ground_truth[image_id] = {
                        "true_classes": gt_classes,
                        "true_bboxes": gt_bboxes
                    }
            except Exception as e:
                self.logger.error(f"Kesalahan saat memproses {label_file}: {e}")
                continue

        # Simpan ground truth ke file JSON dengan nama berdasarkan fold
        gt_json_path = self.output_gt / f"fold_{self.fold}_labels.json"
        try:
            with open(gt_json_path, "w") as f:
                json.dump(ground_truth, f, indent=4)
            self.logger.info(f"Label ground truth berhasil disimpan di {gt_json_path}")
        except Exception as e:
            self.logger.error(f"Gagal menyimpan ground truth ke JSON: {e}")
        
        return ground_truth

    def filter_predictions(self):

        if not self.predictions_json.exists():
            print(f"File prediksi {self.predictions_json} tidak ditemukan.")
            return
        
        with open(self.predictions_json, "r") as f:
            data = json.load(f)

        unique_data = {}
        for item in data:
            image_id = item.get("image_id")
            pred_label = self.label_mapping.get(item.get("category_id"), item.get("category_id"))  # Mapping label
            bbox = item.get("bbox", [])

            if pred_label is None:
                self.logger.error(f"Data tanpa category_id ditemukan: {item}")  # Tambahkan logging
                continue
            
            if pred_label in self.valid_labels:
                if image_id not in unique_data or len(bbox) > len(unique_data[image_id]["bbox"]):
                    unique_data[image_id] = {
                        "image_id": image_id,
                        "pred_label": pred_label,
                        "bbox": bbox
                    }
        
        filtered_json_path = self.output_pred / f"fold_{self.fold}_filtered.json"
        with open(filtered_json_path, 'w') as f:
            json.dump(list(unique_data.values()), f, indent=4)
        self.logger.info(f"Filtered data saved to {filtered_json_path}")

    def match_pred_with_gt(self):
        filtered_json_path = self.output_pred /  f"fold_{self.fold}_filtered.json"
        
        if not filtered_json_path.exists():
            print(f"File prediksi yang telah difilter tidak ditemukan: {filtered_json_path}")
            return

        try:
            with open(filtered_json_path, "r") as f:
                predictions = json.load(f)
                
            gt_json_path = self.output_gt / f"fold_{self.fold}_labels.json"
            if gt_json_path.exists():
                with open(gt_json_path, "r") as f:
                    ground_truth = json.load(f)
                self.logger.info(f"Ground truth dimuat dari {gt_json_path}")
            else:
                ground_truth = self.load_gt_labels()
            
            for pred in predictions:
                image_id = str(pred["image_id"])
                pred_class = pred["pred_label"]  # Pastikan sesuai dengan format dari filter_predictions()
                
                if image_id in ground_truth:
                    true_labels = list(set(ground_truth[image_id]["true_classes"]))  # Hapus duplikasi
                    
                    if not any(d["image_id"] == image_id for d in self.matched_data):  
                        self.matched_data.append({
                            "image_id": image_id,
                            "true_label": true_labels,
                            "pred_label": [pred_class],
                        })
            
            self.save_matched_predictions()
        except Exception as e:
            self.logger.error(f"Kesalahan saat matching prediksi: {e}")
    
    def save_matched_predictions(self):
        output_json_path = self.output_json / f"fold_{self.fold}_matched.json"
        
        try:
            self.output_json.mkdir(parents=True, exist_ok=True)
            with open(output_json_path, "w") as f:
                json.dump(self.matched_data, f, indent=4)
            self.logger.info(f"Hasil pencocokan disimpan di {output_json_path}")
            
            # Setelah menyimpan, buat confusion matrix
            self.compute_confusion_matrix(output_json_path)
        except Exception as e:
            self.logger.error(f"Gagal menyimpan hasil ke {output_json_path}: {e}")
    
    def compute_confusion_matrix(self, matched_json_path):
        """ Membuat dan menyimpan confusion matrix berdasarkan data yang telah dicocokkan. """
        try:
            with open(matched_json_path, "r") as f:
                matched_data = json.load(f)
            
            # Pastikan true_labels dan pred_labels diambil dengan format yang benar
            true_labels = []
            pred_labels = []

            for item in matched_data:
                # Cek apakah "true_label" ada dan dalam bentuk yang benar
                if isinstance(item["true_label"], list) and len(item["true_label"]) > 0:
                    true_labels.append(item["true_label"][0])  # Ambil elemen pertama jika berbentuk list
                elif isinstance(item["true_label"], int):  # Jika bukan list, langsung tambahkan
                    true_labels.append(item["true_label"])
                else:
                    self.logger.error(f"Format tidak valid di true_label: {item['true_label']}")

                # Cek apakah "pred_label" ada dan dalam bentuk yang benar
                if isinstance(item["pred_label"], list) and len(item["pred_label"]) > 0:
                    pred_labels.append(item["pred_label"][0])  # Ambil elemen pertama jika berbentuk list
                elif isinstance(item["pred_label"], int):
                    pred_labels.append(item["pred_label"])
                else:
                    self.logger.error(f"Format tidak valid di pred_label: {item['pred_label']}")

            # Pastikan panjang kedua label sama sebelum membuat confusion matrix
            if len(true_labels) == 0 or len(pred_labels) == 0:
                self.logger.error("Gagal membuat confusion matrix: Tidak ada data label yang valid.")
                return

            if len(true_labels) != len(pred_labels):
                self.logger.error(f"Gagal membuat confusion matrix: Ukuran true_labels ({len(true_labels)}) dan pred_labels ({len(pred_labels)}) tidak sama.")
                return
            cm = confusion_matrix(true_labels, pred_labels, labels=list(self.valid_labels.keys()))
            cm_labels = [self.valid_labels[i] for i in self.valid_labels.keys()]
            cm_path = self.output_cm / f"fold_{self.fold}_cm.png"
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Oranges", xticklabels=cm_labels, yticklabels=cm_labels)
            plt.xlabel("Predicted Label")
            plt.ylabel("True Label")
            plt.title(f"Confusion Matrix Klasifikasi {self.data}- Fold {self.fold}")
            plt.savefig(cm_path)
            plt.close()
            
            self.logger.info(f"Confusion matrix disimpan di {cm_path}")

        except Exception as e:
            self.logger.error(f"Kesalahan saat membuat confusion matrix: {e}")

class PredictionDetector:
    # Class yang berisi command untuk membuat confusion matrix multi-kelas (deteksi objek)
    def __init__(self, predictions_json, label_dir, olah_path, fold, data, model_type, size):
        self.predictions_json = predictions_json / "train" / "predictions.json"
        self.label_dir = label_dir
        self.fold = fold
        self.data = data # variabel untuk menampilkan / mendefinisikan jenis data
        self.model_type = model_type
        self.size = size
        self.output_path  = olah_path /self.model_type / f'ukuran_{self.size}'/ "hasil_csv"
        self.output_json = olah_path /self.model_type / f'ukuran_{self.size}' / "json_final" # direktori json final untuk menjadi bahan confusion matrix deteksi 
        self.output_gt = olah_path /self.model_type / f'ukuran_{self.size}' / "json_gt_deteksi" # direktori untuk membuat file json ground truth dari file labels (untuk deteksi)
        self.output_pred = olah_path /self.model_type / f'ukuran_{self.size}' / "json_pred" # direktori untuk membuat file json prediksi yang difiltrasi dari file mentah predictions.json ( ini juga tidak perlu)
        self.output_cm_d = olah_path /self.model_type / f'ukuran_{self.size}' / "cm_deteksi" # direktori untuk membuat confusion matrix multi-kelas (deteksi)
        # Pastikan folder json_gt dan cm ada
        self.output_gt.mkdir(parents=True, exist_ok=True)
        self.output_cm_d.mkdir(parents=True, exist_ok=True)
        self.output_pred.mkdir(parents=True, exist_ok=True)
        self.output_json.mkdir(parents=True, exist_ok=True)
        self.matched_data = []
        # Dictionary mapping kategori prediksi ke nilai yang benar
        self.valid_labels = {
            0 : "bercak cokelat",
            1 : "bercak cokelat tipis",
            2 : "blas daun",
            3 : "lepuh daun",
            4 : "hawar daun bakteri",
            5 : "sehat"
        }
        self.label_mapping = {
            1: 0,  # bercak cokelat
            2: 1,  # bercak cokelat tipis
            3: 2,  # blas daun
            4: 3,  # lepuh daun
            5: 4,  # hawar daun bakteri
            6: 5   # sehat
        }
        self.logger = LoggerManager(log_file=self.output_json / "log_cm_deteksi.txt")

    def load_gt_labels(self):
        ground_truth = {}

        if not os.path.exists(self.label_dir):
            self.logger.error(f"Direktori label {self.label_dir} tidak ditemukan.")
            return ground_truth

        for label_file in os.listdir(self.label_dir):
            try:
                image_id = os.path.splitext(label_file)[0]
                label_path = os.path.join(self.label_dir, label_file)

                with open(label_path, "r") as f:
                    gt_bboxes = []
                    gt_classes = []
                    for line in f.readlines():
                        parts = line.strip().split()
                        class_id = int(parts[0])
                        bbox = list(map(float, parts[1:5]))
                        gt_classes.append(class_id)
                        gt_bboxes.append(bbox)

                    ground_truth[image_id] = {
                        "true_classes": gt_classes,
                        "true_bboxes": gt_bboxes
                    }
            except Exception as e:
                self.logger.error(f"Kesalahan saat memproses {label_file}: {e}")
                continue

        # Simpan ground truth ke file JSON dengan nama berdasarkan fold
        gt_json_path = self.output_gt / f"fold_{self.fold}_labels.json"
        try:
            with open(gt_json_path, "w") as f:
                json.dump(ground_truth, f, indent=4)
            self.logger.info(f"Label ground truth berhasil disimpan di {gt_json_path}")
        except Exception as e:
            self.logger.error(f"Gagal menyimpan ground truth ke JSON: {e}")
        
        return ground_truth
    
    def compute_iou(self,box1, box2):
        """
        Menghitung IoU (Intersection over Union) antara dua bbox format [x, y, w, h]
        """
        x1_min = box1[0]
        y1_min = box1[1]
        x1_max = box1[0] + box1[2]
        y1_max = box1[1] + box1[3]

        x2_min = box2[0]
        y2_min = box2[1]
        x2_max = box2[0] + box2[2]
        y2_max = box2[1] + box2[3]

        inter_xmin = max(x1_min, x2_min)
        inter_ymin = max(y1_min, y2_min)
        inter_xmax = min(x1_max, x2_max)
        inter_ymax = min(y1_max, y2_max)

        inter_area = max(0, inter_xmax - inter_xmin) * max(0, inter_ymax - inter_ymin)
        area1 = box1[2] * box1[3]
        area2 = box2[2] * box2[3]
        union_area = area1 + area2 - inter_area

        return inter_area / union_area if union_area > 0 else 0.0
    
    def log_detection_statistics(self, matched_data, title=""):
        """
        Logging jumlah TP, FP, FN per kelas berdasarkan matched_data,
        dengan nama kelas dari self.valid_labels.
        Hasil juga disimpan ke file CSV.
        """

        stats = {
            "TP": defaultdict(int),
            "FP": defaultdict(int),
            "FN": defaultdict(int),
        }

        for item in matched_data:
            true = item.get("true_label")
            pred = item.get("pred_label")

            if true is not None and pred is not None:
                if true == pred:
                    stats["TP"][true] += 1
                else:
                    stats["FP"][pred] += 1
                    stats["FN"][true] += 1
            elif true is None and pred is not None:
                stats["FP"][pred] += 1
            elif true is not None and pred is None:
                stats["FN"][true] += 1

        self.logger.info(f"==== Statistik Evaluasi Deteksi: {title} ====")
        all_labels = sorted(set(list(stats["TP"].keys()) + list(stats["FP"].keys()) + list(stats["FN"].keys())))

        csv_rows = []
        for label in all_labels:
            label_name = self.valid_labels.get(label, f"Unknown ({label})")
            tp = stats["TP"].get(label, 0)
            fp = stats["FP"].get(label, 0)
            fn = stats["FN"].get(label, 0)
            self.logger.info(f"Kelas '{label_name}': TP={tp}, FP={fp}, FN={fn}")

            csv_rows.append({
                "kelas_id": label,
                "kelas_nama": label_name,
                "TP": tp,
                "FP": fp,
                "FN": fn
            })

        # Simpan ke CSV
        csv_output_path = self.output_path / f"fold_{self.fold}_stats_{title.replace(' ', '_').lower()}.csv"
        try:
            with open(csv_output_path, mode="w", newline='', encoding="utf-8") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=["kelas_id", "kelas_nama", "TP", "FP", "FN"])
                writer.writeheader()
                writer.writerows(csv_rows)
            self.logger.info(f"Statistik deteksi disimpan ke {csv_output_path}")
        except Exception as e:
            self.logger.error(f"Gagal menyimpan statistik ke CSV: {e}")

    # Melakukan pembuatan Confusion Matrix dengan Score
    def match_pred_with_gt(self, score_threshold, iou_threshold):
        """
        Pencocokan prediksi dengan ground truth multi-kelas menggunakan confidence score dan IoU.
        """
        if not self.predictions_json.exists():
            self.logger.error(f"File prediksi tidak ditemukan: {self.predictions_json}")
            return

        try:
            with open(self.predictions_json, "r") as f:
                predictions = json.load(f)

            gt_json_path = self.output_gt / f"fold_{self.fold}_labels.json"
            if gt_json_path.exists():
                with open(gt_json_path, "r") as f:
                    ground_truth = json.load(f)
                self.logger.info(f"Ground truth dimuat dari {gt_json_path}")
            else:
                ground_truth = self.load_gt_labels()

            self.matched_data = []

            for pred in predictions:
                if pred["score"] < score_threshold:
                    continue

                image_id = str(pred["image_id"])
                pred_class = self.label_mapping.get(pred["category_id"], pred["category_id"])
                pred_bbox = pred["bbox"]
                pred_score = pred["score"]

                if image_id not in ground_truth:
                    continue

                gt_classes = ground_truth[image_id]["true_classes"]
                gt_bboxes = ground_truth[image_id]["true_bboxes"]

                best_iou = 0
                best_idx = -1
                for idx, gt_bbox in enumerate(gt_bboxes):
                    iou = self.compute_iou(pred_bbox, gt_bbox)
                    if iou > best_iou:
                        best_iou = iou
                        best_idx = idx

                if best_iou >= iou_threshold:
                    gt_class = gt_classes[best_idx]
                    self.matched_data.append({
                        "image_id": image_id,
                        "true_label": gt_class,
                        "pred_label": pred_class,
                        "score": pred_score,
                        "iou": best_iou
                    })

            self.save_matched_predictions()

        except Exception as e:
            self.logger.error(f"Kesalahan saat matching prediksi dan GT: {e}")
    
    def save_matched_predictions(self):
        """
        Menyimpan hasil pencocokan GT vs Prediksi ke file JSON,
        lalu memanggil compute_confusion_matrix untuk visualisasi.
        """
        output_json_path = self.output_json / f"fold_{self.fold}_matched_deteksi.json"

        try:
            self.output_json.mkdir(parents=True, exist_ok=True)
            with open(output_json_path, "w") as f:
                json.dump(self.matched_data, f, indent=4)
            self.logger.info(f"Hasil pencocokan disimpan di {output_json_path}")

            # Setelah menyimpan, buat confusion matrix
            self.compute_confusion_matrix(output_json_path)

        except Exception as e:
            self.logger.error(f"Gagal menyimpan hasil ke {output_json_path}: {e}")

    def compute_confusion_matrix(self, matched_json_path):
        """
        Membuat dan menyimpan confusion matrix multi-kelas dari matched.json
        (menggunakan score dan IoU).
        """
        try:
            with open(matched_json_path, "r") as f:
                matched_data = json.load(f)
                self.log_detection_statistics(matched_data, title="Dengan Score + IoU")

            # Hanya ambil pasangan yang punya true_label valid
            y_true = [item["true_label"] for item in matched_data if item["true_label"] is not None]
            y_pred = [item["pred_label"] for item in matched_data if item["true_label"] is not None]

            labels = sorted(list(set(y_true + y_pred)))
            cm = confusion_matrix(y_true, y_pred, labels=labels)
            cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True)

            plt.figure(figsize=(10, 8))
            sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Reds",
                        xticklabels=labels, yticklabels=labels)

            plt.xlabel("Predicted Label")
            plt.ylabel("True Label")
            plt.title(f"Confusion Matrix (Dengan Score + IoU) - Fold {self.fold}")

            output_img_path = self.output_cm_d / f"fold_{self.fold}_confusion_matrix_thresholded.png"

            plt.savefig(output_img_path)
            plt.close()

            self.logger.info(f"Confusion matrix (thresholded) disimpan di {output_img_path}")
            self.logger.info(f"Total pasangan valid (GT + Pred): {len(y_true)}")

        except Exception as e:
            self.logger.error(f"Gagal membuat confusion matrix dari {matched_json_path}: {e}")
    
    # Confusion Matrix without Score
    def extract_all_predictions_without_score(self):
        """
        Mengekstrak semua prediksi dari predictions.json tanpa mempertimbangkan nilai score,
        untuk digunakan dalam analisis confusion matrix multi-kelas mentah.
        """
        if not self.predictions_json.exists():
            self.logger.error(f"File prediksi tidak ditemukan: {self.predictions_json}")
            return

        try:
            with open(self.predictions_json, "r") as f:
                data = json.load(f)

            extracted_predictions = []
            for item in data:
                image_id = item.get("image_id")
                category_id = item.get("category_id")
                pred_label = self.label_mapping.get(category_id, category_id)
                bbox = item.get("bbox", [])

                if pred_label in self.valid_labels:
                    extracted_predictions.append({
                        "image_id": image_id,
                        "pred_label": pred_label,
                        "bbox": bbox
                    })

            output_json_path = self.output_pred / f"fold_{self.fold}_raw_predictions.json"
            self.output_pred.mkdir(parents=True, exist_ok=True)
            with open(output_json_path, "w") as f:
                json.dump(extracted_predictions, f, indent=4)

            self.logger.info(f"Semua prediksi tanpa score disimpan di {output_json_path}")

        except Exception as e:
            self.logger.error(f"Kesalahan saat mengekstrak prediksi tanpa score: {e}")

    def match_raw_pred_with_gt(self, iou_threshold):
        """
        Pencocokan prediksi mentah dengan GT menggunakan IoU tanpa confidence score.
        Mencocokkan prediksi dari raw_predictions.json (tanpa score) dengan ground truth labels.json.
        Hasil disimpan dalam matched_raw.json untuk pembuatan confusion matrix tanpa threshold.
        """
        raw_pred_path = self.output_pred / f"fold_{self.fold}_raw_predictions.json"
        if not raw_pred_path.exists():
            self.logger.error(f"File prediksi mentah tidak ditemukan: {raw_pred_path}")
            return

        try:
            with open(raw_pred_path, "r") as f:
                raw_predictions = json.load(f)

            gt_json_path = self.output_gt / f"fold_{self.fold}_labels.json"
            if gt_json_path.exists():
                with open(gt_json_path, "r") as f:
                    ground_truth = json.load(f)
                self.logger.info(f"Ground truth dimuat dari {gt_json_path}")
            else:
                ground_truth = self.load_gt_labels()

            self.matched_data = []

            pred_by_image = {}
            for pred in raw_predictions:
                image_id = str(pred["image_id"])
                pred_by_image.setdefault(image_id, []).append(pred)

            for image_id, gt in ground_truth.items():
                gt_classes = gt["true_classes"]
                gt_bboxes = gt["true_bboxes"]
                preds = pred_by_image.get(image_id, [])

                matched_idxs = set()

                for pred in preds:
                    pred_class = pred["pred_label"]
                    pred_bbox = pred["bbox"]

                    best_iou = 0
                    best_idx = -1
                    for idx, gt_bbox in enumerate(gt_bboxes):
                        if idx in matched_idxs:
                            continue  # hanya cocokkan satu kali
                        iou = self.compute_iou(pred_bbox, gt_bbox)
                        if iou > best_iou:
                            best_iou = iou
                            best_idx = idx

                    if best_iou >= iou_threshold:
                        gt_class = gt_classes[best_idx]
                        self.matched_data.append({
                            "image_id": image_id,
                            "true_label": gt_class,
                            "pred_label": pred_class,
                            "iou": best_iou
                        })
                        matched_idxs.add(best_idx)
                    else:
                        self.matched_data.append({
                            "image_id": image_id,
                            "true_label": None,
                            "pred_label": pred_class,
                            "iou": best_iou
                        })

            self.save_raw_matched_predictions()

        except Exception as e:
            self.logger.error(f"Kesalahan saat matching raw pred dan GT: {e}")

    def save_raw_matched_predictions(self):
        """
        Menyimpan hasil pencocokan prediksi mentah dan ground truth ke dalam JSON.
        """
        output_json_path = self.output_json / f"fold_{self.fold}_matched_raw.json"

        try:
            self.output_json.mkdir(parents=True, exist_ok=True)
            with open(output_json_path, "w") as f:
                json.dump(self.matched_data, f, indent=4)
            self.logger.info(f"Hasil pencocokan raw disimpan di {output_json_path}")

            # Bisa dikomentar jika compute belum dibuat
            # self.compute_confusion_matrix(output_json_path)

        except Exception as e:
            self.logger.error(f"Gagal menyimpan matched raw ke {output_json_path}: {e}")
    
    def compute_confusion_matrix_raw(self):
        """
        Membuat dan menyimpan confusion matrix multi-kelas dari matched_raw.json
        (tanpa menggunakan score, tetap menggunakan IoU).
        """
        matched_json_path = self.output_json / f"fold_{self.fold}_matched_raw.json"

        if not matched_json_path.exists():
            self.logger.error(f"File matched_raw.json tidak ditemukan di {matched_json_path}")
            return

        try:
            with open(matched_json_path, "r") as f:
                matched_data = json.load(f)
                self.log_detection_statistics(matched_data, title="Tanpa Score (IoU Only)")

            # Abaikan pasangan yang tidak memiliki GT valid
            y_true = [item["true_label"] for item in matched_data if item["true_label"] is not None]
            y_pred = [item["pred_label"] for item in matched_data if item["true_label"] is not None]

            labels = sorted(list(set(y_true + y_pred)))
            cm = confusion_matrix(y_true, y_pred, labels=labels)
            cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True)

            plt.figure(figsize=(10, 8))
            sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Greens",
                        xticklabels=labels, yticklabels=labels)

            plt.xlabel("Predicted Label")
            plt.ylabel("True Label")
            plt.title(f"Confusion Matrix (Tanpa Score, IoU Only) - Fold {self.fold}")

            output_img_path = self.output_cm_d / f"fold_{self.fold}_confusion_matrix_raw.png"
            plt.savefig(output_img_path)
            plt.close()

            self.logger.info(f"Confusion matrix (raw) disimpan di {output_img_path}")
            self.logger.info(f"Total pasangan valid (GT + Pred): {len(y_true)}")

        except Exception as e:
            self.logger.error(f"Gagal membuat confusion matrix raw dari {matched_json_path}: {e}")

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
    'score':0.5,
    'iou':0.5
}

def main():
    # Running Program
    print("\n=== Memulai Pembuatan Grafik Variabel-Variabel dan Confusion Matrix ===\n")
    data = input(f"Masukkan jenis data citra yang ingin diolah hasilnya (nonbg/bg/mix): ").strip()
    
    # Pastikan input valid
    if data not in {"nonbg", "bg", "mix"}:
        raise ValueError(f"Jenis data '{data}' tidak valid. Harus 'nonbg', 'bg', atau 'mix'.")
    
    # memilih salah satu
    # Input jenis model (biasa/kustom)
    model_type = input("Masukkan hasil model yang ingin kamu gunakan (biasa/kustom): ").strip()
    if model_type not in {"biasa", "kustom"}:
        raise ValueError("Maaf, jenis model yang tersedia hanya 'biasa' atau 'kustom'.")

    # Tentukan fold mode
    fold_mode = input("Ingin menjalankan semua fold atau fold tertentu saja? (all/nomor): ").strip()
    if fold_mode == "all":
        fold_range = range(1, OLAH_CONFIG['n_folds'] + 1)
    elif fold_mode.isdigit() and 1 <= int(fold_mode) <= OLAH_CONFIG['n_folds']:
        fold_range = [int(fold_mode)]
    else:
        raise ValueError("Input fold tidak valid. Masukkan 'all' atau angka fold yang valid.")
    
    for fold in fold_range:
        torch.cuda.empty_cache()
        print(f"\nProcessing fold {fold}...")
        # Pilih path hasil model fitting berdasarkan input
        size = input("\nMasukkan size hasil model (n/s/m/l/xl untuk biasa, s/m/l untuk kustom): ").strip()
        valid_sizes = {"n", "s", "m", "l", "xl"} if model_type == "biasa" else {"s", "m", "l"}
        if size not in valid_sizes:
            raise ValueError(f"Size model '{size}' tidak valid untuk model {model_type}.")

        proses_path = MAIN_PATH /"olah_data" / data # Direktori Main Olah Data
        log_path = proses_path /f"olah_data_{data}.txt"
        result_path = MAIN_PATH / data / f"hasil_{data}" / model_type / f"ukuran_{size}" # Direktori Hasil Proses Fitting, dan Evaluating Model
        json_path = result_path /f"validating_{fold}_{data}" # Direktori Hasil Proses Evaluating Model
        labels_path = DATASET_PATH /f"dataset_{data}"/"fold"/f"fold_{fold}"/"val"/"labels" # Direktori Label Dataset Fold
        try:
            # Inisialisasi logger
            logger=LoggerManager(log_file=log_path)
            
            # Menjalankan Pemindahan Data CSV Mentah ke Data CSV Olah Data
            logger.debug("\nMemulai Pemindahan CSV berdasarkan Variabel-Variabel nya untuk diolah ...\n")
            csv_remaker = CSVRemaker(result_path, proses_path, fold, data, model_type, size)
            csv_remaker.save_fold_metrics()
            csv_remaker.save_combined_metrics()
            logger.debug("\nPemindahan CSV telah selesai dan dapat diolah untuk Grafik dan Confusion Matrix.\n")
            
            # Menjalankan Plotting hasil CSV Olah Data
            logger.debug("\nMemulai menjalankan Plotting...\n")
            plotter = CSVPlotter(proses_path, fold, data, model_type, size)
            plotter.plot_all()
            logger.debug("\nPlotting telah selesai.\n")

            # Ganti 'hasil.json' dengan path file JSON kamu
            cm_generator =  PredictionClassificator(json_path, labels_path, proses_path, fold, data, model_type, size)
            
            # Menghitung dan menampilkan confusion matrix biner
            logger.info("Memulai untuk Membuat File GT berformat JSON...")
            cm_generator.filter_predictions()
            cm_generator.match_pred_with_gt()
            logger.info("Olah Data dan Confusion Matrix Binerselesai dibuat.")

            # ===============================
            # Deteksi Multi-Kelas (Detector)
            # ===============================
            logger.info("Memulai proses Confusion Matrix Deteksi Multi-Kelas...")

            detector = PredictionDetector(json_path, labels_path, proses_path, fold, data, model_type, size)

            # (1) Gunakan prediksi mentah (tanpa score)
            detector.extract_all_predictions_without_score()  # Simpan raw_predictions.json
            detector.match_raw_pred_with_gt(iou_threshold=OLAH_CONFIG['iou'])  # Cocokkan dengan IoU
            detector.compute_confusion_matrix_raw()  # Visualisasi dan log

            # (2) Gunakan prediksi dengan score threshold
            detector.match_pred_with_gt(score_threshold=OLAH_CONFIG['score'], iou_threshold=OLAH_CONFIG['iou'])
            # Ini otomatis akan simpan + compute confusion matrix thresholded

            logger.info("Confusion Matrix Deteksi Multi-Kelas selesai.\n")
            
        except Exception as e:
            logger.error(f"\nError processing fold {fold}:")
            logger.error(f"Error details: {str(e)}")
            logger.error(f"Fold {fold} failed: {str(e)}\n")
            continue

if __name__ == "__main__":
    main()