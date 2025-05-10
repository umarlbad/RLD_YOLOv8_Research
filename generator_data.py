import yaml, shutil, traceback, logging
import numpy as np
from pathlib import Path
from collections import Counter
from sklearn.model_selection import StratifiedKFold
from dataset_loader import YOLODataset, DatasetVerificator
from utilisasi import VisualisasiDatasetAwal

# Setup logging yang lebih terstruktur
class LoggerManager:
    def __init__(self, log_file):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        # File handler
        if not self.logger.handlers:
            try:
                file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
                file_handler.setLevel(logging.INFO)
                file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
                file_handler.setFormatter(file_formatter)
            
                # Console handler
                console_handler = logging.StreamHandler()
                console_handler.setLevel(logging.INFO)
                console_formatter = logging.Formatter('%(levelname)s: %(message)s')
                console_handler.setFormatter(console_formatter)
                
                # Add handlers
                self.logger.addHandler(file_handler)
                self.logger.addHandler(console_handler)

            except Exception as e:
                print(f"Error setting up logger: {e}")
                raise
        
    def info(self, message):
        """Log info message to both file and console"""
        self.logger.info(message)
        
    def error(self, message):
        """Log error message to both file and console"""
        self.logger.error(message)
        
    def debug(self, message):
        """Log debug message to both file and console"""
        self.logger.debug(message)

class DataGenerator:
    def __init__(self, dataset, validation_split, n_splits, shuffle, log_dir:Path, data:str):
        """
        Initialize DataGenerator with cross validation
        Args:
            dataset: YOLODatasetnn instance
            validation_split: proportion of validation data 
            n_splits: number of folds for cross validation
            shuffle: whether to shuffle the data
        """
        self.log_file = log_dir
        self.logger = LoggerManager(self.log_file)
        self.logger.info("\n ----- Inisialisasi DataGenerator ----- \n")

        try:
            if not 0 < validation_split < 1:
                raise ValueError("validation_split must be between 0 and 1")
            if not isinstance(n_splits, int) or n_splits < 2:
                raise ValueError("n_splits must be an integer greater than 1")
            
            self.dataset = dataset
            self.data = data
            self.validation_split = validation_split
            self.test_split = 0.05
            self.n_splits = n_splits
            self.shuffle = shuffle
            self.data = data
            # Setup directory structure
            self.main_dir = Path(dataset.main_dir)
            self.logger.info(f"Main Directory: \n{self.main_dir}\n")
            self.images_dir = Path(dataset.images_dir)
            self.logger.info(f"Image Directory: \n{self.images_dir}\n")
            self.labels_dir = Path(dataset.labels_dir)
            self.logger.info(f"Labels Directory: \n{self.labels_dir}\n")
            self.classes = dataset.classes
            self.nc = len(self.classes)
            self.logger.info(f"\nJumlah Kelas: {self.nc}")
            self.logger.info(f"Validation split: {validation_split}")
            self.logger.info(f"Number of folds: {n_splits}\n")

            # Log class information
            self.logger.info("\nClass Information:")
            for idx, class_name in enumerate(self.classes):
                self.logger.info(f"Class {idx}: {class_name}")
                
            self.logger.info("DataGenerator initialized successfully")
            
            # Convert valid_pairs to Path objects
            self.valid_pairs = [
                (Path(img_path), Path(label_path)) 
                for img_path, label_path in dataset.valid_pairs
            ]
        except Exception as e:
            self.logger.error(f"Error initializing DataGenerator: {str(e)}")
            raise

    def _is_background(self, img_id: int, class_name: str) -> bool:
        """Determine if an image is background based on its ID"""
        bg_ranges = {
            "bercak cokelat": (3601, 4200),
            "bercak cokelat tipis": (4201, 4800),
            "blas daun": (4801, 5400),
            "lepuh daun": (6001, 6600),
            "hawar daun bakteri": (5401, 6000),
            "sehat": (6600, 7200)
        }
        return bg_ranges[class_name][0] <= img_id <= bg_ranges[class_name][1]
    
    def _is_nonbackground(self, img_id: int, class_name: str) -> bool:
        """Determine if an image is background based on its ID"""
        nobg_ranges = {
            "bercak cokelat": (1, 600),
            "bercak cokelat tipis": (601, 1200),
            "blas daun": (1201, 1800),
            "lepuh daun": (2401, 3000),
            "hawar daun bakteri": (1081, 2400),
            "sehat": (3001, 3600)
        }
        return nobg_ranges[class_name][0] <= img_id <= nobg_ranges[class_name][1]
    
    def _is_mix(self, img_id :int, class_name : str) -> bool:
        mix_ranges = {
            "bercak cokelat": (1, 1200),
            "bercak cokelat tipis": (1201, 2400),
            "blas daun": (2401, 3600),
            "lepuh daun": (4801, 6000),
            "hawar daun bakteri": (3601, 4800),
            "sehat": (6001, 7200)
        }
        return mix_ranges[class_name][0] <= img_id <= mix_ranges[class_name][1]

    def _create_stratification_labels(self):
        """
        Membuat label stratifikasi yang mencakup kelas dan status background
        Returns:
            tuple: (array label stratifikasi, array indeks)
        """
        try:
            strat_labels = []
            pair_indices = []
            distributions = {cls: {'bg': [], 'nonbg': [], 'mix':[]} for cls in self.classes}
            missing_count = 0
        
            self.logger.info(f"\nDebugging stratifikasi label dari {self.data.upper()}:")
            for idx, (img_path, _) in enumerate(self.valid_pairs):
                try:
                    # Dapatkan kelas dari dataset
                    label = self.dataset[idx][2][0].item()
                    if label  == -1:
                        continue

                    class_name = self.classes[label]
                    img_id = int(img_path.stem[-4:])
                    
                    # Periksa status background
                    is_nonbg = self._is_nonbackground(img_id, class_name)
                    is_bg =  self._is_background(img_id, class_name)
                    is_mix = self._is_mix(img_id, class_name)
                    
                    # Tambahkan ke distributions
                    if is_nonbg:
                        distributions[class_name]["nonbg"].append(idx)
                    elif is_bg:
                        distributions[class_name]["bg"].append(idx)
                    elif is_mix:
                        distributions[class_name]["mix"].append(idx)
                    
                    # Tambahkan ke strat_labels dan pair_indices sesuai jenis data
                    if self.data == "nonbg" and is_nonbg:
                        strat_labels.append(f"{class_name}_nonbg")
                        pair_indices.append(idx)
                    elif self.data == "bg" and is_bg:
                        strat_labels.append(f"{class_name}_bg")
                        pair_indices.append(idx)
                    elif self.data == "mix" and is_mix:
                        strat_labels.append(f"{class_name}_mix")
                        pair_indices.append(idx)
                    else:
                        missing_count += 1
                        self.logger.error(f"Data tidak masuk kategori jenis data {self.data.upper()} - ID: {img_id}, Kelas: {class_name}")
                        
                except Exception as e:
                    self.logger.error(f"Error processing index {idx}: {str(e)}")
                    continue
                
            # Logging tambahan jika ada data yang tidak masuk kategori mana pun
            if missing_count > 0:
                self.logger.error(f"Total {missing_count} data tidak masuk kategori {self.data.upper()} dan dilewatkan.")

            # Verifikasi panjang array
            self.logger.info(f"\nJumlah strat_labels: {len(strat_labels)}")
            self.logger.info(f"Jumlah pair_indices: {len(pair_indices)}\n")
            
            # Validasi distribusi per kelas
            self.logger.info(f"\nDistribusi per kelas dataset untuk jenis data {self.data.upper()}:")
            for class_name in self.classes:
                bg_count = len(distributions[class_name]['bg'])
                nonbg_count = len(distributions[class_name]['nonbg'])
                mix_count = len(distributions[class_name]['mix'])
                
                if self.data == "nonbg":
                    self.logger.info(f"{class_name}: {nonbg_count} total (Non-BG: {nonbg_count})")
                elif self.data == "bg":
                    self.logger.info(f"{class_name}: {bg_count} total (BG: {bg_count})")
                elif self.data == "mix":
                    self.logger.info(f"{class_name}: {mix_count} total (Mix Data: {mix_count})")
                    
            # Pastikan panjang array sama
            if len(strat_labels) != len(pair_indices):
                raise ValueError(
                    f"Ketidaksesuaian panjang array: strat_labels ({len(strat_labels)}) "
                    f"!= pair_indices ({len(pair_indices)})"
                )
            
            self.logger.info(f"\nSinkronisasi selesai: {len(strat_labels)} sampel diproses")
            
            return np.array(strat_labels), np.array(pair_indices), distributions
        
        except Exception as e:
            self.logger.error(f"Error in stratification process: {str(e)}")
            raise

    def log_bbox_distribution(self, indices, split_name, fold):
        """
        Logging jumlah bounding box per kelas untuk set data latih, validasi, dan uji
        """
        bbox_count = {cls: 0 for cls in self.classes}
        for idx in indices:
            _, label_path = self.valid_pairs[idx]
            label_file = self.labels_dir / label_path
            try:
                with open (label_file, 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        class_id = int(line.split()[0])
                        class_name = self.classes[class_id]
                        bbox_count[class_name] += 1

            except Exception as e:
                self.logger.error(f"Error membaca label dari bounding box {label_path}: {e}")

        self.logger.info(f"\nDistribusi Bounding Box pada {split_name.capitalize()} (Fold {fold}):")
        for class_name, count in bbox_count.items():
            self.logger.info(f"Kelas {class_name}: {count} bounding boxes.")

    def _get_distribution(self, indices):
        """
        Mendapatkan distribusi data berdasarkan label untuk daftar indeks tertentu.
        
        Args:
            indices (list[int]): Daftar indeks data.
        
        Returns:
            dict: Distribusi data berdasarkan kelas dan status (background).
        """
        distribution = {cls: {'bg': 0,'nonbg': 0, 'mix' : 0} for cls in self.classes}

        for idx in indices:
            img_path, _ = self.valid_pairs[idx]
            label = self.dataset[idx][2][0].item()
            class_name = self.classes[label]
            img_id = int(img_path.stem[-4:])

            if self._is_nonbackground(img_id, class_name):
                distribution[class_name]['nonbg'] += 1
            elif self._is_background(img_id, class_name):
                distribution[class_name]['bg'] += 1
            elif self._is_mix(img_id, class_name):
                distribution[class_name]['mix'] += 1

        return distribution
    
    def _copy_files_to_fold(self, fold_path: Path, splits: dict):
        """Copy files ke direktori fold dengan struktur YOLO"""
        # Handle train dan val splits
        for split_type in ['train', 'val']:
            for img_path, lbl_path in splits[split_type]:
                dest_img_path = fold_path / split_type / 'images' / img_path.name
                dest_lbl_path = fold_path / split_type / 'labels' / lbl_path.name
                
                shutil.copy2(self.images_dir / img_path, dest_img_path)
                shutil.copy2(self.labels_dir / lbl_path, dest_lbl_path)

    def _create_yaml_config(self, fold: int, fold_path: Path) -> Path:
        """Create YAML configuration file for YOLO"""
        yaml_content = {
            'path': str(fold_path.parent),  # Path root
            'train': str(fold_path / 'train'),
            'val': str(fold_path / 'val'),
            'test': str(fold_path.parent / 'test'),
            'nc': self.nc,
            'names': {i: name for i, name in enumerate(self.classes)}
        }
        
        yaml_path = fold_path / f'fold_{fold}_{self.data}_config.yaml'
        with open(yaml_path, 'w') as f:
            yaml.safe_dump(yaml_content, f, sort_keys=False, default_flow_style=False)

        self.logger.info(f"\nCreated YAML config for fold {fold}: {yaml_path}")
        return yaml_path

    def _verify_fold_distribution(self, splits, fold):
        """
        Verifikasi distribusi kelas dan background dalam setiap split.
        
        Args:
            splits (dict): Dictionary dengan key 'train', 'val', dan 'test'.
            fold (int): Nomor fold.
        """
        self.logger.info(f"\n=== Distribusi Fold {fold} untuk jenis data {self.data.upper()} ===")

        for split_name, split_data in splits.items():
            self.logger.info(f"\nDistribusi {split_name.capitalize()} (Fold {fold}):")
            # Hitung distribusi untuk setiap kelas
            class_distribution = {cls: {"bg":0, "nonbg":0, "mix" :0} for cls in self.classes}
            
            try:
                # Proses setiap file dalam split
                for img_path, label_path in split_data:
                    # Dapatkan nama file dari path
                    img_filename = img_path.name if isinstance(img_path, Path) else Path(img_path).name
                    
                    # Ekstrak ID gambar dari nama file
                    try:
                        img_id = int(img_filename[-8:-4])  # Mengambil 4 digit terakhir sebelum ekstensi
                    except ValueError:
                        self.logger.error(f"Tidak dapat mengekstrak ID dari {img_filename}")
                        continue
                    
                    # Dapatkan kelas dari label path
                    try:
                        with open(self.labels_dir / label_path, 'r') as f:
                            line = f.readline().strip()
                            class_id = int(line.split()[0])
                            class_name = self.classes[class_id]
                            
                            # Periksa jenis data
                            if self._is_nonbackground(img_id, class_name):
                                class_distribution[class_name]["nonbg"] += 1
                            elif self._is_background(img_id, class_name):
                                class_distribution[class_name]["bg"] += 1
                            elif self._is_mix(img_id, class_name):
                                class_distribution[class_name]["mix"] += 1

                    except Exception as e:
                        self.logger.error(f"Warning: Error membaca label untuk {label_path}: {e}")
                        self.logger.error(f"Error membaca label untuk {label_path}: {e}")
                        continue
                
                # Tampilkan distribusi
                for class_name, counts in class_distribution.items():
                    if self.data == "nonbg":
                        self.logger.info(f"  {class_name}: {counts['nonbg']} gambar nonbackground")
                    elif self.data == "bg":
                        self.logger.info(f"  {class_name}: {counts['bg']} gambar background")
                    elif self.data == "mix":
                        self.logger.info(f"  {class_name}: {counts['mix']} gambar mix data")
                
            except Exception as e:
                print(f"Error dalam verifikasi {split_name}: {e}")
                self.logger.error(f"Error dalam verifikasi {split_name}: {e}")
                continue

    def create_fold_datasets(self, output_root: str):
        """
        Membuat dataset dengan format YOLO dan stratifikasi
        Args:
            output_root: direktori output utama
        """
        output_root = Path(output_root)
        self.logger.info(f"\nMembuat fold datasets untuk jenis data {self.data.upper()} di dalam {output_root}...")
        
        try:
            self.logger.info(f"\n===== Membuat Dataset dari jenis data {self.data.upper()} dengan Format YOLO dan metode Cross-Validation =====")
            
            # Dapatkan label stratifikasi dan distribusi
            self.logger.info("- Mendapatkan stratifikasi labels...")
            strat_labels, pair_indices, distributions = self._create_stratification_labels()
            
            # Debugging jumlah data
            self.logger.info(f"Total valid_pairs: {len(self.valid_pairs)}")
            self.logger.info(f"Jumlah strat_labels: {len(strat_labels)}")
            self.logger.info(f"Jumlah pair_indices: {len(pair_indices)}")

            expected_count = 7200 if self.data == "mix" else 3600
            
            if len(self.valid_pairs) != expected_count:
                raise ValueError(f"Jumlah valid_pairs tidak sesuai dengan target: {len(self.valid_pairs)} (seharusnya {expected_count}).")
            
            if len(pair_indices) != expected_count:
                raise ValueError(f"Jumlah pair_indices tidak sesuai: {len(pair_indices)} (seharusnya {expected_count}).")
            
            if len(strat_labels) != expected_count:
                raise ValueError(f"Jumlah strat_labels tidak sesuai: {len(strat_labels)} (seharusnya {expected_count}).")
            
            # Pisahkan test set dengan mempertahankan proporsi background/non-background
            test_indices = []
            train_val_indices = []
            
            # Untuk setiap kelas, pilih proporsi yang seimbang untuk test set
            for class_name in self.classes:
                bg_indices = distributions[class_name].get("bg", [])
                nonbg_indices = distributions[class_name].get("nonbg", [])
                mix_indices = distributions[class_name].get("mix", [])
                
                if self.data == "nonbg":
                    selected_indices = nonbg_indices
                elif self.data == "bg":
                    selected_indices = bg_indices
                elif self.data == "mix":  
                    selected_indices = mix_indices
                
                # Hitung jumlah sampel yang dibutuhkan untuk test set
                test_sample_size = int(len(selected_indices) * self.test_split) if len(selected_indices) > 0 else 0
                
                # Pilih sampel secara random
                np.random.seed(42)  # Untuk reproducibility
                test_samples = np.random.choice(selected_indices, test_sample_size, replace=False)
                
                # Gabungkan ke test indices dan train-val
                test_indices.extend(test_samples)
                train_val_indices.extend([idx for idx in selected_indices if idx not in test_samples])
            
            # Konversi ke numpy array
            test_indices = np.array(test_indices)
            # Simpan data test
            test_pairs = [self.valid_pairs[i] for i in test_indices]

            # Debugging panjang array
            self.logger.debug(f"Jumlah test_indices: {len(test_indices)}")
            
            # Proses direktori test sekali saja Print distribusi test set
            self.logger.debug("\n=== Distribusi Test Set ===")
            test_distribution = self._get_distribution(test_indices)
            for class_name, counts in test_distribution.items():
                self.logger.debug(f"\n{class_name}:")
                if self.data == "nonbg":
                    self.logger.debug(f"  Nonbackground: {counts['nonbg']}")
                elif self.data == "bg":
                    self.logger.debug(f"  Background: {counts['bg']}")
                elif self.data == "mix":
                    self.logger.debug(f"  Mix Data: {counts['mix']}")
                
            test_dir = output_root / 'test'
            (test_dir / 'images').mkdir(parents=True, exist_ok=True)
            (test_dir / 'labels').mkdir(parents=True, exist_ok=True)
            
            # Copy file test
            for img_path, label_path in test_pairs:
                shutil.copy2(self.images_dir / img_path, test_dir / 'images' / img_path.name)
                shutil.copy2(self.labels_dir / label_path, test_dir / 'labels' / label_path.name)
                
            # Validasi jumlah total
            total_data = len(test_indices) + len(train_val_indices)
            if total_data != expected_count:
                raise ValueError(f"Jumlah total data (train_val + test) tidak sesuai: {total_data} (seharusnya 2628).")
            
            # Dapatkan label untuk train_val set
            train_val_indices = np.array(train_val_indices, dtype=int)
            train_val_labels = strat_labels[train_val_indices]
            self.logger.debug(f"Jumlah train_val_indices: {len(train_val_indices)}")
            
            # Tambahkan setelah mendapatkan train_val_labels
            self.logger.debug("\nDistribusi label sebelum split:")
            for label in set(train_val_labels):
                count = sum(1 for x in train_val_labels if x == label)
                self.logger.debug(f"Label {label}: {count} sampel")
                
            # Periksa distribusi data train-val
            train_val_counts = Counter(train_val_labels)
            
            # Debugging distribusi data
            self.logger.debug(f"\nDistribusi Train-Val: {train_val_counts}")
            self.logger.debug(f"Distribusi Test: {self._get_distribution(test_indices)}\n")

            # Hitung jumlah minimum sampel per kelas
            min_samples_per_class = min(train_val_counts.values())
            if self.n_splits >= min_samples_per_class:
                self.logger.debug(f"\nn_splits={self.n_splits} terlalu besar untuk data ini. "
                    f"Mengubah n_splits menjadi {min_samples_per_class}.")
                
                return min_samples_per_class
        
            # Lanjutkan dengan cross validation seperti sebelumnya
            skf = StratifiedKFold(n_splits=self.n_splits, shuffle=self.shuffle, random_state=42)
            
            # Iterasi melalui fold
            for fold, (train_idx, val_idx) in enumerate(skf.split(train_val_indices, train_val_labels), 1):
                
                self.logger.info(f"\nMemproses Fold {fold}...")
                
                # Proses fold seperti sebelumnya...
                train_indices = train_val_indices[train_idx]
                val_indices = train_val_indices[val_idx]
                
                # Hitung Presentase
                total_samples = total_data
                train_percent = len(train_indices) / total_samples * 100
                val_percent = len(val_indices) / total_samples * 100
                test_percent = len(test_indices) / total_samples * 100

                self.logger.info(f"Fold {fold} distribution:")
                self.logger.info(f"Training: {len(train_indices)}, samples ({train_percent:.2f}%)")
                self.logger.info(f"Validation: {len(val_indices)}, samples ({val_percent:.2f}%)")
                self.logger.info(f"Testing: {len(test_indices)}, samples ({test_percent:.2f}%)")

                # Setup direktori fold
                fold_path = output_root / f"fold_{fold}"

                # Buat struktur direktori YOLO
                for split in ['train', 'val']:
                    split_path = fold_path / split
                    self.logger.info(f"\nCreating {split} directory: {split_path}")
                    (split_path / 'images').mkdir(parents=True, exist_ok=True)
                    (split_path / 'labels').mkdir(parents=True, exist_ok=True)
                
                # Konversi indeks ke pasangan file dan salin
                splits = {
                    'train': [self.valid_pairs[i] for i in train_indices],
                    'val': [self.valid_pairs[i] for i in val_indices],
                    'test': test_pairs
                }
                
                # Copy files ke struktur YOLO
                self._copy_files_to_fold(fold_path, splits)
                
                self.log_bbox_distribution(train_indices, 'train', fold)
                self.log_bbox_distribution(val_indices, 'val', fold)
                self.log_bbox_distribution(test_indices, 'test', fold)

                # Buat konfigurasi YAML
                yaml_path = self._create_yaml_config(fold, fold_path)
                self.logger.info(f"\nCreated YAML config: {yaml_path}\n")
                self.logger.info(f"\nYAML konfigurasi dibuat untuk Fold {fold}: {yaml_path}\n")
                
                # Verifikasi distribusi
                self._verify_fold_distribution(splits, fold)
                
                yield train_indices, val_indices, test_indices, yaml_path

        except Exception as e:
            self.logger.error(f"\nError dalam create_fold_datasets: {e} \n")
            traceback.print_exc()

    def validate_fold_labels(self, main_dir, classes, n_splits):
        """
        Validasi label untuk semua fold di dataset
        
        Args:
            main_dir (str): Direktori utama dataset
            classes (list): Daftar nama kelas
            n_splits (int): Jumlah fold cross-validation
        """
        self.logger.info("\n===== Validasi Label Secara Menyeluruh =====")
        logging_path = self.log_file
        data = self.data
        
        validator = LabelValidator(main_dir, classes, logging_path, data)
        
        # Validasi label training dan validasi untuk setiap fold
        for fold in range(1, n_splits + 1):
            self.logger.info(f"\n--- Fold {fold} ---")
            
            fold_base_dir = Path(main_dir) / 'fold' / f'fold_{fold}'
            
            for split in ['train', 'val']:
                label_dir = fold_base_dir / split / 'labels'
                self.logger.info(f"\nValidasi label {split.upper()}:")
                
                validator.validate_labels(label_dir)
                validator.print_validation_results()
        
        # Validasi label testing
        self.logger.info("\n--- Test Set ---")
        test_label_dir = Path(main_dir) / 'test' / 'labels'
        validator.validate_labels(test_label_dir)
        validator.print_validation_results()

class LabelValidator:
    def __init__(self, dataset_dir, classes, log_dir:Path, data:str):
        """
        Inisialisasi validator label
        
        Args:
            dataset_dir (str/Path): Direktori utama dataset
            classes (list): Daftar nama kelas
        """
        self.dataset_dir = Path(dataset_dir)
        self.classes = classes
        self.data = data
        self.log_file = log_dir
        self.invalid_files = []
        self.logger = LoggerManager(log_file=self.log_file)

    def validate_labels(self, label_dir):
        """
        Validasi file label dalam direktori
        
        Args:
            label_dir (str/Path): Direktori label yang akan divalidasi
        
        Returns:
            list: Daftar file label yang tidak valid
        """
        label_dir = Path(label_dir)
        self.invalid_files = []

        for file_path in label_dir.glob('*/*.txt'):
            try:
                with open(file_path, 'r') as f:
                    for line_num, line in enumerate(f, 1):
                        parts = line.strip().split()
                        
                        # Validasi jumlah kolom (harus 5)
                        if len(parts) != 5:
                            self.invalid_files.append({
                                'file': file_path.name,
                                'line': line_num,
                                'error': "Jumlah kolom tidak sesuai"
                            })
                            continue

                        # Validasi format kelas dan koordinat
                        try:
                            class_id = int(parts[0])
                            if class_id < 0 or class_id >= len(self.classes):
                                self.invalid_files.append({
                                    'file': file_path.name,
                                    'line': line_num,
                                    'error': f"ID kelas tidak valid: {class_id}"
                                })
                                continue

                            # Validasi koordinat (x, y, w, h) dalam rentang 0-1
                            coords = list(map(float, parts[1:]))
                            if not all(0 <= coord <= 1 for coord in coords):
                                self.invalid_files.append({
                                    'file': file_path.name,
                                    'line': line_num,
                                    'error': "Koordinat di luar rentang 0-1"
                                })
                        except ValueError:
                            self.invalid_files.append({
                                'file': file_path.name,
                                'line': line_num,
                                'error': "Format koordinat atau kelas tidak valid"
                            })

            except Exception as e:
                self.logger.error(f"Error membaca file {file_path}: {e}")

        return self.invalid_files

    def print_validation_results(self):
        """Cetak hasil validasi label"""
        if not self.invalid_files:
            self.logger.info("✅ Semua file label valid.")
        else:
            self.logger.info("❌ Ditemukan label tidak valid:")
            for invalid in self.invalid_files:
                self.logger.info(f" - File: {invalid['file']}, Baris: {invalid['line']}, Error: {invalid['error']}")

# Config
DATASET_DIR = Path("D:/Riset Skripsi/dataset_skripsi")
MAIN_DIR = Path("D:/Riset Skripsi/script riset/deteksi_citra")
CLASSES = ["bercak cokelat",
        "bercak cokelat tipis",
        "blas daun",
        "lepuh daun",
        "hawar daun bakteri",
        "sehat"]
CLASS_MAPPING = {0: "bercak cokelat", 1: "bercak cokelat tipis", 2: "blas daun", 3: "lepuh daun", 4: "hawar daun bakteri", 5: "sehat"}

def main():
    # Running Program
    print("\n=== Memulai Generator Dataset Untuk Proses Pembagian menjadi set data latih:validasi:uji ===\n")
    data = input(f"Masukkan jenis data citra yang ingin diolah hasilnya (nonbg/bg/mix): ").strip()
    
    # Pastikan input valid
    if data not in {"nonbg", "bg", "mix"}:
        raise ValueError(f"Jenis data '{data}' tidak valid. Harus 'nonbg', 'bg', atau 'mix'.")# Initialize Logging function
    
    # Initialize paths and parameters
    dataset_path = DATASET_DIR / f"dataset_{data}"
    
    
    # Create output directory for visualizations
    gambar_dir = MAIN_DIR / data / "gambar"
    fold_dir = DATASET_DIR / f"dataset_{data}" / "fold"
    fold_dir.mkdir(parents=True, exist_ok=True)
    
    # Inisialisasi Logging
    LOG_PATH = MAIN_DIR / data / 'logging'
    log_file=LOG_PATH / f"dataset{data}_summary.txt"
    logger = LoggerManager(log_file=log_file)

    try:
        # Create dataset instance
        logger.info(f"\n=== Memulai Generator Dataset jenis data {data.upper()} Untuk Proses Pembagian menjadi set data latih:validasi:uji ===\n")
        logger.info("\nInisialisasi Dataset YOLOV8 . . .")
        dataset = YOLODataset(dataset_path, CLASSES, LOG_PATH, data)
        
        # Tambahkan logging untuk memverifikasi dataset
        logger.info(f"\nDataset {data.upper()} Verification:")
        logger.info(f"Total dataset length: {len(dataset)}")
        logger.info(f"Classes: {CLASSES}")
        
        # Verifikasi dataset
        info_dataset = DatasetVerificator(dataset, CLASSES, LOG_PATH, data)
        logger.info("Verifikasi dataset berhasil")

        # Tampilkan informasi detail dataset
        if 'info_dataset' in locals():
            info_dataset.display_detailed_dataset_info()
            info_dataset.display_label_file_info()
            info_dataset.display_image_directory_info()
            info_dataset.display_sample(1)  # Tampilkan 1 sampel random`
            
        # Initialize data generator with parameters
        logger.info(f"\nSetting up data generator {data.upper()}...")
        data_generator = DataGenerator(
            dataset=dataset,
            validation_split=0.10,
            n_splits=5,  # n-fold cross validation
            shuffle=True,
            log_dir=log_file,
            data=data
        )
        
        # Inisialisasi Visualisasi Dataset berdasarkan set data
        visualizer = VisualisasiDatasetAwal(CLASS_MAPPING,  hide_labels=False, logger=logger)
        
        # Create datasets for each fold with enhanced logging
        fold_stats = []    
        # Create datasets for each fold
        logger.info(f"\nGenerating {data.upper()} Data cross-validation folds...")
        for fold_idx, (train_indices, val_indices, test_indices, yaml_path) in enumerate(data_generator.create_fold_datasets(fold_dir), 1):
            test_indices_list = test_indices

            logger.debug(f"\nFold {fold_idx} Created:")
            logger.debug("-" * 50)
            
            logger.debug(f"\nYAML config: {yaml_path}")
            logger.debug(f"Training samples: {len(train_indices)}")
            logger.debug(f"Validation samples: {len(val_indices)}")
            logger.debug(f"Total testing samples: {len(test_indices_list)}\n")
            
            # Store fold statistics
            fold_stats.append({
                'fold': fold_idx,
                'train_samples': len(train_indices),
                'val_samples': len(val_indices),
                'test_samples': len(test_indices_list)
            })
            
            # Summary of all folds
            logger.info("\n--- Fold Statistics Summary ---")
            for stat in fold_stats:
                print(f"Fold {stat['fold']}: "
                    f"Train={stat['train_samples']}, "
                    f"Val={stat['val_samples']}, "
                    f"Test={stat['test_samples']}")
            
            # Visualize samples for each fold
            logger.info(f"\nVisualizing the original data for fold {fold_idx}...\n")
            visualizer.visualize_samples(data_generator.dataset, test_indices, num_samples=4, output_path=gambar_dir / "test_samples.png", title="Test Samples")
            visualizer.visualize_samples(data_generator.dataset, train_indices, num_samples=4, output_path=gambar_dir / f"train_samples_fold_{fold_idx}.png", title="Train Samples")
            visualizer.visualize_samples(data_generator.dataset, val_indices, num_samples=4, output_path=gambar_dir / f"val_samples_fold_{fold_idx}.png", title="Validation Samples")
    
        # Melakukan Validasi kembali dataset yang telah dibagi baik pada gambar atau label
        data_generator.validate_fold_labels(dataset_path, CLASSES, n_splits=5)
        logger.info(f"Dataset summary has been saved to {log_file}")
    except Exception as e:
        logger.error(f"\nError occurred during setup: {str(e)}")
        raise
        
if __name__ == "__main__":
    main()