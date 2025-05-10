# fitting_model_nn.py
import torch, os, yaml, traceback, logging, time
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO
from trainer import TrainerModel
from validator import ValidatorModel


# Device configuration
device = 'cuda' if torch.cuda.is_available() else 'cpu'
os.environ["TORCH_USE_CUDA_DSA"] = "1"

# Path configurations
MAIN_DIR = Path("D:/Riset Skripsi/script riset/deteksi_citra")
DATASET_DIR = Path("D:/Riset Skripsi/dataset_skripsi")

CLASSES = [
    "bercak cokelat",
    "bercak cokelat tipis",
    "blas daun",
    "lepuh daun",
    "hawar daun bakteri",
    "sehat"
]

TRAINING_CONFIG = {
    'epochs': 150,
    'batch_size': 16,
    'image_size': 640,
    'n_folds': 5,
    'workers':6,
    'patience': 50,
    'save_period': 75,
    'freeze': [0,1,2,3,4,5,6,7,8],
    'optimizer': 'AdamW',
    'learning_rate': 25e-4,
    'learning_rate_final':5e-4,
    'momentum': 0.95,
    'weight_decay': 5e-4,
    'warmup_epochs': 20,
    'warmup_momentum': 0.85,
    'warmup_bias_lr': 15e-4,
    'close_mosaic' : 25,
    'seed' : 42,
    'box': 0.75, # bobot untuk box_loss (alpha)
    'cls': 0.075, # bobot untuk cls_loss (beta)
    'dfl': 0.235, # bobot untuk dfl_loss (gamma)
    'conf': 0.35, # nilai threshold score untuk hasil prediksi (pada validasi) = untuk penentuan jumlah hasil prediksi dan dikategorikan berdasarkan TP, FP, FN, dan TN yang kemudian dimasukkan ke dalam confusion matrix
    'kobj':1.5,
    'pose':12.0,
    'dropout':0.25,
    'nbs':16,
    'iou':0.5,
    'line_width':2,
    'augment_zero':0,
    'augment_first':0.25,
    'augment_sec':0.5
}
class LoggerManager:
    def __init__(self, log_file):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        # Membuat direktori jika belum ada
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

        # File Handler
        if not self.logger.handlers:
            try:
                # File handler
                file_handler = logging.FileHandler(log_file, mode='w')
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

def check_dataset(yaml_path, logger):
    
    try:
        # 1. Periksa apakah file YAML ada
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"YAML file not found: {yaml_path}")
        
        # 2. Load file YAML
        logger.info(f"\nYAML Path: {yaml_path}")
        with open(yaml_path, 'r', encoding='utf-8') as file:
            data = yaml.safe_load(file)
        
        # 3. Ambil path train, val, dan konfigurasi lain dari YAML
        train_dir = Path(data['train'])
        val_dir = Path(data['val'])
        logger.info("\nIsi YAML file:")
        logger.info(data)
        
        # 4. Hitung file gambar dan label
        train_images = list(train_dir.glob('**/*.[JjPp][PpNn][Gg]'))  # jpg, JPG, png, PNG, etc.
        train_labels = list((train_dir / 'labels').glob('**/*.txt'))
        val_images = list(val_dir.glob('**/*.[JjPp][PpNn][Gg]'))  # jpg, JPG, png, PNG, etc.
        val_labels = list((val_dir / 'labels').glob('**/*.txt'))
        
        # 5. Cetak informasi dataset
        logger.info("\n1). YAML Configuration:")
        logger.info(f"- Number of Classes = {data.get('nc')}")
        logger.info(f"- Class Names = {data.get('names')}")
        
        logger.info("\n2). Directory Paths:")
        logger.info(f"-> a). Train Directory Path: {train_dir}")
        logger.info(f"-> b). Val Directory Path: {val_dir}")
        
        logger.info("\n3). Dataset Overview:")
        logger.info(f"-> c). Total Training Images: {len(train_images)}")
        logger.info(f"-> d). Total Training Labels: {len(train_labels)}")
        logger.info(f"-> e). Total Validation Images: {len(val_images)}")
        logger.info(f"-> f). Total Validation Labels: {len(val_labels)}")
        
        # 6. Debugging detail - File gambar dan label
        logger.info("\n4). Train Images Sample:")
        logger.info(train_images[:5])  # Cetak 5 file gambar pertama
        logger.info("Train Labels Sample:")
        logger.info(train_labels[:5])  # Cetak 5 file label pertama
        
        logger.info("\n5). Val Images Sample:")
        logger.info(val_images[:5])  # Cetak 5 file gambar pertama
        logger.info("Val Labels Sample:")
        logger.info(val_labels[:5])  # Cetak 5 file label pertama
        
        # 7. Cek gambar yang tidak memiliki label (mismatch)
        logger.info("\n6). Mismatch Check:")
        train_images_stems = {img.stem for img in train_images}
        train_labels_stems = {label.stem for label in train_labels}
        images_without_labels = train_images_stems - train_labels_stems
        labels_without_images = train_labels_stems - train_images_stems
        
        logger.info(f"Train Images without Labels: {len(images_without_labels)}")
        logger.info(f"Train Labels without Images: {len(labels_without_images)}")
        if images_without_labels:
            logger.info("Images without Labels:", images_without_labels)
        if labels_without_images:
            logger.info("Labels without Images:", labels_without_images)

        logger.info("\nDataset check completed successfully!")

    except Exception as e:
        logger.error(f"Dataset validation error: {str(e)}")
            

def get_checkpoint_path(weights_path):
    weights_dir = weights_path
    if not weights_dir.exists():
        return None
    checkpoints = sorted(weights_dir.glob("epoch75.pt"))
    return checkpoints[-1] if checkpoints else None

def main():
    try:
        
        # Running Program
        print("\n=== Memulai Fitting Model Untuk Proses Deteksi Citra dengan YOLOV8 ===\n")
        # input jenis data
        data = input(f"Masukkan jenis data citra yang ingin diolah hasilnya (nonbg/bg/mix): ").strip()
        
        # Pastikan input valid
        if data not in {"nonbg", "bg", "mix"}:
            raise ValueError(f"Jenis data '{data}' tidak valid. Harus 'nonbg', 'bg', atau 'mix'.")# Initialize Logging function
        
        PROJECT_DIR = MAIN_DIR / data / f'hasil_{data}'
        LOG_PATH = MAIN_DIR / data / 'logging'
        MODELS_PATH = MAIN_DIR / 'models'
        
        
        # memilih salah satu input jenis model (biasa/kustom)
        model_type = input("Masukkan model yang ingin kamu gunakan (biasa/kustom): ").strip()
        if model_type not in {"biasa", "kustom"}:
            raise ValueError("Maaf, jenis model yang tersedia hanya 'biasa' atau 'kustom'.")

        # Pilih path model berdasarkan input
        if model_type == "biasa":
            size = input("Masukkan size model YOLOv8 (n/s/m/l/xl): ").strip()
            if size not in {"n", "s", "m", "l", "xl"}:
                raise ValueError(f"Size model '{size}' tidak valid.")
            model_path = MODELS_PATH / f"yolov8{size}.pt"

        else:  # model_type == "kustom"
            size = input("Masukkan size model kustom (s/m/l): ").strip()
            if size not in {"s", "m", "l"}:
                raise ValueError(f"Size model kustom '{size}' tidak valid.")
            model_path = MODELS_PATH / f"cbamc3_yolov8_rld_{size}.yaml"

        # Tentukan fold mode
        fold_mode = input("Ingin menjalankan semua fold atau fold tertentu saja? (all/nomor): ").strip()
        if fold_mode == "all":
            fold_range = range(1, TRAINING_CONFIG['n_folds'] + 1)
        elif fold_mode.isdigit() and 1 <= int(fold_mode) <= TRAINING_CONFIG['n_folds']:
            fold_range = [int(fold_mode)]
        else:
            raise ValueError("Input fold tidak valid. Masukkan 'all' atau angka fold yang valid.")
        
        # Pilihan untuk melanjutknan training dari checkpoint sebelumnya
        resume_choice = input("Apakah ingin melanjutkan training dari checkpoint sebelumnya? (y/n): ").strip().lower()
        resume_training = resume_choice == 'y'

        # Memulai inisialisasi trainer dan validator
        trainer = TrainerModel(device=device, project_dir=PROJECT_DIR, log_dir=LOG_PATH, data=data, model_type=model_type, size_model = size)
        validator = ValidatorModel(device=device, project_dir=PROJECT_DIR, log_dir= LOG_PATH, data=data, model_type=model_type, size_model = size)

        # Training loops
        for fold in fold_range:
            torch.cuda.empty_cache()
            yaml_path = DATASET_DIR / f'dataset_{data}' / 'fold' / f'fold_{fold}' / f'fold_{fold}_{data}_config.yaml'
            
            # Memulai inisialisasi Logging
            log_file = LOG_PATH / f'sum_fitting_model_{model_type}_{size}_{data}_{fold}.txt'
            logger = LoggerManager(log_file=log_file)
            logger.info(f"\n\nMemulai proses fitting model dengan jenis data: {data.upper()}\n")

            # Memulai inisialisasi Path YAML (data)
            try:
                with open(yaml_path, 'r') as f:
                    yaml_config = yaml.safe_load(f)
            except Exception as e:
                logger.error(f"Error reading YAML file: {e}")
            
            # Melakukan check dataset berdasarkan Path YAML
            check_dataset(yaml_path, logger)
            logger.info(f"Loaded YAML Configuration: {yaml_config}")

            try:
                # Memulai Fitting Model dengan memilih untuk menggunakan model dari epoch awal atau lanjutan
                WEIGHS_PATH = PROJECT_DIR / model_type / f"ukuran_{size}" / f"training_{fold}_{data}" / "train" / "weights"
                if resume_training:
                    checkpoint_path = get_checkpoint_path(WEIGHS_PATH)
                    logger.info(f"\n\nMelanjutkan training dari checkpoint: {checkpoint_path.name}")
                    model = YOLO(str(checkpoint_path), task='detect', verbose=True)
                else:
                    logger.info(f"\nInisialisasi model baru dari: {model_path}")

                model = YOLO(model=str(model_path), task='detect', verbose=True)
                logger.info(f"\nModel YOLO berhasil di-load untuk fold ke-{fold}.")

                torch.cuda.empty_cache()

                # =======================
                # Memulai Pelatihan Model
                # =======================
                start_time_train = time.time()
                start_time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                logger.info(f"\nTraining dimulai pada: {start_time_str}")

                trainer.train_fold(model, yaml_path, TRAINING_CONFIG, fold)

                end_time_train = time.time()
                end_time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                logger.info(f"\nTraining selesai pada: {end_time_str}")
                logger.info(f"\nDurasi total training: {(end_time_train - start_time_train)/60:.2f} menit")

                # ==============================
                # Memulai Validasi Pelatihan Model
                # ==============================
                torch.cuda.empty_cache()
                start_time_val = time.time()
                val_start_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                logger.info(f"\nValidating dimulai pada: {val_start_str}")

                validator.validation_data(model, yaml_path, TRAINING_CONFIG, fold)

                end_time_val = time.time()
                val_end_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                logger.info(f"\nValidating selesai pada: {val_end_str}")
                logger.info(f"\nDurasi total validating: {(end_time_val - start_time_val)/60:.2f} menit")

                torch.cuda.empty_cache()

            except Exception as e:
                logger.error(f"\nError processing fold {fold}:")
                logger.error(f"Error details: {str(e)}")
                logger.error(f"Fold {fold} failed: {str(e)}\n")
                continue
            
        logger.info(f"\nFitting dengan Model YOLOv8-{model_type}-{size} untuk jenis data {data.upper() } berhasil dengan sukses! \n")

    except Exception as e:
        logger.error(f"Critical error during training: {e}")
        logger.error("\nDetailed error traceback:")
        logger.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    main()