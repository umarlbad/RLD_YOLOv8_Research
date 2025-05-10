import torch, os, traceback, logging, json, time
from datetime import datetime
from pathlib import Path
from typing import Union, Dict
from ultralytics import YOLO
from validator import ValidatorModel
from predictor import PredictModel, TRTPredictModel
from exporter import export_model, ModelExporter
from utilisasi import VisualisasiDatasetPred

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

EVALUATE_CONFIG = {
    'batch':1, # batch size untuk ekspor model = Menentukan ukuran inferensi batch model ekspor atau jumlah maksimum gambar yang akan diproses model yang diekspor secara bersamaan dalam mode prediksi.
    'image_size': (640,640),
    'max_det':200,
    'n_folds': 5,
    'conf': 0.35,
    'iou':0.50,
    'line_width':3
}

VALIDATE_CONFIG = {
    'epochs':150,
    'batch_size':16,
    'iou':0.5,
    'conf':0.35,
    'n_folds':5,
}

class LoggerManager:
    def __init__(self, log_file):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
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

def parse_metrics(metrics: Union[str, Dict]) -> Dict:
    """
    Parse metrics whether they're in string or dictionary format
    """
    if isinstance(metrics, str):
        try:
            return json.loads(metrics)
        except json.JSONDecodeError:
            # If it's not JSON, try to parse it as a simple key-value string
            metrics_dict = {}
            try:
                pairs = metrics.strip('{}').split(',')
                for pair in pairs:
                    key, value = pair.split(':')
                    key = key.strip().strip("'\"")
                    value = float(value.strip())
                    metrics_dict[key] = value
                return metrics_dict
            except:
                return {}
    elif isinstance(metrics, dict):
        return metrics
    return {}

def save_evaluate_summary(successful_folds: int, all_metrics: list, logger) -> None:
    """
    Log evaluate summary into the existing logger instead of writing to a separate file.
    """
    logger.info("\n==================== RINGKASAN AKHIR EVALUASI ====================\n")

    if successful_folds > 0:
        summary_keys = ['class', 'confidence', 'bbox']
        metriks_parse = [parse_metrics(m) for m in all_metrics]

        try:
            summary = {'successful_folds': successful_folds}
            for key in summary_keys:
                values = [m.get(key, 0) for m in metriks_parse if key in m]
                summary[f'average_{key}'] = sum(values) / len(values) if values else 0

            for key, value in summary.items():
                logger.info(f"{key}: {value}")

            logger.info("\n============= Summary evaluate metrics saved successfully =============\n")
        except Exception as e:
            logger.error(f"Error processing metrics: {str(e)}")
    else:
        logger.error("Tidak ada fold yang berhasil diselesaikan.")


def main():
    try:
        # Running Program
        print("\n=== Memulai Evaluasi dan Prediksi Model Untuk Proses Deteksi Citra dengan YOLOV8 ===\n")
        data = input(f"Masukkan jenis data citra yang ingin diolah hasilnya (nonbg/bg/mix): ").strip()
        successful_folds = 0
        
        # Pastikan input valid
        if data not in {"nonbg", "bg", "mix"}:
            raise ValueError(f"Jenis data '{data}' tidak valid. Harus 'nonbg', 'bg', atau 'mix'.")# Initialize Logging function
        
        PROJECT_DIR = MAIN_DIR / data / f'hasil_{data}'
        LOG_PATH = MAIN_DIR / data / 'logging'
        uji_path = DATASET_DIR / f'dataset_{data}' / 'fold' /'test' / 'images' #set data uji
        anotasi_path = DATASET_DIR / f'dataset_{data}' / 'fold' /'test' / 'labels'
        
        # Validasi keberadaan file Data Testing
        if not uji_path.exists():
            raise FileNotFoundError(f"Set data uji tidak ditemukan di path : {uji_path}")
        elif not anotasi_path.exists():
            raise FileNotFoundError(f"Set data uji tidak ditemukan di path : {anotasi_path}")

        # memilih salah satu
        # Input jenis model (biasa/kustom)
        model_type = input("Masukkan hasil model yang ingin kamu gunakan (biasa/kustom): ").strip()
        if model_type not in {"biasa", "kustom"}:
            raise ValueError("Maaf, jenis model yang tersedia hanya 'biasa' atau 'kustom'.")

        # Tentukan fold mode
        fold_mode = input("Ingin menjalankan semua fold atau fold tertentu saja? (all/nomor): ").strip()
        if fold_mode == "all":
            fold_range = range(1, EVALUATE_CONFIG['n_folds'] + 1)
        elif fold_mode.isdigit() and 1 <= int(fold_mode) <= EVALUATE_CONFIG['n_folds']:
            fold_range = [int(fold_mode)]
        else:
            raise ValueError("Input fold tidak valid. Masukkan 'all' atau angka fold yang valid.")
        
        # Melakukan Validasi Model
        validasi_choice = input("Apakah ingin melakukan validasi model sebelum prediksi? (y/n): ").strip().lower()
        do_validation = validasi_choice == 'y'

        # Training loops
        for fold in fold_range:
            torch.cuda.empty_cache()
            try:
                # Pilih path hasil model fitting berdasarkan input
                size = input("\nMasukkan size hasil model (n/s/m/l/xl untuk biasa, s/m/l untuk kustom): ").strip()
                valid_sizes = {"n", "s", "m", "l", "xl"} if model_type == "biasa" else {"s", "m", "l"}
                if size not in valid_sizes:
                    raise ValueError(f"Size model '{size}' tidak valid untuk model {model_type}.")
                
                # Initialize Logging function
                log_file = LOG_PATH / f'evaluatemodel_{fold}_{model_type}_{size}_{data}.txt'

                logger = LoggerManager(log_file=log_file)
                logger.debug(f"\n\nMemulai proses Evaluate Model dengan jenis data: {data.upper()}\n")

                # Inisialisasi path variabel-variabel penyokong data testing
                best_path = PROJECT_DIR / model_type / f"ukuran_{size}" / f"training_{fold}_{data}" / "train" / "weights" / "best.pt"
                yaml_path = DATASET_DIR / f'dataset_{data}' / 'fold' / f'fold_{fold}' / f'fold_{fold}_{data}_config.yaml'
                pred_path = PROJECT_DIR / f'testing_{fold}_{model_type}_{size}' / 'predict' #path hasil gambar prediksi
                output_path = MAIN_DIR / data / 'gambar' / f'comp_{fold}_{model_type}_{size}'
                EXPORT_PATH = MAIN_DIR / data / "checkpoints"
                engine_path =EXPORT_PATH / "best_engine"

                # Mengunggah Model
                model = YOLO(best_path, task="detect", verbose=True)
                logger.debug(f"Model YOLO {model_type}-{size} berhasil diunggah untuk fold ke-{fold}.")

                # Initialize Modul-Modul
                validator = ValidatorModel(device=device, project_dir=PROJECT_DIR, log_dir= LOG_PATH, data=data, model_type=model_type, size_model = size)
                predictor = PredictModel(device, PROJECT_DIR, CLASSES, LOG_PATH, data, model_type, size)
                predictor_trt = TRTPredictModel(device, PROJECT_DIR, CLASSES, LOG_PATH, data, model_type, size)

                # Memulai Validasi Pelatihan Model
                if do_validation:
                    logger.info("\n=== Memulai Validasi Model ===")
                    torch.cuda.empty_cache()
                    start_time_val = time.time()
                    val_start_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    logger.info(f"Memulai validasi model dengan model: \n{best_path}\n")
                    logger.info(f"Validating dimulai pada: {val_start_str}")
                    validator.validation_data(model, yaml_path, VALIDATE_CONFIG, fold)
                    val_duration = (time.time() - start_time_val) / 60
                    val_end_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    logger.info(f"\nValidating selesai pada: {val_end_str}")
                    logger.info(f"Validasi selesai dalam {val_duration:.2f} menit.\n")
                    
                    torch.cuda.empty_cache()
                else:
                    logger.info("\n=== Melewati Validasi Model ===")

                logger.info(f"\nMemulai Prediksi dengan data uji {data.upper()}...")
                predictor.predict_model(uji_path, model, config=EVALUATE_CONFIG, current_fold=fold)
                successful_folds += 1
                
                torch.cuda.empty_cache()
                
                # Ekspor model
                torch.cuda.empty_cache()
                export_choice = input(f"Apakah ingin melakukan ekspor model? (y/n)").strip().lower()
                if export_choice =="y":    
                    logger.info(f"\nMemulai Ekspor model dengan hasil model dari {model_type} ...")
                    export_results = export_model(model, EVALUATE_CONFIG, device, EXPORT_PATH, fold, LOG_PATH, data, model_type, size)
                    if export_results:
                        logger.info("Ekspor model selesai dengan sukses!")
                    else:
                        logger.error("Ekspor model gagal.")
                else:
                    logger.error("Proses ekspor model dilewati, lanjut prediksi dengan TensorRT/Evaluasi Selesai. \n")
                
                # Prediksi dengan Model Engine TensorRT
                tensor_choice = input(f"Apakah ingin melakukan Prediksi dengan TensorRT? (y/n)").strip().lower()
                if tensor_choice == "y":
                    logger.info(f"\n=== Memulai Inferensi dengan TensorRT Fold ke-{fold} ===")
                    predictor_trt.predict_trt(uji_path, engine_path, config=EVALUATE_CONFIG, current_fold=fold)
                else:
                    logger.info(f"\nProses Prediksi dengan TensorRT dilewati.")

            except Exception as e:
                logger.error(f"\nError processing fold {fold}:")
                logger.error(f"Error details: {str(e)}")
                logger.error(f"Fold {fold} failed: {str(e)}\n")
                continue
            
        torch.cuda.empty_cache()
        
        if successful_folds > 0:
            logger.info("\n===== Memulai untuk Membuat Komparasi Gambar Asli dengan Gambar Prediksi =====")
            visualizer = VisualisasiDatasetPred(uji_path, pred_path, anotasi_path, output_path, data, model_type, size, fold)
            visualizer.visualize_side_by_side()
            visualizer.visualize_overlay()
            logger.info(f"\n======= Proses Evaluasi Model dengan Fold ke-{fold} telah selesai dilakukan =======\n")
        else:
            logger.error("Tidak ada fold yang berhasil dijalankan. Visualisasi tidak dilakukan.")

    except Exception as e:
        logger.error(f"Critical error during evaluating: {e}")
        logger.error("\nDetailed error traceback:")
        logger.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    main()