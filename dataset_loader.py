import os, random, torch, logging
import numpy as np
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset

# Setup logging yang lebih terstruktur
class LoggerManager:
    def __init__(self, log_file):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        # Pastikan direktori tempat menyimpan log ada
        log_dir = Path(log_file).parent
        log_dir.mkdir(parents=True, exist_ok=True)

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

class YOLODataset(Dataset):
    def __init__(self, main_dir, classes, log_dir:Path, data:str):
        """
        Initialize YOLODataset with precise image-label matching and class-wise counting.
        Args:
            images_dir (str): Path to the directory with images.
            labels_dir (str): Path to the directory with labels.
            classes (list): List of class names.
        """
        self.main_dir = main_dir
        self.data = data
        self.images_dir = os.path.join(main_dir, f"gambar_{self.data}")
        self.labels_dir = os.path.join(main_dir, f"anotasi_{self.data}")
        self.classes = classes
        self.nc = len(self.classes)
        self.image_files = []
        self.all_labels = []
        self.valid_pairs = []
        self.class_counts = {cls: 0 for cls in classes}  # Count labels per class
        self.has_printed = set() # Menyimpan indeks yang sudah dicetak
        self.log_file = log_dir / f"loader_dataset_{self.data}.txt"
        self.logger  = LoggerManager(log_file=self.log_file)

        # Validate and match image-label pairs precisely, independent of class structure.
        for root, _, files in os.walk(self.images_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.JPG', '.PNG')):
                    img_rel_path = os.path.relpath(os.path.join(root, file), self.images_dir)
                    img_name = os.path.splitext(file)[0]
                    label_rel_path = os.path.join(os.path.dirname(img_rel_path), img_name + '.txt')
                    label_full_path = os.path.join(self.labels_dir, label_rel_path)
                    if os.path.exists(label_full_path):
                        self.valid_pairs.append((img_rel_path, label_rel_path))
                        self.image_files.append(img_rel_path)

        self.logger.info(f"\nTotal Pasangan gambar-label '{self.data.upper()}' yang valid ditemukan sebanyak: {len(self.valid_pairs)}")

        # Populate class counts by iterating through valid label files
        for _, label_file in self.valid_pairs:
            full_label_path = os.path.join(self.labels_dir, label_file)
            with open(full_label_path, 'r') as f:
                for line in f:
                    class_id = int(line.split()[0]) # Mendapatkan ID kelas dari file label
                    if 0 <= class_id < len(self.classes):
                        self.class_counts[self.classes[class_id]] += 1 #  Menambah jumlah   untuk kelas tertentu
                        self.all_labels.append(class_id)  # menyimpan ID kelas ke dalam daftar

        # Convert labels to numpy array for further processing
        self.all_labels = np.array(self.all_labels)

    def preprocess_image(self, image):
        """ Pra-pemrosesan gambar dengan penanganan channel """
        try:
            if isinstance(image, Image.Image):
                image = image.convert('RGB')  # melakukan konversi  ke RGB jika data belum terkonversi
                image = np.array(image, dtype=np.float32)

            # Pastikan gambar memiliki 3 channel
            if len(image.shape) == 2:
                image = np.stack([image, image, image], axis=-1)
            elif image.shape[-1] == 1:
                image = np.concatenate([image, image, image], axis=-1)
            elif image.shape[-1] == 4:  # Handle RGBA
                image = image[:, :, :3]

            image = image / 255.0 # menormalisasikan piksel menjadi nilai 0 - 1

            # Convert to PyTorch tensor and transpose dimensions
            image = torch.from_numpy(image).float()
            image = image.permute(2, 0, 1)  # Change from (H,W,C) to (C,H,W)

            return image

        except Exception as e:
            self.logger.error(f"Error in preprocess_image: {e}")
            raise
    
    def __len__(self):
        """Return total number of valid pairs."""
        return len(self.valid_pairs)
    
    def get_label(self, _, lbl_path):
        """
        Retrieve label(s) from the label file.
        Args:
            img_path: Path to the image file (relative or absolute).
            lbl_path: Path to the corresponding label file (relative or absolute).
        Returns:
            List of class IDs found in the label file.
        """
        try:
            full_label_path = os.path.join(self.labels_dir, lbl_path)
            if not os.path.exists(full_label_path):
                return []  # No label file exists for this image
            
            with open(full_label_path, 'r') as f:
                labels = [int(line.split()[0]) for line in f.readlines() if line.strip()]
            
            return labels
        except Exception as e:
            self.logger.error(f"Error retrieving labels for {lbl_path}: {e}")
            return []

    def __getitem__(self, idx):
        """ Get image and annotation at a specific index """
        if idx >= len(self.valid_pairs):
            raise IndexError(f"Index {idx} out of range")

        try:
            img_file, label_file = self.valid_pairs[idx]
            img_path = os.path.join(self.images_dir, img_file)
            label_path = os.path.join(self.labels_dir, label_file)

            # Load and Preprocess Image
            with Image.open(img_path) as image:
                processed_image = self.preprocess_image(image)

            boxes, labels = [], []
            with open(label_path, 'r') as file:
                for line in file:
                    data = line.strip().split()
                    if len(data) >= 5:
                        class_id = int(data[0]) # ID kelas objek
                        x_center, y_center, width, height = map(float, data[1:5])  # Koordinat bounding box
                        boxes.append([x_center, y_center, width, height])
                        labels.append(class_id)

            # Convert to PyTorch tensors
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)

            return processed_image, boxes, labels

        except Exception as e:
            self.logger.error(f"Error processing index {idx}: {e}")
            raise
    
class DatasetVerificator:
    def __init__(self, dataset, classes, log_dir:Path, data:str):
        """
        Initialize DatasetVisualizer with a dataset instance.
        Args:
            dataset (YOLODatasetnn): An instance of the YOLODatasetnn class.
        """
        self.data = data
        self.log_file = log_dir / f"loader_dataset_{self.data}.txt"
        self.logger = LoggerManager(log_file=self.log_file)
        self.dataset = dataset
        self.classes = classes if classes is not None else dataset.classes
        self.class_counts = {cls: 0 for cls in self.classes}

        self.logger.info(f"Debug: Input classes = {classes}")
        self.logger.info(f"Debug: Dataset classes = {dataset.classes}")
        self.logger.info(f"Debug: Final classes = {self.classes}")
        # Update class_counts dari dataset
        for cls, count in dataset.class_counts.items():
            if cls in self.class_counts:
                self.class_counts[cls] = count

    def display_sample(self, num_samples=1):
        """ Display a random sample of images with their bounding box annotations """
        self.logger.info(f"Total data {self.data} di dataset: {len(self.dataset)} data berkas gambar-label")
        sample_size = min(num_samples, len(self.dataset))
        random_indices = random.sample(range(len(self.dataset)), sample_size)

        self.logger.info(f"\nInformasi Sampel Random dari Dataset {self.data}:")
        self.logger.info("="*60)

        for i in random_indices:
            img_file, label_file = self.dataset.valid_pairs[i]
            img_path = os.path.join(self.dataset.images_dir, img_file)
            image, boxes, labels = self.dataset[i]

            with Image.open(img_path) as original_image:
                self.logger.info(f"Image file: {img_file}")
                self.logger.info(f"Label file: {label_file}")
                self.logger.info(f"Resolusi Gambar Orisinil: {original_image.size}")
                self.logger.info(f"Ukuran Gambar: {tuple(image.shape)}")
                self.logger.info(f"jumlah bounding boxes sampel: {len(boxes)}")
                if labels.size(0) > 0:
                    self.logger.info("Label-label BBoxes yang terdeteksi:")
                    for idx, (box, label) in enumerate(zip(boxes, labels)):
                        if label.item() != 0:  # Ignore padding labels
                            self.logger.info(f"  Box {idx + 1}: {self.dataset.classes[label.item()]} - Coordinates BBoxes: {box.tolist()}")
                else:
                    self.logger.info("No labels detected")
                self.logger.info("-"*60)
            
    def display_detailed_dataset_info(self):
        """Menampilkan informasi detail dataset termasuk jumlah gambar dan label per kelas"""
        self.logger.info("\nInformasi Detail Dataset:")
        self.logger.info("-" * 40)
        self.logger.info(f"Total jumlah data gambar-label: {len(self.dataset)}")
        
        self.logger.info("\nDistribusi jumlah label berdasarkan kelas:")
        for class_name, count in self.dataset.class_counts.items():
            self.logger.info(f"  Kelas '{class_name}' = '{count}' label kotak (bounding boxes).")

    def display_label_file_info(self):
        """Menampilkan informasi detail file label (txt)"""
        # Debug: Cetak path direktori label
        self.logger.info(f"\nPath Direktori Label: {self.dataset.labels_dir}")
        
        # Periksa apakah direktori label benar-benar ada
        if not os.path.exists(self.dataset.labels_dir):
            self.logger.error(f"PERINGATAN: Direktori label tidak ditemukan di {self.dataset.labels_dir}")
            return

        # Pemeriksaan isi direktori
        try:
            # List semua file di direktori label dengan penanganan error
            label_files = []
            for root, _, files in os.walk(self.dataset.labels_dir):
                txt_files = [f for f in files if f.endswith('.txt')]
                label_files.extend(txt_files)
                
            self.logger.info("\nInformasi File Label (txt):")
            self.logger.info("-" * 40)
            self.logger.info(f"Total jumlah file label (txt): {len(label_files)} (.txt).")
            
            # Menghitung jumlah file txt per subdirektori
            total_txt_files = 0
            label_subdirs = {}
            for root, _, files in os.walk(self.dataset.labels_dir):
                txt_files = [f for f in files if f.endswith('.txt')]
                total_txt_files+= len(txt_files)
                
                if txt_files:
                    subdir = os.path.relpath(root, self.dataset.labels_dir)
                    label_subdirs[subdir] = len(txt_files)
            
            self.logger.info("\nJumlah file label (txt) per subdirektori:")
            for subdir, count in label_subdirs.items():
                self.logger.info(f"  Subdirektori '{subdir}': {count} file (.txt).")

        except Exception as e:
            self.logger.error(f"Error saat membaca direktori label: {e}")

    def display_image_directory_info(self):
        """Menampilkan informasi detail direktori gambar"""
        image_files = []
        for root, _, files in os.walk(self.dataset.images_dir):
            image_files.extend([f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', 'JPEG', 'JPG', 'PNG'))])
        
        self.logger.info("\nInformasi Direktori Gambar:")
        self.logger.info("-" * 40)
        self.logger.info(f"Total jumlah file gambar: {len(image_files)} .")
        
        # Menghitung jumlah gambar per subdirektori
        image_subdirs = {}
        for root, _, files in os.walk(self.dataset.images_dir):
            img_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', 'JPEG', 'JPG', 'PNG'))]
            if img_files:
                subdir = os.path.relpath(root, self.dataset.images_dir)
                image_subdirs[subdir] = len(img_files)
        
        self.logger.info("\nJumlah file gambar per subdirektori:")
        for subdir, count in image_subdirs.items():
            self.logger.info(f"  Subdirektori '{subdir}': {count} file.")

# Config
DATASET_DIR = Path("D:/Riset Skripsi/dataset_skripsi")
MAIN_DIR = Path("D:/Riset Skripsi/script riset/deteksi_citra")

# Example usage
def main():
    try:
        # Running Program
        print("\n=== Memulai Loader Dataset Untuk Proses Deteksi Citra dengan YOLOV8 ===\n")
        data = input(f"Masukkan jenis data citra yang ingin diolah hasilnya (nonbg/bg/mix): ").strip()
        
        # Pastikan input valid
        if data not in {"nonbg", "bg", "mix"}:
            raise ValueError(f"Jenis data '{data}' tidak valid. Harus 'nonbg', 'bg', atau 'mix'.")# Initialize Logging function
        
        main_dir = DATASET_DIR /f"dataset_{data}"
        LOG_PATH = MAIN_DIR / data / 'logging'
        
        classes = ["bercak cokelat",
            "bercak cokelat tipis",
            "blas daun",
            "lepuh daun",
            "hawar daun bakteri",
            "sehat"]  # Replace with actual class names
        
        dataset = YOLODataset(main_dir, classes, LOG_PATH, data)
        visualizer = DatasetVerificator(dataset, classes, LOG_PATH, data)
        
        # Tampilkan detail dataset
        visualizer.display_detailed_dataset_info()
        
        # Tampilkan informasi file label
        visualizer.display_label_file_info()
        
        # Tampilkan informasi direktori gambar
        visualizer.display_image_directory_info()
        # Tampilkan informasi sampel random dari dataset
        visualizer.display_sample(1)
        
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()