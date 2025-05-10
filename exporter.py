import torch, shutil, time
from pathlib import Path
from ultralytics import YOLO
from typing import Dict, Optional, Any

class ModelExporter:
    def __init__(self, best_path: Path, device: str, log_dir: Path, data: str, model_type:str, size_model:str):
        """
        Inisialisasi ModelExporter dengan direktori proyek dan konfigurasi perangkat.
        """
        self.best_path = best_path
        self.device = device
        self.data = data
        self.model_type = model_type
        self.size_model = size_model
        self.log_file = log_dir / f'export_{self.model_type}{self.size_model}_{self.data}.txt'

    def log_message(self, message: str):
            """Internal logging method"""
            timestamp = time.strftime('[%Y-%m-%d %H:%M:%S]')
            log_message = f"{timestamp} {message}"
            print(log_message)
            with open(self.log_file, 'a') as f:
                f.write(log_message + '\n')

    def validate_config(self, config: Dict[str, Any]) -> None:
        """
        Validasi konfigurasi ekspor.
        
        Args:
            config (Dict[str, Any]): Konfigurasi yang akan divalidasi
            
        Raises:
            ValueError: Jika konfigurasi tidak valid
        """
        required_keys = ['image_size', 'batch']
        
        # Cek keys yang diperlukan
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required key in config: {key}")
        
        # Validasi image_size
        if isinstance(config['image_size'], (list, tuple)):
            if len(config['image_size']) != 2:
                raise ValueError("Image size harus berupa tuple/list dengan 2 elemen")
            if any(not isinstance(size, int) or size <= 0 for size in config['image_size']):
                raise ValueError("Image size harus berupa integer positif")
        elif not isinstance(config['image_size'], int) or config['image_size'] <= 0:
            raise ValueError("Image size harus berupa integer positif")
            
        # Validasi batch_size
        if not isinstance(config['batch'], int) or config['batch'] <= 0:
            raise ValueError("Batch size harus berupa integer positif")
        
    def check_system_compatibility(self) -> None:
        """
        Periksa kompatibilitas sistem untuk ekspor.
        
        Raises:
            RuntimeError: Jika sistem tidak kompatibel
        """
        self.log_message("Memeriksa kompatibilitas sistem...")
        
        # Cek CUDA
        if self.device == 'cuda':
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA tidak tersedia di sistem ini")
            
            self.log_message(f"CUDA tersedia: {torch.cuda.get_device_name(0)}")
            self.log_message(f"Memory GPU tersedia: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    def export_tensorrt(self, model: YOLO, config: Dict[str, Any],current_fold: int,) -> Optional[Dict[str, str]]:
        """
        Mengekspor model ke format TensorRT.
        
        Args:
            model (YOLO): Model yang akan diekspor
            config (Dict[str, Any]): Konfigurasi ekspor
            current_fold (int): Fold/iterasi saat ini
        
        Returns:
            Optional[Dict[str, str]]: Hasil ekspor atau None jika gagal
        """
        try:
            # Validasi konfigurasi
            self.log_message(f"Memulai ekspor TensorRT dengan data {self.data}...")
            self.validate_config(config)
            self.check_system_compatibility()
            
            # Log konfigurasi
            self.log_message(f"Konfigurasi ekspor:")
            self.log_message(f"- Image size: {config['image_size']}")
            self.log_message(f"- Batch size: {config['batch']}")
            self.log_message(f"- Device: {self.device}")

            tensorrt_results = model.export(
                format="engine",
                imgsz=config['image_size'],
                dynamic=False,
                simplify=True,
                half=True,
                int8=False,
                device=self.device,
                workspace=8,
                nms=True,
                batch=config['batch']
            )
            # Debug hasil ekspor
            self.log_message(f"Export TensorRT results: {tensorrt_results}")\
            #Periksa hasil ekspor
            if isinstance(tensorrt_results, str):
                # Jika output berupa string, anggap itu adalah path file ONNX
                tensorrt_path = tensorrt_results
                self.log_message(f"Path TensorRT: {tensorrt_path}")
            elif isinstance(tensorrt_results, dict) and "engine" in tensorrt_results:
                # Jika output berupa dictionary, ambil path ONNX dari kunci 'onnx'
                tensorrt_path = tensorrt_results["engine"]
                self.log_message(f"Path TensorRT: {tensorrt_path}")
            else:
                raise ValueError("Unexpected output format from model.export() for TensorRT.")
            # Buat direktori ekspor
            export_dir = self.best_path / 'best_engine'
            export_dir.mkdir(parents=True, exist_ok=True)
            
            # Pindahkan file ke direktori yang sesuai
            tensorrt_dest = export_dir / f'model_{self.model_type}_{self.size_model}_fold_{current_fold}.engine'
            shutil.move(tensorrt_path, str(tensorrt_dest))
            
            # Log detail ekspor
            self.log_message(f"Path TensorRT: {tensorrt_path}")
            
            # Log hasil
            self.log_message(f"\nEkspor Model TensorRT berhasil:")
            self.log_message(f"Path TensorRT: {tensorrt_dest}")
        
            return {"engine": str(tensorrt_dest)}
        
        except Exception as e:
            self.log_message(f"Kesalahan saat ekspor model TensorRT: {str(e)}")
            return None

    def export_onnx(self, model: YOLO, config: Dict[str, Any],current_fold: int,) -> Optional[Dict[str, str]]:
        """
        Mengekspor model ke format ONNX.
        
        Args:
            model (YOLO): Model yang akan diekspor
            config (Dict[str, Any]): Konfigurasi ekspor
            current_fold (int): Fold/iterasi saat ini
        
        Returns:
            Optional[Dict[str, str]]: Hasil ekspor atau None jika gagal
        """
        try:
            # Validasi konfigurasi
            self.log_message(f"Memulai ekspor ONNX dengan data {self.data}...")
            self.validate_config(config)
            self.check_system_compatibility()

            onnx_results = model.export(
                format="onnx",
                imgsz=config['image_size'],
                dynamic=True,
                half=True,
                int8=False,
                device=self.device,
                workspace=8,
                nms=False,
                batch=config['batch']
            )
            # Debug hasil ekspor
            self.log_message(f"Export ONNX results: {onnx_results}")

            #Periksa hasil ekspor
            if isinstance(onnx_results, str):
                # Jika output berupa string, anggap itu adalah path file ONNX
                onnx_path = onnx_results
                self.log_message(f"Path ONNX: {onnx_path}")
            elif isinstance(onnx_results, dict) and "onnx" in onnx_results:
                # Jika output berupa dictionary, ambil path ONNX dari kunci 'onnx'
                onnx_path = onnx_results["onnx"]
                self.log_message(f"Path ONNX: {onnx_path}")
            else:
                raise ValueError("Unexpected output format from model.export() for ONNX.")
            
            # Buat direktori ekspor
            export_dir = self.best_path /'best_onnx'
            export_dir.mkdir(parents=True, exist_ok=True)
            
            # Pindahkan file ke direktori yang sesuai
            onnx_dest = export_dir / f'model_{self.model_type}_{self.size_model}_fold_{current_fold}.onnx'
            shutil.move(onnx_path, str(onnx_dest))
            
            # Log detail ekspor
            self.log_message(f"Path ONNX: {onnx_path}")
            
            # Log hasil
            self.log_message(f"\nEkspor Model ONNX berhasil:")
            self.log_message(f"Path ONNX: {onnx_dest}")
        
            return {"onnx": str(onnx_dest)}
        
        except Exception as e:
            self.log_message(f"Kesalahan saat ekspor model ONNX : {str(e)}")
            return None

def export_model(model: YOLO, config: Dict[str, Any],  device:str, best_path: Path, fold:int, log_dir:Path, data:str, model_type:str, size_model:str):
    """
    Fungsi utama untuk mengekspor model ke ONNX dan TensorRT.
    
    Args:
        model (YOLO): Model yang akan diekspor
        config (Dict[str, Any]): Konfigurasi ekspor
        fold (int): Fold/iterasi saat ini
        project_dir (str): Direktori proyek
        logger (Optional[Any], optional): Logger kustom. Defaults to None.
    
    Returns:
        bool: True jika ekspor berhasil, False jika gagal
    """
    try:
        exporter = ModelExporter(best_path, device, log_dir, data, model_type, size_model)

        # Ekspor ONNX
        onnx_path = exporter.export_onnx(model, config, fold)
        if not onnx_path:
            exporter.log_message("Gagal mengekspor model ke ONNX")
            return False

        exporter.log_message(f"Model berhasil diekspor ke ONNX: {onnx_path}")

        # Ekspor TensorRT
        engine_path = exporter.export_tensorrt(model, config, fold)
        if not engine_path:
            exporter.log_message("Gagal mengekspor model ke TensorRT")
            return False

        exporter.log_message(f"Model berhasil diekspor ke TensorRT: {engine_path}")
        return True

    except Exception as e:
        exporter.log_message(f"Error dalam proses ekspor: {str(e)}", exc_info=True)
        return False