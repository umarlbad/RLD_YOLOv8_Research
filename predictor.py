import torch, os, time
from pathlib import Path
from typing import List, Dict, Any
from ultralytics import YOLO
from collections import Counter

class PredictModel:
    def __init__(self, device: str, project_dir: str, classes: List[str], log_dir:Path, data : str, model_type:str, size_model:str):
        """
        Inisialisasi kelas PredictModel

        Parameter:
        - device: Perangkat komputasi (CPU/GPU)
        - project_dir: Direktori proyek untuk menyimpan hasil
        - classes: Daftar kelas yang akan diprediksi
        """
        self.device = device
        self.data = data
        self.model_type = model_type
        self.size_model = size_model
        self.project_dir = Path(project_dir)
        self.classes = classes
        self.log_file = log_dir / f'predict_{self.data}_{self.model_type}_{self.size_model}.txt'
        self.supported_formats = {'.jpg','.JPG' ,'.jpeg', '.png', '.PNG','.bmp', '.tif', '.tiff'}
        self.nonbg_ranges = {
            "bercak cokelat": (1, 600),
            "bercak cokelat tipis": (601, 1200),
            "blas daun": (1201, 1800),
            "lepuh daun": (2401, 3000),
            "hawar daun bakteri": (1081, 2400),
            "sehat": (3001, 3600)
        }
        self.bg_ranges = {
            "bercak cokelat": (3601, 4200),
            "bercak cokelat tipis": (4201, 4800),
            "blas daun": (4801, 5400),
            "lepuh daun": (6001, 6600),
            "hawar daun bakteri": (5401, 6000),
            "sehat": (6600, 7200)
        }
        self.mix_ranges = {
            "bercak cokelat": (1, 1200),
            "bercak cokelat tipis": (1201, 2400),
            "blas daun": (2401, 3600),
            "lepuh daun": (4801, 6000),
            "hawar daun bakteri": (3601, 4800),
            "sehat": (6001, 7200)
        }

    def get_class_from_id(self, img_id: int) -> str:
        """
        Mendapatkan nama kelas berdasarkan ID gambar dari bg_ranges, nobg_ranges, atau mix_ranges.
        
        Parameter:
        - img_id: ID gambar (angka dari nama file pdp{number})
        
        Return:
        - Nama kelas atau None jika tidak ditemukan
        """
        
        # Prioritas pertama: bg_ranges
        if self.data == "bg":
            for class_name, (start, end) in self.bg_ranges.items():
                if start <= img_id <= end:
                    return class_name
        
        # Prioritas kedua: nobg_ranges
        elif self.data == "nonbg":
            for class_name, (start, end) in self.nonbg_ranges.items():
                if start <= img_id <= end:
                    return class_name
            
        # Prioritas ketiga: mix_ranges
        elif self.data == "mix":
            for class_name, (start, end) in self.mix_ranges.items():
                if start <= img_id <= end:
                    return class_name
            
        return None

    def log_message(self, message: str):
        """Internal logging method"""
        timestamp = time.strftime('[%Y-%m-%d %H:%M:%S]')
        log_message = f"{timestamp} {message}"
        print(log_message)
        with open(self.log_file, 'a') as f:
            f.write(log_message + '\n')

    def init_predict(self, test_path: Path, current_fold: int):
        pred_dir = self.project_dir / self.model_type / f"ukuran_{self.size_model}" / f"testing_{current_fold}"
        pred_dir.mkdir(parents=True, exist_ok=True)

        if not test_path.exists():
            raise FileNotFoundError(f"Path data testing tidak ditemukan: {test_path}")

        image_files = sorted(
            [f for f in test_path.iterdir() if f.suffix.lower() in self.supported_formats],
            key=lambda x: int(x.stem.replace("pdp", ""))
        )
        if not image_files:
            raise ValueError("Tidak ada file gambar yang ditemukan")

        return pred_dir, image_files
    
    def predict_model(self, test_path: Path, model: YOLO, config: Dict, current_fold: int):
        try:
            self.log_message(f"\n\nMemulai Prediksi Gambar untuk Fold-{current_fold}... \n")
            pred_dir, image_files = self.init_predict(test_path, current_fold)

            results_data = []
            with torch.no_grad():
                for idx, img_path in enumerate(image_files, start=1):
                    self.log_message(f"Memprediksi Gambar ke-{idx} dari total {len(image_files)}: {img_path}.")
                    file_name = img_path.stem
                    img_id = int(file_name.replace('pdp', ''))
                    true_class = self.get_class_from_id(img_id)
                    if not true_class:
                        continue

                    results = list(model.predict(
                        source=str(img_path),
                        device=self.device,
                        conf=config['conf'],
                        iou=config['iou'],
                        max_det=config['max_det'],
                        save=True,
                        save_conf=True,
                        save_txt=True,
                        project=str(pred_dir),
                        line_width=config['line_width'],
                        exist_ok=True,
                        stream=True,
                        plots=True,
                        verbose=True
                    ))

                    results_data.append((img_path, results, true_class))

            utils_results = self.utils_class(results_data)
            
            return results_data, utils_results

        except Exception as e:
            raise RuntimeError(f"Error selama proses prediksi: {str(e)}") from e
        
    def utils_class(self, results_data: List[tuple]) -> Dict[str, Any]:
        max_detections_per_class = 1
        total_predictions = 0
        correct_predictions = 0
        total_preprocess_time = 0
        total_inference_time = 0
        total_postprocess_time = 0

        true_labels, pred_labels = [], []
        class_metrics = {
            cls: {'TP': 0, 'FP': 0, 'FN': 0, 'confidence_sum': 0, 'count': 0}
            for cls in self.classes
        }

        expected_class_distribution = {cls: 0 for cls in self.classes}

        # Hitung distribusi kelas ground truth berdasarkan ID gambar
        for img_path, _, _ in results_data:
            img_id = int(img_path.stem.replace('pdp', ''))
            true_class = self.get_class_from_id(img_id)
            if true_class:
                expected_class_distribution[true_class] += 1

        # Proses setiap gambar
        for idx, (img_path, results, _) in enumerate(results_data, 1):
            file_name = img_path.stem
            img_id = int(file_name.replace('pdp', ''))
            true_class = self.get_class_from_id(img_id)

            if not true_class:
                self.log_message(f"\nTidak bisa menentukan kelas ground truth untuk {img_path}. \n")
                continue

            if not results or len(results[0].boxes) == 0:
                # Tidak ada deteksi
                true_labels.append(true_class)
                pred_labels.append(None)
                class_metrics[true_class]['FN'] += 1
                total_predictions += 1
                continue

            result = results[0]
            preprocess_time = result.speed.get("preprocess", 0)
            inference_time = result.speed.get("inference", 0)
            postprocess_time = result.speed.get("postprocess", 0)

            # Ambil deteksi dengan confidence tertinggi (setelah filter)
            detections = sorted([
                {
                    'class': self.classes[int(box.cls[0])],
                    'confidence': float(box.conf[0]),
                    'bbox': box.xyxy[0].tolist()
                } for box in result.boxes
            ], key=lambda x: x['confidence'], reverse=True)

            # Ambil 1 deteksi terbaik per kelas
            filtered = []
            class_counts = {}
            for det in detections:
                cls = det['class']
                if class_counts.get(cls, 0) < max_detections_per_class:
                    filtered.append(det)
                    class_counts[cls] = class_counts.get(cls, 0) + 1

            if not filtered:
                true_labels.append(true_class)
                pred_labels.append(None)
                class_metrics[true_class]['FN'] += 1
                total_predictions += 1
                continue

            best_det = max(filtered, key=lambda x: x['confidence'])
            pred_class = best_det['class']
            confidence = best_det['confidence']

            true_labels.append(true_class)
            pred_labels.append(pred_class)
            total_predictions += 1

            # Update metrik per kelas
            class_metrics[pred_class]['confidence_sum'] += confidence
            class_metrics[pred_class]['count'] += 1

            if pred_class == true_class:
                correct_predictions += 1
                class_metrics[pred_class]['TP'] += 1
            else:
                self.log_message(f"YOLO salah deteksi gambar {img_path} sebagai {pred_class} (seharusnya {true_class}) dengan confidence {confidence:.3f}. \n")
                class_metrics[pred_class]['FP'] += 1
                class_metrics[true_class]['FN'] += 1

            self.log_message(f"Deteksi gambar ke-{idx} {img_path}: {filtered}")
            self.log_message(f"Kelas sebenarnya: {true_class}")
            self.log_message(f"Waktu: Preprocess: {preprocess_time:.2f}ms, Inference: {inference_time:.2f}ms, Postprocess: {postprocess_time:.2f}ms\n")

            total_preprocess_time += preprocess_time
            total_inference_time += inference_time
            total_postprocess_time += postprocess_time

        # Ringkasan metrik
        avg_pre = total_preprocess_time / total_predictions if total_predictions else 0
        avg_inf = total_inference_time / total_predictions if total_predictions else 0
        avg_post = total_postprocess_time / total_predictions if total_predictions else 0
        accuracy = correct_predictions / total_predictions if total_predictions else 0

        self.log_message(f"\nTotal gambar dari data {self.data.upper()} yang terprediksi dengan Laptop: {total_predictions}")
        self.log_message(f"Prediksi Gambar benar : {correct_predictions}")
        self.log_message(f"Akurasi Klasifikasi Jumlah Gambar: {accuracy:.3f}")
        self.log_message(f"Rata-rata waktu proses Deteksi: \nPreprocess: {avg_pre:.2f}ms, Inference: {avg_inf:.2f}ms, Postprocess: {avg_post:.2f}ms.")

        # Metrik per kelas
        for cls in self.classes:
            TP = class_metrics[cls]['TP']
            FP = class_metrics[cls]['FP']
            FN = class_metrics[cls]['FN']
            count = class_metrics[cls]['count']
            conf_sum = class_metrics[cls]['confidence_sum']
            precision = TP / (TP + FP) if TP + FP > 0 else 0
            recall = TP / (TP + FN) if TP + FN > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
            avg_conf = conf_sum / count if count > 0 else 0

            self.log_message(f"Kelas: {cls}\n  Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}, Avg. Conf: {avg_conf:.3f}")

        # Analisis keseimbangan
        balance_info = {}
        _ = Counter(true_labels)
        pred_count = Counter(pred_labels)

        self.log_message("\n\nAnalisis Keseimbangan Prediksi:")
        for cls in self.classes:
            expected = expected_class_distribution.get(cls, 0)
            pred = pred_count.get(cls, 0)
            is_balanced = abs(pred - expected) <= max(1, 0.001 * expected)
            if not is_balanced:
                self.log_message(f"[WARNING] Ketidakseimbangan terdeteksi pada kelas: {cls}. \n")

            balance_info[cls] = {
                "expected": expected,
                "pred": pred,
                "is_balanced": is_balanced
            }
            self.log_message(f"  {cls}: Expected = {expected}, Predicted = {pred}, Balanced = {is_balanced}")

        return {
            'total_images': total_predictions,
            'correct_predictions': correct_predictions,
            'prediction_accuracy': accuracy,
            'average_preprocess_time': avg_pre,
            'average_inference_time': avg_inf,
            'average_postprocess_time': avg_post,
            'class_metrics': class_metrics,
            'class_balance': balance_info,
            'expected_distribution': expected_class_distribution
        }
    
class TRTPredictModel:
    def __init__(self, device, project_dir, classes:List[str], log_dir:Path, data: str, model_type:str, size_model:str):
        """
        Kelas untuk melakukan prediksi menggunakan TensorRT Engine
        """
        self.device = device
        self.classes = classes
        self.data = data
        self.model_type = model_type
        self.size_model = size_model
        self.project_dir = Path(project_dir)
        self.log_file = log_dir / f"predicttrt_{self.data}_{self.model_type}_{self.size_model}_log.txt"
        self.supported_formats = {'.jpg','.JPG', '.jpeg', '.png','.PNG', '.bmp', '.tif', '.tiff'}
        self.nonbg_ranges = {
            "bercak cokelat": (1, 600),
            "bercak cokelat tipis": (601, 1200),
            "blas daun": (1201, 1800),
            "lepuh daun": (2401, 3000),
            "hawar daun bakteri": (1081, 2400),
            "sehat": (3001, 3600)
        }
        self.bg_ranges = {
            "bercak cokelat": (3601, 4200),
            "bercak cokelat tipis": (4201, 4800),
            "blas daun": (4801, 5400),
            "lepuh daun": (6001, 6600),
            "hawar daun bakteri": (5401, 6000),
            "sehat": (6600, 7200)
        }
        self.mix_ranges = {
            "bercak cokelat": (1, 1200),
            "bercak cokelat tipis": (1201, 2400),
            "blas daun": (2401, 3600),
            "lepuh daun": (4801, 6000),
            "hawar daun bakteri": (3601, 4800),
            "sehat": (6001, 7200)
        }

    def get_class_from_id(self, img_id: int) -> str:
        """
        Mendapatkan nama kelas berdasarkan ID gambar dari bg_ranges, nobg_ranges, atau mix_ranges.
        
        Parameter:
        - img_id: ID gambar (angka dari nama file pdp{number})
        
        Return:
        - Nama kelas atau None jika tidak ditemukan
        """
        
        # Prioritas pertama: bg_ranges
        if self.data == "bg":
            for class_name, (start, end) in self.bg_ranges.items():
                if start <= img_id <= end:
                    return class_name
        
        # Prioritas kedua: nobg_ranges
        elif self.data == "nonbg":
            for class_name, (start, end) in self.nonbg_ranges.items():
                if start <= img_id <= end:
                    return class_name
            
        # Prioritas ketiga: mix_ranges
        elif self.data == "mix":
            for class_name, (start, end) in self.mix_ranges.items():
                if start <= img_id <= end:
                    return class_name
            
        return None
    
    def log_message(self, message: str):
        """Simple logging function"""
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        log_entry = f"[{timestamp}] {message}\n"
        print(log_entry.strip())
        with open(self.log_file, 'a') as f:
            f.write(log_entry)
            
    def init_predict(self, test_path: Path, current_fold: int):
            pred_dir = self.project_dir / self.model_type / f"ukuran_{self.size_model}" / f"testing_trt_{current_fold}"
            pred_dir.mkdir(parents=True, exist_ok=True)

            if not test_path.exists():
                raise FileNotFoundError(f"Path data testing tidak ditemukan: {test_path}")

            image_files = sorted(
                [f for f in test_path.iterdir() if f.suffix.lower() in self.supported_formats],
                key=lambda x: int(x.stem.replace("pdp", ""))
            )
            if not image_files:
                raise ValueError("Tidak ada file gambar yang ditemukan")

            return pred_dir, image_files
    
    def predict_trt(self, test_path: Path, engine_path:Path, config: Dict, current_fold: int) -> Dict[str, Any]:
        try:
            self.log_message(f"\n\nMemulai Prediksi Gambar dengan TensorRT untuk Fold-{current_fold}... \n")
            pred_dir, image_files = self.init_predict(test_path, current_fold)

            # Menginisialisasi Model Engine untuk diprediksi data uji tersebut
            model_trt = engine_path / f"model_{self.model_type}_{self.size_model}_fold_{current_fold}.engine"
            model = YOLO(str(model_trt), task="detect", verbose=True)

            results_data = []
            with torch.no_grad():
                for idx, img_path in enumerate(image_files, start=1):
                    self.log_message(f"\nMemprediksi Gambar ke-{idx} dari {len(image_files)}: {img_path} \n")
                    file_name = img_path.stem
                    img_id = int(file_name.replace('pdp', ''))
                    true_class = self.get_class_from_id(img_id)
                    if not true_class:
                        continue

                    results = list(model.predict(
                        source=str(img_path),
                        device=self.device,
                        conf=config['conf'],
                        iou=config['iou'],
                        max_det=config['max_det'],
                        save=True,
                        save_conf=True,
                        save_txt=True,
                        project=str(pred_dir),
                        line_width=config['line_width'],
                        exist_ok=True,
                        stream=True,
                        plots=True,
                        verbose=True
                    ))

                    results_data.append((img_path, results, true_class))
            utils_results = self.utils_class(results_data)

            return results_data, utils_results

        except Exception as e:
            raise RuntimeError(f"Error selama proses prediksi dengan TensorRT: {str(e)}") from e
        
    def utils_class(self, results_data: List[tuple]) -> Dict[str, Any]:
        from collections import Counter

        max_detections_per_class = 1
        total_predictions = 0
        correct_predictions = 0
        total_preprocess_time = 0
        total_inference_time = 0
        total_postprocess_time = 0
        num_images = 0
        class_metrics = {cls: {'total': 0, 'correct': 0, 'confidence_sum': 0, 'TP': 0, 'FP': 0, 'FN': 0} for cls in self.classes}
        true_labels, pred_labels = [], []

        for img_path, results, true_class in results_data:
            if not results or len(results[0].boxes) == 0:
                self.log_message(f"\n\nTidak ada hasil yang didapatkan dari {img_path}\n\n")
                continue

            result = results[0]
            detections = sorted([
                {
                    'class': self.classes[int(box.cls[0])],
                    'confidence': float(box.conf[0]),
                    'bbox': box.xyxy[0].tolist()
                } for box in result.boxes
            ], key=lambda x: x['confidence'], reverse=True)

            filtered_detections, class_counts = [], {}
            for det in detections:
                pred_class = det['class']
                if class_counts.get(pred_class, 0) < max_detections_per_class:
                    filtered_detections.append(det)
                    class_counts[pred_class] = class_counts.get(pred_class, 0) + 1

            preprocess_time = result.speed.get("preprocess", 0)
            inference_time = result.speed.get("inference", 0)
            postprocess_time = result.speed.get("postprocess", 0)

            if filtered_detections:
                best_detection = max(filtered_detections, key=lambda x: x['confidence'])
                pred_class = best_detection['class']
                true_labels.append(true_class)
                pred_labels.append(pred_class)

                class_metrics[pred_class]['total'] += 1
                class_metrics[pred_class]['confidence_sum'] += best_detection['confidence']
                if pred_class == true_class:
                    correct_predictions += 1
                    class_metrics[pred_class]['correct'] += 1
                    class_metrics[pred_class]['TP'] += 1
                else:
                    class_metrics[pred_class]['FP'] += 1
                    class_metrics[true_class]['FN'] += 1

            total_predictions += 1
            total_inference_time += inference_time
            total_preprocess_time += preprocess_time
            total_postprocess_time += postprocess_time
            num_images += 1

            self.log_message(f"\n\nDeteksi (setelah filtering) dengan TensorRT dalam gambar {img_path}: {filtered_detections}")
            self.log_message(f"Gambar {img_path}: Preprocess: {preprocess_time:.2f}ms, Inference: {inference_time:.2f}ms, Postprocess: {postprocess_time:.2f}ms. \n")

        # Log keseluruhan
        avg_pre = total_preprocess_time / num_images if num_images else 0
        avg_inf = total_inference_time / num_images if num_images else 0
        avg_post = total_postprocess_time / num_images if num_images else 0
        accuracy = correct_predictions / total_predictions if total_predictions else 0

        self.log_message(f"\nHasil Prediksi Data {self.data} dengan TensorRT Laptop:")
        self.log_message(f"Rata-rata waktu per gambar:\nPreprocess: {avg_pre:.2f}ms,\nInference: {avg_inf:.2f}ms,\nPostprocess: {avg_post:.2f}ms")
        self.log_message(f"\nTotal gambar diproses: {total_predictions}")
        self.log_message(f"Prediksi benar: {correct_predictions}")
        self.log_message(f"Akurasi klasifikasi Gambar yang diproses: {accuracy:.3f}")
        self.log_message(f"\nKlasifikasi Jumlah Gambar {self.data} yang terdeteksi per kelasnya:")

        for cls in self.classes:
            total = class_metrics[cls]['total']
            correct = class_metrics[cls]['correct']
            conf_avg = class_metrics[cls]['confidence_sum'] / total if total > 0 else 0
            cls_accuracy = correct / total if total > 0 else 0
            precision = class_metrics[cls]['TP'] / (class_metrics[cls]['TP'] + class_metrics[cls]['FP']) if (class_metrics[cls]['TP'] + class_metrics[cls]['FP']) > 0 else 0
            recall = class_metrics[cls]['TP'] / (class_metrics[cls]['TP'] + class_metrics[cls]['FN']) if (class_metrics[cls]['TP'] + class_metrics[cls]['FN']) > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            self.log_message(f"kelas {cls}: akurasi kelas = {cls_accuracy:.3f}, Conf = {conf_avg:.3f} ({correct}/{total})\n Precision: {precision:.3f}, Recall: {recall:.3f}, F1-score: {f1_score:.3f}")

        # Evaluasi keseimbangan
        true_count = Counter(true_labels)
        pred_count = Counter(pred_labels)
        class_balance = {}
        self.log_message("\nAnalisis Keseimbangan Prediksi:")
        for cls in self.classes:
            true_val = true_count.get(cls, 0)
            pred_val = pred_count.get(cls, 0)
            is_balanced = abs(true_val - pred_val) <= max(1, 0.1 * true_val)
            class_balance[cls] = {
                "true": true_val,
                "pred": pred_val,
                "is_balanced": is_balanced
            }
            self.log_message(f"Kelas {cls}: True = {true_val}, Pred = {pred_val}, Seimbang = {is_balanced}")

        return {
            'total_images': total_predictions,
            'correct_predictions': correct_predictions,
            'prediction_accuracy': accuracy,
            'average_preprocess_time': avg_pre,
            'average_inference_time': avg_inf,
            'average_postprocess_time': avg_post,
            'class_metrics': class_metrics,
            'class_balance': class_balance
        }