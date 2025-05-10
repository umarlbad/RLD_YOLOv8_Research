import time, traceback
from pathlib import Path

class ValidatorModel:
    def __init__(self, device:str, project_dir:str, log_dir:Path, data : str, model_type:str, size_model:str):
        self.device = device
        self.data = data # untuk menentukan jenis data apa yang digunakan
        self.model_type = model_type
        self.size_model = size_model
        self.project_dir = project_dir
        self.time_list = []

        # Create log file
        self.log_file = log_dir / f'validation_{self.data}_{self.model_type}_{self.size_model}_log.txt'
        # Create Tensorboard Writer

    def log_message(self, message: str):
        """Internal logging method"""
        timestamp = time.strftime('[%Y-%m-%d %H:%M:%S]')
        log_message = f"{timestamp} {message}"
        print(log_message)
        with open(self.log_file, 'a') as f:
            f.write(log_message + '\n')

    def log_val_metrics(self, results, epoch: int, fold: int):
        """
        Log validation metrics with strong validation checks.
        Args:
            results: Object containing validation results.
            epoch: Current epoch number.
            fold: Current fold number in validation.
        """
        try:
            # Debugging untuk hasil lengkap
            results_dict = getattr(results, 'results_dict', None)

            if results_dict:
                # Validasi ketersediaan kunci
                val_precision = results_dict.get('metrics/precision(B)', None)
                val_recall = results_dict.get('metrics/recall(B)', None)
                val_mAP50 = results_dict.get('metrics/mAP50(B)', None)
                val_mAP50_95 = results_dict.get('metrics/mAP50-95(B)', None)
                val_fitnes = results_dict.get('fitness', None)
                
                if val_precision is not None and val_recall is not None:
                    # Log metrik jika tersedia
                    self.log_message(f"\n[Epoch {epoch}] Validation Metrics {self.data} for Fold {fold}:")
                    self.log_message(f" Precision  : {val_precision:.3f}")
                    self.log_message(f" Recall     : {val_recall:.3f}")
                    self.log_message(f" mAP@50     : {val_mAP50:.3f}")
                    self.log_message(f" mAP@50-95  : {val_mAP50_95:.3f}")
                    self.log_message(f" Fitness : {val_fitnes:.3f}\n")

                else:
                    self.log_message("[Warning] Validation metrics not found in results_dict.")
                    self.log_message(f"No validation metrics available for Fold {fold} at Epoch {epoch}.")
        
        except Exception as e:
            self.log_message(f"[Error] Logging validation metrics failed: {str(e)}")
            self.log_message(f"Error logging validation metrics for Fold {fold}, Epoch {epoch}: {str(e)}")

    def validation_data(self, model, yaml_path: str | Path, config:dict, current_fold: int):
        """Evaluate model performance on test set"""
        try:
            # Print YAML path for debugging
            self.log_message(f"\nValidation {self.data} YAML Path: {yaml_path}\n")
            
            self.log_message(f"\n\nStarting VALIDATION with val data {self.data} for fold {current_fold}\n\n")

            start_time = time.time()
            val_results = model.val(
                data=str(yaml_path),
                device=self.device,
                iou=config['iou'],
                conf=config['conf'],
                batch = config['batch_size'],
                nms=True,
                split='val',  # Specify validation split
                plots=True,    # Generate plots
                dnn = False, # berbasis DNN Module untuk ONNX model inferensi = TensorRT (digunakan apabila model berbasis onnx/engine dari tensorrt)
                rect = True, # rectangular reference untuk mengurangi padding
                save_json=True,
                save_crop=True,
                save_conf=True,
                #save_hybrid=True, # menyimpan hasil antara bbox original (gt) dan hasil predict
                save=True,      # Save results
                project = str(self.project_dir / self.model_type / f"ukuran_{self.size_model}" / f"validating_{current_fold}_{self.data}"),# pindahin direktori project untuk data testing
            )
            epoch = config['epochs']
            
            duration = time.time() - start_time
            durasi = duration / 3600
            # Log val set metrics
            self.log_val_metrics(val_results, epoch=epoch, fold=current_fold)
            self.log_message(f"\nHasil Box Metriks Validasi {self.data.upper()}:")
            self.log_message(f"Hasil Box mAP50-95 : {val_results.box.map:.2f}")
            self.log_message(f"Hasil Box mAP50 : {val_results.box.map50:.2f}")
            self.log_message(f"Hasil Box AP : {val_results.box.ap}")
            self.log_message(f"Hasil Box Precision : {val_results.box.p}")
            self.log_message(f"Hasil Box F1-Score : {val_results.box.f1}")
            self.log_message(f"Hasil Box Recall : {val_results.box.r}\n")
            metrics = {
                'val_precision': val_results.results_dict.get('precision', 0),
                'val_recall': val_results.results_dict.get('recall', 0),
                'val_mAP50': val_results.results_dict.get('mAP50', 0),
                'val_mAP50_95': val_results.results_dict.get('mAP50-95', 0),
                'val_fitness':val_results.results_dict.get('fitness', 0),
                'validation_time': duration
            }
            self.time_list.append(duration)
            self.log_message(f"\nFold {current_fold} completed - Validating data {self.data} Time: {duration:.2f}s atau {durasi:.2f} jam \n")
            return metrics
        
        except Exception as e:
            error_message = f"\nError in fold {current_fold}: {str(e)}"
            self.log_message(error_message)
            self.log_message(traceback.format_exc())
            raise RuntimeError(error_message) from e