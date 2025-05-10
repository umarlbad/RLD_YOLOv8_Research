import time, traceback
from pathlib import Path

class TrainerModel:
    def __init__(self, device:str, project_dir: Path, log_dir:Path, data: str, model_type :str, size_model:str):
        self.device = device
        self.data = data # menentukan jenis data yang digunakan untuk fitting model
        self.model_type = model_type
        self.size_model = size_model
        self.project_dir = project_dir
        self.time_list = []
        # Ensure directories exist
        self.project_dir.mkdir(parents=True, exist_ok=True)
        # Create log file
        self.log_file = log_dir / f'training{self.data}_{self.model_type}_{self.size_model}_log.txt'
        
    def log_message(self, message: str):
        """Internal logging method"""
        timestamp = time.strftime('[%Y-%m-%d %H:%M:%S]')
        log_message = f"{timestamp} {message}"
        print(log_message)
        with open(self.log_file, 'a') as f:
            f.write(log_message + '\n')

    def log_training_metrics(self, results, epoch: int, fold: int):
        """
        Log training metrics with strong validation.
        """
        try:
            results_dict = getattr(results, 'results_dict', None)
            #loss_dict = getattr(results, '')
            if results_dict:
                # Validasi ketersediaan kunci
                precision = results_dict.get('metrics/precision(B)', None)
                recall = results_dict.get('metrics/recall(B)', None)
                mAP50 = results_dict.get('metrics/mAP50(B)', None)
                mAP50_95 = results_dict.get('metrics/mAP50-95(B)', None)
                fitness = results_dict.get('fitness',None)
                
                if precision is not None and recall is not None:
                    # Log metrik jika tersedia
                    self.log_message(f"\n[Epoch {epoch}] Training Metrics {self.data} for Fold {fold}:")
                    self.log_message(f"  Precision  : {precision:.3f}")
                    self.log_message(f"  Recall     : {recall:.3f}")
                    self.log_message(f"  mAP@50     : {mAP50:.3f}")
                    self.log_message(f"  mAP@50-95  : {mAP50_95:.3f}")
                    self.log_message(f"  Fitness    : {fitness:.3f}\n")

                else:
                    self.log_message("[Warning] Metrics not found in results_dict.")
        
        except Exception as e:
            self.log_message(f"[Error] Logging training metrics failed: {str(e)}")
            self.log_message(f"Error logging training metrics for Fold {fold}, Epoch {epoch}: {str(e)}")
        
    def train_fold(self, model, yaml_path: str | Path, config: dict, current_fold: int) -> dict:
        """Train model for a single fold"""
        try:
            self.log_message(f"\n\nStarting training data {self.data} for fold {current_fold}\n")
            
            # Setup training parameters
            train_args = dict(
                data=str(yaml_path),
                epochs=config['epochs'],
                imgsz=config['image_size'],
                batch=config['batch_size'],
                device=self.device,
                val=True,
                workers=config['workers'],
                close_mosaic=config['close_mosaic'],
                pretrained = True,
                patience = config['patience'],
                optimizer=config['optimizer'],
                lr0=config['learning_rate'],
                lrf=config['learning_rate_final'],
                cos_lr=True,
                exist_ok=True,
                rect=False,
                save_json = True,
                warmup_epochs=config['warmup_epochs'],
                warmup_momentum=config['warmup_momentum'],
                warmup_bias_lr=config['warmup_bias_lr'],
                momentum=config['momentum'],
                weight_decay=config['weight_decay'],
                freeze=config['freeze'],
                box=config['box'],
                cls=config['cls'],
                dfl=config['dfl'],
                conf=config['conf'],
                kobj = config['kobj'],
                pose = config['pose'],
                plots=True,
                profile = False,
                nms=True,
                save=True,
                save_period = config['save_period'],
                multi_scale=False, # Memungkinkan pelatihan multi-skala dengan meningkatkan/menurunkan imgsz hingga faktor 0,5 selama pelatihan. 
                line_width=config['line_width'],
                resume=False,
                iou = config['iou'],
                amp=True,
                seed = config['seed'],
                nbs=config['nbs'],
                project=str(self.project_dir / self.model_type / f"ukuran_{self.size_model}" / f"training_{current_fold}_{self.data}"),
                dropout =config['dropout'],

                # enable augmentations for training data
                augment=True,
                # Horizontal and vertical flip
                fliplr=config['augment_sec'],    # 50% chance of horizontal flip
                flipud=config['augment_sec'],    # 50% chance of vertical flip
                
                # Rotation, translate and scaling
                degrees=config['augment_sec'],    # Rotate up to 90 degrees and 180 degrees
                translate=config['augment_first'],
                scale=config['augment_sec'],
                mixup = config['augment_sec'],
                mosaic=config['augment_sec'],
                shear=config['augment_zero'],
                bgr=config['augment_zero'],
                perspective=config['augment_zero'],
                copy_paste=config['augment_sec'],
                # Color and brightness adjustments
                hsv_h=config['augment_first'],   # Hue adjustment
                hsv_s=config['augment_first'],    # Saturation adjustment (-25% to +25%)
                hsv_v=config['augment_first'],    # Value/Brightness adjustment (-25% to +25%)
                erasing=config['augment_sec'], # Secara acak menghapus sebagian gambar selama pelatihan klasifikasi, mendorong model untuk fokus pada fitur yang kurang jelas untuk dikenali.
                crop_fraction=config['augment_sec'] # Memangkas gambar klasifikasi menjadi sebagian kecil dari ukurannya untuk menekankan fitur utama dan menyesuaikan dengan skala objek, mengurangi gangguan latar belakang.
            )
            # Add log for checking the training params
            self.log_message(f"\nTraining Parameters: {train_args}\n")
            # Start training
            start_time = time.time()
            results = model.train(**train_args)
            
            duration = time.time() - start_time
            durasi = duration / 3600
            epoch = config['epochs']
            
            # Get metrics
            self.log_message(f"\n------ Results Object {self.data.upper()} TRAINING ------")
            self.log_training_metrics(results, epoch=epoch, fold=current_fold)

            metrics = {
                'train_precision': results.results_dict.get('precision', 0),
                'train_recall': results.results_dict.get('recall', 0),
                'train_mAP50': results.results_dict.get('mAP50', 0),
                'train_mAP50_95': results.results_dict.get('mAP50-95', 0),
                'fitness': results.results_dict.get('fitness',0),
                'training_time': duration
                }
            
            self.time_list.append(duration)
            self.log_message(f"\nFold {current_fold} completed - Training {self.data} Time: {duration:.2f} sekon atau {durasi:.2f} jam.\n")
            
            return metrics
            
        except Exception as e:
            error_message = f"\nError in Fold {current_fold}: {str(e)}"
            self.log_message(error_message)
            self.log_message(traceback.format_exc())
            raise RuntimeError(error_message) from e