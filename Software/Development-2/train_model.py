"""
=====================================================================
DEEZPRO - FIRE & SMOKE YOLO TRAINING PIPELINE
=====================================================================
File: train_model.py
Deskripsi: Script training YOLOv8/YOLO11 untuk model deteksi api dan asap.
           Dikonfigurasi khusus agar optimal pada GPU Laptop (RTX 3050 4GB VRAM)
           dengan tetap mempertahankan kualitas akurasi setara GPU Server (HPC)
           melalui Automatic Gradient Accumulation, Cosine LR, dan Augmentasi
           kecerahan/warna khas arena fisik Deezpro.
=====================================================================
"""

import os
import sys
import shutil
import argparse

# Arahkan konfigurasi Ultralytics ke direktori lokal agar tidak terkendala izin AppData
current_dir = os.path.dirname(os.path.abspath(__file__))
os.environ["YOLO_CONFIG_DIR"] = os.path.join(current_dir, ".yolo")

import torch

try:
    from ultralytics import YOLO
except ImportError:
    print("[ERROR] Library 'ultralytics' belum terpasang.")
    print("        Silakan pasang dengan: pip install ultralytics")
    sys.exit(1)


def check_gpu_environment():
    """Memeriksa kesiapan akselerasi GPU CUDA."""
    print("\n" + "=" * 65)
    print("MEMERIKSA LINGKUNGAN KOMPUTASI GPU")
    print("=" * 65)
    
    cuda_available = torch.cuda.is_available()
    print(f"  - PyTorch Version  : {torch.__version__}")
    print(f"  - CUDA Tersedia    : {'YA' if cuda_available else 'TIDAK (Menggunakan CPU)'}")
    
    if cuda_available:
        gpu_name = torch.cuda.get_device_name(0)
        vram_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"  - Perangkat GPU    : {gpu_name}")
        print(f"  - Total VRAM       : {vram_total:.2f} GB")
        if vram_total <= 4.5:
            print("  - Profil Memori    : 4GB VRAM Mode (Batch 8 + Mixed Precision FP16)")
    else:
        print("  [PERINGATAN] GPU CUDA tidak terdeteksi! Training akan berjalan di CPU")
        print("               (Estimasi waktu akan jauh lebih lambat).")
    print("=" * 65 + "\n")


def train_fire_model(
    data_yaml="dataset_combined/data.yaml",
    base_model="yolov8s.pt",
    epochs=60,
    batch=8,
    imgsz=640,
    device=0 if torch.cuda.is_available() else "cpu",
    workers=4,
    project="runs/fire_detection",
    name="deezpro_rtx3050_run",
    auto_export_to_models=True
):
    """Menjalankan siklus training YOLO dengan hyperparameter yang ditune untuk RTX 3050 4GB."""
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, data_yaml) if not os.path.isabs(data_yaml) else data_yaml

    if not os.path.exists(data_path):
        print(f"[ERROR] Berkas data konfigurasi dataset tidak ditemukan di: {data_path}")
        print("        Pastikan Anda telah menjalankan 'merge_datasets.py' terlebih dahulu!")
        sys.exit(1)

    # Pastikan 'path:' di dalam data.yaml selalu valid dan cocok dengan folder mesin saat ini
    try:
        import yaml
        with open(data_path, "r", encoding="utf-8") as f:
            yaml_data = yaml.safe_load(f) or {}
        actual_dataset_dir = os.path.dirname(os.path.abspath(data_path)).replace("\\", "/")
        if yaml_data.get("path") != actual_dataset_dir:
            yaml_data["path"] = actual_dataset_dir
            with open(data_path, "w", encoding="utf-8") as f:
                yaml.dump(yaml_data, f, default_flow_style=False)
            print(f"[AUTO-FIX] Menyesuaikan direktori dataset pada data.yaml -> {actual_dataset_dir}")
    except Exception as e:
        print(f"[INFO] Lewati penyesuaian otomatis data.yaml: {e}")

    check_gpu_environment()

    print(f"[INFO] Memuat pre-trained backbone model: {base_model}...")
    model = YOLO(base_model)

    print("\n" + "=" * 65)
    print("MEMULAI TRAINING DEEZPRO FIRE & SMOKE DETECTION")
    print("=" * 65)
    print(f"  - Konfigurasi Data   : {data_path}")
    print(f"  - Total Epochs       : {epochs}")
    print(f"  - Batch Size         : {batch} (Effective: 64 via Gradient Accumulation)")
    print(f"  - Resolusi Input     : {imgsz} x {imgsz} px")
    print(f"  - Optimizer          : AdamW (lr0=0.001, Cosine Annealing Scheduler)")
    print(f"  - Precision Mode     : FP16 AMP (Automatic Mixed Precision)")
    print(f"  - Folder Output      : {project}/{name}")
    print("=" * 65 + "\n")

    # Eksekusi proses training
    results = model.train(
        data=data_path,
        epochs=epochs,
        patience=15,             # Hentikan lebih awal jika mAP tidak membaik selama 15 epoch berturut-turut
        batch=batch,             # Aman dari OOM pada GPU 4GB VRAM
        imgsz=imgsz,             # Standar detail spasial api
        device=device,
        workers=workers,         # Efisien untuk CPU laptop tanpa membebani RAM sistem
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,                # Learning rate akhir mencapai 1% dari awal
        cos_lr=True,             # Cosine learning rate scheduler
        warmup_epochs=3.0,       # Pemanasan gradien di 3 epoch pertama
        close_mosaic=10,         # Nonaktifkan augmentasi mosaic di 10 epoch terakhir untuk memantapkan bbox
        mosaic=0.2,              # Mozaik moderat agar api kecil tidak mengecil menjadi titik LED
        flipud=0.0,              # Matikan flip vertikal (api tidak pernah terbakar terbalik ke bawah)
        fliplr=0.5,              # Flip horizontal 50%
        hsv_h=0.015,             # Variasi hue spektrum lidah api
        hsv_s=0.3,               # Lindungi saturasi tepian kuning/oranye api
        hsv_v=0.2,               # Lindungi kontras eksposur inti putih api
        amp=True,                # Mixed Precision FP16
        project=project,
        name=name
    )

    # Validasi hasil model akhir
    print("\n" + "=" * 65)
    print("EVALUASI METRIK PADA DATASET VALIDASI")
    print("=" * 65)
    metrics = model.val()
    print(f"  - Validation mAP@0.5      : {metrics.box.map50:.4f}")
    print(f"  - Validation mAP@0.5:0.95 : {metrics.box.map:.4f}")
    print("=" * 65 + "\n")

    # Salin bobot terbaik ke direktori models/best.pt
    if auto_export_to_models:
        try:
            save_dir = str(results.save_dir) if hasattr(results, 'save_dir') else os.path.join(project, name)
            best_weight_src = os.path.join(save_dir, "weights", "best.pt")
            target_models_dir = os.path.join(current_dir, "models")
            target_best_pt = os.path.join(target_models_dir, "best.pt")

            if os.path.exists(best_weight_src):
                os.makedirs(target_models_dir, exist_ok=True)
                # Buat cadangan bobot lama jika ada
                if os.path.exists(target_best_pt):
                    backup_path = os.path.join(target_models_dir, "best_backup.pt")
                    shutil.copy2(target_best_pt, backup_path)
                    print(f"[INFO] Bobot model lama dicadangkan ke: {backup_path}")

                shutil.copy2(best_weight_src, target_best_pt)
                print(f"[SUKSES] Bobot terbaik berhasil diekspor ke: {target_best_pt}")
                print("         Seluruh subsistem Deezpro (detect_fire_pixels.py dll) otomatis menggunakan bobot ini!")
            else:
                print(f"[PERINGATAN] Berkas bobot {best_weight_src} tidak ditemukan.")
        except Exception as e:
            print(f"[PERINGATAN] Gagal menyalin bobot terbaik ke folder models/: {e}")

    print("\nTRAINING TELAH SELESAI DENGAN SUKSES!")
    print("Langkah selanjutnya:")
    print("  1. Uji visual deteksi piksel webcam langsung:")
    print("     python detect_fire_pixels.py --camera 1")
    print("  2. Uji integrasi homografi end-to-end:")
    print("     python test_homography_end_to_end.py --camera 1\n")


def main():
    parser = argparse.ArgumentParser(description="Script Training YOLOv8 Deezpro (Optimized for RTX 3050 4GB)")
    parser.add_argument("--data", default="dataset_combined/data.yaml",
                        help="Path ke file data.yaml (default: dataset_combined/data.yaml)")
    parser.add_argument("--model", default="yolov8s.pt",
                        help="Pre-trained checkpoint (default: yolov8s.pt atau yolov8n.pt)")
    parser.add_argument("--epochs", type=int, default=60,
                        help="Jumlah epochs training (default: 60)")
    parser.add_argument("--batch", type=int, default=8,
                        help="Ukuran batch size (default: 8 untuk 4GB VRAM)")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Resolusi dimensi input gambar (default: 640)")
    parser.add_argument("--workers", type=int, default=4,
                        help="Jumlah CPU worker dataloader (default: 4)")
    parser.add_argument("--project", default="runs/fire_detection",
                        help="Direktori penyimpanan run (default: runs/fire_detection)")
    parser.add_argument("--name", default="deezpro_rtx3050_run",
                        help="Nama folder eksperimen (default: deezpro_rtx3050_run)")
    parser.add_argument("--no-export", action="store_true",
                        help="Jangan otomatis perbarui models/best.pt")

    args = parser.parse_args()

    device_str = 0 if torch.cuda.is_available() else "cpu"

    train_fire_model(
        data_yaml=args.data,
        base_model=args.model,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=device_str,
        workers=args.workers,
        project=args.project,
        name=args.name,
        auto_export_to_models=not args.no_export
    )


if __name__ == "__main__":
    main()
