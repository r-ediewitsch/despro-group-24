"""
prelabel_dataset.py
-------------------
Eksekusi Batch Pre-Labeling Otomatis Menggunakan Foundation Model (STEPS.md Fase 2.1):
- Mendukung Grounding DINO ('IDEA-Research/grounding-dino-tiny')
- Mendukung Microsoft Florence-2 ('microsoft/Florence-2-base')
- Mendukung YOLO ('models/best.pt')

Skrip ini membaca seluruh citra di folder mentah (dataset_raw/), mendeteksi
objek api ('fire') dan asap ('smoke') secara zero-shot semantik, lalu
menghasilkan file label format YOLO (.txt) di dataset_annotated/labels/.
Gambar di subfolder 'negative' otomatis diberi label file kosong (0 byte).
"""

import os
import sys
import glob
import argparse
import time

# Pastikan import src.vision bekerja
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.vision.foundation_labeler import FoundationLabeler
import cv2


def run_batch_prelabel(raw_dir="dataset_raw", out_dir="dataset_annotated",
                       backend="grounding_dino", model_id=None,
                       box_thresh=0.30, device=None):
    img_dir = os.path.join(out_dir, "images")
    lbl_dir = os.path.join(out_dir, "labels")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    print("\n" + "=" * 65)
    print("BATCH PRE-LABELING DENGAN FOUNDATION MODEL")
    print(f"  - Sumber Citra : {raw_dir}")
    print(f"  - Output Label : {lbl_dir}")
    print(f"  - Backend      : {backend}")
    print(f"  - Model ID     : {model_id or 'Default HuggingFace'}")
    print(f"  - Box Thresh   : {box_thresh}")
    print("=" * 65 + "\n")

    # Inisialisasi Foundation Model
    labeler = FoundationLabeler(backend=backend, model_id=model_id, device=device)

    # Temukan semua gambar mentah
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
    all_imgs = []
    for ext in exts:
        all_imgs.extend(glob.glob(os.path.join(raw_dir, "**", ext), recursive=True))
    all_imgs.sort()

    if not all_imgs:
        print(f"[ERROR] Tidak ada gambar ditemukan di '{raw_dir}'.")
        return

    print(f"[Pre-Label] Ditemukan {len(all_imgs)} gambar. Memulai proses inferensi...\n")

    total_boxes = 0
    bg_count = 0
    start_time = time.time()

    for idx, img_path in enumerate(all_imgs):
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        target_img_path = os.path.join(img_dir, base_name + ".jpg")
        target_lbl_path = os.path.join(lbl_dir, base_name + ".txt")

        # Baca citra
        bgr = cv2.imread(img_path)
        if bgr is None:
            continue
        h, w = bgr.shape[:2]

        # Simpan gambar ke folder dataset_annotated/images
        cv2.imwrite(target_img_path, bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])

        # Kasus 1: Subfolder Negative (Hard Negative Background)
        if "negative" in img_path.lower():
            open(target_lbl_path, "w").close()
            bg_count += 1
            print(f"[{idx+1}/{len(all_imgs)}] {base_name}: [BACKGROUND] 0 byte .txt")
            continue

        # Kasus 2: Inferensi dengan Foundation Model
        try:
            preds = labeler.predict(bgr, box_threshold=box_thresh)
        except Exception as e:
            print(f"[WARNING] Gagal memproses {base_name}: {e}")
            preds = []

        if len(preds) == 0:
            # Tidak ada deteksi api/asap -> simpan sebagai background 0 byte
            open(target_lbl_path, "w").close()
            bg_count += 1
            print(f"[{idx+1}/{len(all_imgs)}] {base_name}: Tidak ada api/asap (0 byte)")
        else:
            with open(target_lbl_path, "w") as f:
                for p in preds:
                    cls_id = p["class_id"]
                    x1, y1, x2, y2 = p["box"]
                    # Format YOLO: <class_id> <x_center> <y_center> <width> <height> (0.0 - 1.0)
                    xc = ((x1 + x2) / 2.0) / w
                    yc = ((y1 + y2) / 2.0) / h
                    bw = (x2 - x1) / w
                    bh = (y2 - y1) / h
                    f.write(f"{cls_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")
            total_boxes += len(preds)
            labels_summary = ", ".join([f"{p['class_name']}({p['confidence']:.2f})" for p in preds])
            print(f"[{idx+1}/{len(all_imgs)}] {base_name}: {len(preds)} box [{labels_summary}]")

    elapsed = time.time() - start_time
    print("\n" + "=" * 65)
    print("PRE-LABELING SELESAI")
    print(f"  - Waktu Eksekusi    : {elapsed:.1f} detik ({elapsed/len(all_imgs):.2f} detik/gambar)")
    print(f"  - Total Gambar      : {len(all_imgs)}")
    print(f"  - Total Bounding Box: {total_boxes}")
    print(f"  - Total Background  : {bg_count} (file 0-byte)")
    print(f"  - Direktori Label   : {lbl_dir}")
    print("\nLangkah selanjutnya:")
    print("  Jalankan 'python annotate_dataset.py' untuk memeriksa & memverifikasi hasil anotasi")
    print("  secara interaktif (Human-in-the-Loop) sebelum mengekspor dataset ke YOLO!")
    print("=" * 65 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Batch Pre-labeling Dataset Deezpro Menggunakan Foundation Model")
    parser.add_argument("--raw", default="dataset_raw", help="Direktori citra mentah (default: dataset_raw)")
    parser.add_argument("--out", default="dataset_annotated", help="Direktori target output (default: dataset_annotated)")
    parser.add_argument("--backend", default="grounding_dino", choices=["grounding_dino", "florence2", "yolo"],
                        help="Backend foundation model (default: grounding_dino)")
    parser.add_argument("--model-id", default=None,
                        help="Nama repo HuggingFace (contoh: IDEA-Research/grounding-dino-tiny atau microsoft/Florence-2-base)")
    parser.add_argument("--box-thresh", type=float, default=0.30,
                        help="Confidence threshold bounding box (default: 0.30)")
    parser.add_argument("--device", default=None, help="Device eksekusi ('cuda' atau 'cpu')")
    args = parser.parse_args()

    run_batch_prelabel(
        raw_dir=args.raw,
        out_dir=args.out,
        backend=args.backend,
        model_id=args.model_id,
        box_thresh=args.box_thresh,
        device=args.device
    )


if __name__ == "__main__":
    main()
