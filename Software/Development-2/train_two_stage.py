"""
=====================================================================
DEEZPRO - TWO-STAGE TRAINING PIPELINE (RESPONSE.MD COMPLIANT)
=====================================================================
File: train_two_stage.py
Deskripsi: Script training 2 langkah (Base Training & Fine-Tuning):
           - Tahap 1 (Base Learning): Melatih model pada 1.200 foto D-Fire
             yang didegradasi sensor webcam (lr0=0.01, mosaic=0.5, 80 epochs).
           - Tahap 2 (Webcam Adaptation Fine-Tuning): Melatih model lanjutan
             menggunakan bobot Stage 1 pada 1.000 foto (410 lokal + 350 copy-paste
             sintetis + 240 D-Fire) dengan lr0=0.001, mosaic=0.2, close_mosaic=15.
           - Validasi kedua tahap MURNI menggunakan 102 foto local_val (untouched).
           - Menghasilkan bobot akhir: models/best-fine.pt.
=====================================================================
"""

import os
import sys
import glob
import random
import shutil
import argparse
import numpy as np

# Pastikan konfigurasi Ultralytics lokal agar aman di Windows & Linux
current_dir = os.path.dirname(os.path.abspath(__file__))
os.environ["YOLO_CONFIG_DIR"] = os.path.join(current_dir, ".yolo")

import torch
import cv2

try:
    from ultralytics import YOLO
except ImportError:
    print("[ERROR] Library 'ultralytics' belum terpasang.")
    print("        Silakan pasang dengan: pip install ultralytics")
    sys.exit(1)


# =====================================================================
# 1. HELPER PREPROCESSING & SENSOR DEGRADATION
# =====================================================================

def degrade_image_sensor(
    img: np.ndarray,
    scale_range=(0.35, 0.50),
    jpeg_quality_range=(35, 60),
    noise_sigma_range=(3.0, 7.0),
    blur_prob=0.5
) -> np.ndarray:
    """Meniru artefak optik dan sensor webcam murah."""
    h, w = img.shape[:2]
    scale = random.uniform(*scale_range)
    sw = max(16, int(w * scale))
    sh = max(16, int(h * scale))
    downsampled = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_AREA)
    upsampled = cv2.resize(downsampled, (w, h), interpolation=cv2.INTER_LINEAR)

    sigma = random.uniform(*noise_sigma_range)
    noise = np.random.normal(0, sigma, upsampled.shape).astype(np.float32)
    noisy = np.clip(upsampled.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    if random.random() < blur_prob:
        noisy = cv2.GaussianBlur(noisy, (3, 3), 0)

    quality = random.randint(*jpeg_quality_range)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    success, encimg = cv2.imencode(".jpg", noisy, encode_param)
    if success:
        return cv2.imdecode(encimg, cv2.IMREAD_COLOR)
    return noisy


def generate_copy_paste_samples(
    bg_pairs: list,
    fg_pairs: list,
    num_samples: int,
    out_img_dir: str,
    out_lbl_dir: str,
    prefix="synth"
) -> int:
    """Memotong kotak api dari D-Fire dan menempelkannya ke background lokal Deezpro."""
    if not bg_pairs or not fg_pairs or num_samples <= 0:
        return 0

    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_lbl_dir, exist_ok=True)

    print(f"  [+] Menghasilkan {num_samples} foto sintetis Copy-Paste...")
    generated_count = 0

    for i in range(num_samples):
        bg_img_path, _ = random.choice(bg_pairs)
        fg_img_path, fg_lbl_path = random.choice(fg_pairs)

        bg = cv2.imread(bg_img_path)
        fg = cv2.imread(fg_img_path)
        if bg is None or fg is None:
            continue

        bh, bw = bg.shape[:2]
        fh, fw = fg.shape[:2]

        with open(fg_lbl_path, "r", encoding="utf-8") as f:
            lines = [line.strip().split() for line in f if len(line.strip().split()) >= 5]

        if not lines:
            continue

        new_labels = []
        canvas = bg.copy()
        selected_boxes = random.sample(lines, min(len(lines), random.randint(1, 2)))

        for box in selected_boxes:
            cls_id = int(box[0])
            cx, cy, bw_norm, bh_norm = map(float, box[1:5])

            px1 = max(0, int((cx - bw_norm / 2) * fw))
            py1 = max(0, int((cy - bh_norm / 2) * fh))
            px2 = min(fw, int((cx + bw_norm / 2) * fw))
            py2 = min(fh, int((cy + bh_norm / 2) * fh))

            crop = fg[py1:py2, px1:px2]
            ch, cw = crop.shape[:2]
            if ch < 10 or cw < 10:
                continue

            scale = random.uniform(0.6, 1.2)
            nw = max(15, min(int(cw * scale), int(bw * 0.45)))
            nh = max(15, min(int(ch * scale), int(bh * 0.45)))
            crop_resized = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_LINEAR)

            mask = np.ones((nh, nw), dtype=np.float32)
            border = max(2, min(nh, nw) // 10)
            mask[:border, :] = 0
            mask[-border:, :] = 0
            mask[:, :border] = 0
            mask[:, -border:] = 0
            mask = cv2.GaussianBlur(mask, (7, 7), 2.0)
            mask_3c = np.repeat(mask[:, :, np.newaxis], 3, axis=2)

            tx = random.randint(10, max(11, bw - nw - 10))
            ty = random.randint(10, max(11, bh - nh - 10))

            roi = canvas[ty:ty+nh, tx:tx+nw]
            blended = (crop_resized.astype(np.float32) * mask_3c + roi.astype(np.float32) * (1.0 - mask_3c)).astype(np.uint8)
            canvas[ty:ty+nh, tx:tx+nw] = blended

            new_cx = (tx + nw / 2.0) / bw
            new_cy = (ty + nh / 2.0) / bh
            new_w = nw / bw
            new_h = nh / bh
            new_labels.append(f"{cls_id} {new_cx:.6f} {new_cy:.6f} {new_w:.6f} {new_h:.6f}")

        if new_labels:
            fname = f"{prefix}_{i+1:04d}.jpg"
            out_img_p = os.path.join(out_img_dir, fname)
            out_lbl_p = os.path.join(out_lbl_dir, f"{prefix}_{i+1:04d}.txt")

            cv2.imwrite(out_img_p, canvas)
            with open(out_lbl_p, "w", encoding="utf-8") as f:
                f.write("\n".join(new_labels) + "\n")
            generated_count += 1

    return generated_count


# =====================================================================
# 2. PERSIAPAN DATASET DUA TAHAP (STAGE 1 & STAGE 2)
# =====================================================================

def prepare_two_stage_datasets(
    local_dir="dataset_annotated",
    public_dir="../Development/fire-detection/datasets",
    stage1_dir="dataset_stage1",
    stage2_dir="dataset_stage2",
    num_dfire_stage1=1200,
    num_dfire_stage2=240,
    num_copy_paste=350
):
    """
    Menyiapkan dataset Stage 1 dan Stage 2 sesuai rencana Response.md:
    - local_val: 102 foto (72 pos, 30 neg) dikunci sebagai benchmark kedua tahap.
    - Stage 1: 1.200 degraded D-Fire.
    - Stage 2: 410 local_train + 350 copy-paste sintetis + 240 degraded D-Fire (~1.000 foto).
    """
    print("\n" + "=" * 70)
    print("PERSIAPAN DATASET TWO-STAGE (RESPONSE.MD SPECIFICATION)")
    print("=" * 70)

    # 1. Kumpulkan data mandiri
    loc_lbl_dir = os.path.join(local_dir, "labels")
    loc_img_dir = os.path.join(local_dir, "images")

    local_positive_pairs = []
    local_negative_pairs = []

    if os.path.exists(loc_lbl_dir):
        all_txt = glob.glob(os.path.join(loc_lbl_dir, "**", "*.txt"), recursive=True)
        for txt_p in all_txt:
            base = os.path.splitext(os.path.basename(txt_p))[0]
            img_candidates = [
                os.path.join(loc_img_dir, base + ".jpg"),
                os.path.join(loc_img_dir, base + ".png"),
                os.path.join(local_dir, base + ".jpg")
            ]
            img_p = next((c for c in img_candidates if os.path.exists(c)), None)
            if not img_p:
                continue

            with open(txt_p, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content:
                    local_positive_pairs.append((img_p, txt_p))
                else:
                    local_negative_pairs.append((img_p, txt_p))

    # Fallback pintar: jika dataset_annotated kosong di server, cari dari dataset_combined
    if not local_positive_pairs and not local_negative_pairs:
        comb_dir = os.path.join(current_dir, "dataset_combined")
        if os.path.exists(comb_dir):
            print(f"  [i] Mengambil data lokal dari '{comb_dir}' karena '{local_dir}' tidak berisi gambar...")
            for sub in ["train", "val"]:
                sub_img = os.path.join(comb_dir, "images", sub)
                sub_lbl = os.path.join(comb_dir, "labels", sub)
                for img_p in glob.glob(os.path.join(sub_img, "local_*.*")):
                    base = os.path.splitext(os.path.basename(img_p))[0]
                    lbl_p = os.path.join(sub_lbl, base + ".txt")
                    if os.path.exists(lbl_p):
                        with open(lbl_p, "r", encoding="utf-8") as f:
                            if f.read().strip():
                                local_positive_pairs.append((img_p, lbl_p))
                            else:
                                local_negative_pairs.append((img_p, lbl_p))

    random.seed(42)
    np.random.seed(42)

    random.shuffle(local_positive_pairs)
    random.shuffle(local_negative_pairs)

    # Split 80/20 data lokal
    pos_s = int(len(local_positive_pairs) * 0.8)
    neg_s = int(len(local_negative_pairs) * 0.8)

    loc_train_pos = local_positive_pairs[:pos_s]
    loc_val_pos = local_positive_pairs[pos_s:]

    loc_train_neg = local_negative_pairs[:neg_s]
    loc_val_neg = local_negative_pairs[neg_s:]

    local_train = loc_train_pos + loc_train_neg
    local_val = loc_val_pos + loc_val_neg

    print(f"  - local_train : {len(local_train)} foto ({len(loc_train_pos)} pos, {len(loc_train_neg)} neg)")
    print(f"  - local_val   : {len(local_val)} foto ({len(loc_val_pos)} pos, {len(loc_val_neg)} neg) -> BENCHMARK DIKUNCI")

    # 2. Kumpulkan data publik D-Fire
    public_pairs = []
    # Cek lokasi public_dir yang diberikan atau cari kandidat umum
    candidate_public_dirs = [
        public_dir,
        os.path.normpath(os.path.join(current_dir, "..", "Development", "fire-detection", "datasets")),
        os.path.normpath(os.path.join(current_dir, "..", "..", "fire-detection", "datasets")),
        os.path.normpath(os.path.join(current_dir, "..", "datasets")),
    ]

    for pdir in candidate_public_dirs:
        if os.path.exists(pdir):
            pub_img_dir = os.path.join(pdir, "images")
            pub_lbl_dir = os.path.join(pdir, "labels")
            pub_imgs = glob.glob(os.path.join(pub_img_dir, "**", "*.jpg"), recursive=True)
            for img_p in pub_imgs:
                base = os.path.splitext(os.path.basename(img_p))[0]
                sub = os.path.basename(os.path.dirname(img_p))
                lbl_candidate = os.path.join(pub_lbl_dir, sub, base + ".txt")
                if not os.path.exists(lbl_candidate):
                    lbl_candidate = os.path.join(pub_lbl_dir, base + ".txt")
                if os.path.exists(lbl_candidate):
                    public_pairs.append((img_p, lbl_candidate))
            if public_pairs:
                print(f"  - D-Fire data ditemukan di '{pdir}': {len(public_pairs)} foto")
                break

    # Fallback publik: ambil dari dataset_combined jika ada
    if not public_pairs:
        comb_dir = os.path.join(current_dir, "dataset_combined")
        if os.path.exists(comb_dir):
            for sub in ["train", "val"]:
                sub_img = os.path.join(comb_dir, "images", sub)
                sub_lbl = os.path.join(comb_dir, "labels", sub)
                for img_p in glob.glob(os.path.join(sub_img, "pub_*.*")):
                    base = os.path.splitext(os.path.basename(img_p))[0]
                    lbl_p = os.path.join(sub_lbl, base + ".txt")
                    if os.path.exists(lbl_p):
                        public_pairs.append((img_p, lbl_p))
            if public_pairs:
                print(f"  [i] Mengambil {len(public_pairs)} foto publik dari '{comb_dir}'!")

    random.shuffle(public_pairs)

    # 3. Setup Stage 1 (1.200 degraded D-Fire train, local_val sebagai val)
    print("\n[Stage 1 Dataset Setup]")
    s1_train_img = os.path.join(stage1_dir, "images", "train")
    s1_train_lbl = os.path.join(stage1_dir, "labels", "train")
    s1_val_img = os.path.join(stage1_dir, "images", "val")
    s1_val_lbl = os.path.join(stage1_dir, "labels", "val")

    for d in (s1_train_img, s1_train_lbl, s1_val_img, s1_val_lbl):
        os.makedirs(d, exist_ok=True)

    # Salin local_val
    for img_p, lbl_p in local_val:
        fname = "local_" + os.path.basename(img_p)
        shutil.copy2(img_p, os.path.join(s1_val_img, fname))
        shutil.copy2(lbl_p, os.path.join(s1_val_lbl, os.path.splitext(fname)[0] + ".txt"))

    # Salin degraded D-Fire untuk Stage 1
    dfire_s1 = public_pairs[:num_dfire_stage1]
    if dfire_s1:
        print(f"  [+] Menyiapkan {len(dfire_s1)} foto D-Fire untuk Stage 1...")
        for img_p, lbl_p in dfire_s1:
            fname = "dfire_" + os.path.basename(img_p)
            dst_img = os.path.join(s1_train_img, fname)
            dst_lbl = os.path.join(s1_train_lbl, os.path.splitext(fname)[0] + ".txt")
            
            # Jika file sudah berawalan 'pub_', berarti sudah didegrade di dataset_combined (hindari double degrade)
            if os.path.basename(img_p).startswith("pub_"):
                shutil.copy2(img_p, dst_img)
            else:
                img = cv2.imread(img_p)
                if img is not None:
                    cv2.imwrite(dst_img, degrade_image_sensor(img))
                else:
                    shutil.copy2(img_p, dst_img)
            shutil.copy2(lbl_p, dst_lbl)
    else:
        print("  [PERINGATAN] Data D-Fire tidak ditemukan! Mengisi train Stage 1 dengan data lokal.")
        for img_p, lbl_p in local_train:
            fname = "local_" + os.path.basename(img_p)
            shutil.copy2(img_p, os.path.join(s1_train_img, fname))
            shutil.copy2(lbl_p, os.path.join(s1_train_lbl, os.path.splitext(fname)[0] + ".txt"))

    # Tulis Stage 1 data.yaml
    s1_yaml_p = os.path.join(stage1_dir, "data.yaml")
    abs_s1 = os.path.abspath(stage1_dir).replace("\\", "/")
    with open(s1_yaml_p, "w", encoding="utf-8") as f:
        f.write(f"path: {abs_s1}\ntrain: images/train\nval: images/val\nnames:\n  0: smoke\n  1: fire\n")

    # 4. Setup Stage 2 (410 local_train + 350 copy-paste + 240 degraded D-Fire)
    print("\n[Stage 2 Dataset Setup]")
    s2_train_img = os.path.join(stage2_dir, "images", "train")
    s2_train_lbl = os.path.join(stage2_dir, "labels", "train")
    s2_val_img = os.path.join(stage2_dir, "images", "val")
    s2_val_lbl = os.path.join(stage2_dir, "labels", "val")

    for d in (s2_train_img, s2_train_lbl, s2_val_img, s2_val_lbl):
        os.makedirs(d, exist_ok=True)

    # Salin local_val yang persis sama
    for img_p, lbl_p in local_val:
        fname = "local_" + os.path.basename(img_p)
        shutil.copy2(img_p, os.path.join(s2_val_img, fname))
        shutil.copy2(lbl_p, os.path.join(s2_val_lbl, os.path.splitext(fname)[0] + ".txt"))

    # Salin 410 local_train
    for img_p, lbl_p in local_train:
        fname = "local_" + os.path.basename(img_p)
        shutil.copy2(img_p, os.path.join(s2_train_img, fname))
        shutil.copy2(lbl_p, os.path.join(s2_train_lbl, os.path.splitext(fname)[0] + ".txt"))

    # Salin 240 degraded D-Fire untuk Stage 2
    dfire_s2 = public_pairs[num_dfire_stage1:num_dfire_stage1 + num_dfire_stage2]
    if not dfire_s2 and public_pairs:
        dfire_s2 = public_pairs[:num_dfire_stage2]

    if dfire_s2:
        print(f"  [+] Menyiapkan {len(dfire_s2)} foto D-Fire pendamping Stage 2...")
        for img_p, lbl_p in dfire_s2:
            fname = "dfire_" + os.path.basename(img_p)
            dst_img = os.path.join(s2_train_img, fname)
            dst_lbl = os.path.join(s2_train_lbl, os.path.splitext(fname)[0] + ".txt")

            # Hindari double degrade jika sudah berawalan 'pub_'
            if os.path.basename(img_p).startswith("pub_"):
                shutil.copy2(img_p, dst_img)
            else:
                img = cv2.imread(img_p)
                if img is not None:
                    cv2.imwrite(dst_img, degrade_image_sensor(img))
                else:
                    shutil.copy2(img_p, dst_img)
            shutil.copy2(lbl_p, dst_lbl)

    # Generate 350 Copy-Paste Sintetis
    if public_pairs and loc_train_neg:
        generate_copy_paste_samples(
            bg_pairs=loc_train_neg,
            fg_pairs=public_pairs,
            num_samples=num_copy_paste,
            out_img_dir=s2_train_img,
            out_lbl_dir=s2_train_lbl,
            prefix="synth_paste"
        )

    # Tulis Stage 2 data.yaml
    s2_yaml_p = os.path.join(stage2_dir, "data.yaml")
    abs_s2 = os.path.abspath(stage2_dir).replace("\\", "/")
    with open(s2_yaml_p, "w", encoding="utf-8") as f:
        f.write(f"path: {abs_s2}\ntrain: images/train\nval: images/val\nnames:\n  0: smoke\n  1: fire\n")

    print("\n[✓] Persiapan dataset Stage 1 dan Stage 2 selesai!")
    return s1_yaml_p, s2_yaml_p


# =====================================================================
# 3. STAGE 1 TRAINING (BASE FIRE LEARNING)
# =====================================================================

def train_stage_1(
    data_yaml="dataset_stage1/data.yaml",
    base_model="yolov8s.pt",
    epochs=80,
    batch=32,
    imgsz=640,
    device=0 if torch.cuda.is_available() else "cpu",
    workers=8,
    project="runs/fire_detection",
    name="deezpro_stage1_run"
) -> str:
    """Melatih Stage 1: Base Fire Learning pada data degraded D-Fire."""
    print("\n" + "=" * 70)
    print("MEMULAI STAGE 1 TRAINING: BASE FIRE LEARNING")
    print(f"  - Model Awal     : {base_model}")
    print(f"  - Dataset        : {data_yaml}")
    print(f"  - Epochs         : {epochs}")
    print(f"  - Batch Size     : {batch}")
    print(f"  - Learning Rate  : 0.01 (Standard LR + Cosine Decay)")
    print(f"  - Augmentasi     : mosaic=0.5, flipud=0.0, fliplr=0.5")
    print("=" * 70 + "\n")

    project_path = os.path.join(current_dir, project) if not os.path.isabs(project) else project

    # Verifikasi ketersediaan gambar training sebelum memanggil Ultralytics
    s1_train_img_dir = os.path.join(os.path.dirname(os.path.abspath(data_yaml)), "images", "train")
    img_count = len([f for f in os.listdir(s1_train_img_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]) if os.path.exists(s1_train_img_dir) else 0
    if img_count == 0:
        raise FileNotFoundError(
            f"\n[FATAL] Tidak ada file gambar ditemukan di: {s1_train_img_dir}\n"
            f"Penyebab: File gambar (.jpg) diabaikan oleh Git (.gitignore), sehingga folder 'dataset_annotated/images' kosong di server HPC.\n"
            f"Solusi:\n"
            f"1. Jika D-Fire tersimpan di folder lain pada server, jalankan:\n"
            f"   python train_two_stage.py --public /path/ke/folder/dfire\n"
            f"2. Atau salin folder dataset dari laptop ke server menggunakan rsync / scp:\n"
            f"   rsync -avzP Software/Development-2/dataset_annotated <user>@<server-ip>:~/Despro-2/despro-group-24/Software/Development-2/"
        )

    model = YOLO(base_model)

    results = model.train(
        data=data_yaml,
        epochs=epochs,
        patience=15,
        batch=batch,
        imgsz=imgsz,
        device=device,
        workers=workers,
        optimizer="AdamW",
        lr0=0.01,             # Standard LR untuk Stage 1
        lrf=0.01,
        cos_lr=True,
        warmup_epochs=3.0,
        mosaic=0.5,           # Moderate mosaic untuk variasi umum
        flipud=0.0,           # Matikan flip vertikal
        fliplr=0.5,
        amp=True,
        project=project_path,
        name=name
    )

    save_dir = str(results.save_dir) if hasattr(results, 'save_dir') else os.path.join(project_path, name)
    stage1_best = os.path.join(save_dir, "weights", "best.pt")

    # Salin ke models/stage1_best.pt sebagai cadangan
    models_dir = os.path.join(current_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    stage1_target = os.path.join(models_dir, "stage1_best.pt")
    if os.path.exists(stage1_best):
        shutil.copy2(stage1_best, stage1_target)
        print(f"\n[STAGE 1 SELESAI] Bobot terbaik Stage 1 disimpan ke: {stage1_target}")
        return stage1_target
    return stage1_best


# =====================================================================
# 4. STAGE 2 FINE-TUNING (WEBCAM ADAPTATION)
# =====================================================================

def train_stage_2(
    stage1_weights="models/stage1_best.pt",
    data_yaml="dataset_stage2/data.yaml",
    epochs=60,
    batch=32,
    imgsz=640,
    device=0 if torch.cuda.is_available() else "cpu",
    workers=8,
    project="runs/fire_detection",
    name="deezpro_stage2_finetune",
    output_model_name="best-fine.pt"
) -> str:
    """Melatih Stage 2: Fine-Tuning Webcam Adaptation (lr0=0.001, close_mosaic=15)."""
    print("\n" + "=" * 70)
    print("MEMULAI STAGE 2 FINE-TUNING: WEBCAM DOMAIN ADAPTATION")
    print(f"  - Inisialisasi   : {stage1_weights}")
    print(f"  - Dataset        : {data_yaml}")
    print(f"  - Epochs         : {epochs}")
    print(f"  - Batch Size     : {batch}")
    print(f"  - Learning Rate  : 0.001 (10x Lebih Kecil untuk Melindungi Fitur Api)")
    print(f"  - Proteksi Api   : mosaic=0.2, close_mosaic=15, hsv_s=0.3, hsv_v=0.2")
    print("=" * 70 + "\n")

    if not os.path.exists(stage1_weights):
        raise FileNotFoundError(f"Berkas bobot Stage 1 tidak ditemukan: {stage1_weights}")

    project_path = os.path.join(current_dir, project) if not os.path.isabs(project) else project
    model = YOLO(stage1_weights)

    results = model.train(
        data=data_yaml,
        epochs=epochs,
        patience=15,
        batch=batch,
        imgsz=imgsz,
        device=device,
        workers=workers,
        optimizer="AdamW",
        lr0=0.001,            # 10x lebih kecil agar tidak terjadi catastrophic forgetting
        lrf=0.01,
        cos_lr=True,
        warmup_epochs=3.0,
        mosaic=0.2,           # Nilai kecil agar api kecil tidak mengecil menjadi titik LED
        close_mosaic=15,      # Matikan mozaik di 15 epoch terakhir untuk menyempurnakan kotak bbox
        flipud=0.0,
        fliplr=0.5,
        hsv_h=0.015,
        hsv_s=0.3,            # Menjaga tepian spektrum kuning/oranye
        hsv_v=0.2,            # Menjaga kontras inti api putih
        amp=True,
        project=project_path,
        name=name
    )

    save_dir = str(results.save_dir) if hasattr(results, 'save_dir') else os.path.join(project_path, name)
    stage2_best = os.path.join(save_dir, "weights", "best.pt")

    # Simpan sebagai models/best-fine.pt
    models_dir = os.path.join(current_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    fine_target = os.path.join(models_dir, output_model_name)

    if os.path.exists(stage2_best):
        shutil.copy2(stage2_best, fine_target)
        print(f"\n" + "=" * 70)
        print("TWO-STAGE TRAINING TELAH SELESAI DENGAN SUKSES!")
        print(f"  - Bobot Model Akhir  : {fine_target}")
        print("=" * 70 + "\n")
        return fine_target
    return stage2_best


# =====================================================================
# 5. MAIN CLI RUNNER
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Script Two-Stage Training Deezpro (Base Learning + Webcam Fine-Tuning)"
    )
    parser.add_argument("--stage", choices=["all", "1", "2"], default="all",
                        help="Tahap yang ingin dijalankan: 'all' (1 lalu 2), '1' (Base saja), '2' (Fine-tuning saja)")
    parser.add_argument("--local", default="dataset_annotated",
                        help="Folder dataset kamera lokal (default: dataset_annotated)")
    parser.add_argument("--public", default="../Development/fire-detection/datasets",
                        help="Folder dataset publik D-Fire (default: ../Development/fire-detection/datasets)")
    parser.add_argument("--base-model", default="yolov8s.pt",
                        help="Backbone model untuk Stage 1 (default: yolov8s.pt)")
    parser.add_argument("--stage1-weights", default="models/stage1_best.pt",
                        help="Path bobot Stage 1 untuk input Stage 2 (default: models/stage1_best.pt)")
    parser.add_argument("--batch", type=int, default=32,
                        help="Batch size (default: 32 untuk HPC, gunakan 8 jika di laptop)")
    parser.add_argument("--workers", type=int, default=8,
                        help="Jumlah worker dataloader (default: 8 untuk HPC, gunakan 4 di laptop)")
    parser.add_argument("--epochs-s1", type=int, default=80,
                        help="Epochs untuk Stage 1 (default: 80)")
    parser.add_argument("--epochs-s2", type=int, default=60,
                        help="Epochs untuk Stage 2 (default: 60)")
    parser.add_argument("--output-model", default="best-fine.pt",
                        help="Nama file model akhir di folder models/ (default: best-fine.pt)")
    parser.add_argument("--skip-prep", action="store_true",
                        help="Lewati pembuatan dataset jika dataset_stage1 dan dataset_stage2 sudah ada")

    args = parser.parse_args()

    # Pastikan path absolut
    local_path = os.path.join(current_dir, args.local) if not os.path.isabs(args.local) else args.local
    public_path = os.path.join(current_dir, args.public) if not os.path.isabs(args.public) else args.public
    stage1_dir = os.path.join(current_dir, "dataset_stage1")
    stage2_dir = os.path.join(current_dir, "dataset_stage2")

    s1_yaml = os.path.join(stage1_dir, "data.yaml")
    s2_yaml = os.path.join(stage2_dir, "data.yaml")

    # Persiapkan dataset jika belum ada atau tidak di-skip
    if not args.skip_prep or not os.path.exists(s1_yaml) or not os.path.exists(s2_yaml):
        s1_yaml, s2_yaml = prepare_two_stage_datasets(
            local_dir=local_path,
            public_dir=public_path,
            stage1_dir=stage1_dir,
            stage2_dir=stage2_dir
        )

    device_str = 0 if torch.cuda.is_available() else "cpu"
    stage1_out_weight = args.stage1_weights

    # Eksekusi Stage 1
    if args.stage in ["all", "1"]:
        stage1_out_weight = train_stage_1(
            data_yaml=s1_yaml,
            base_model=args.base_model,
            epochs=args.epochs_s1,
            batch=args.batch,
            device=device_str,
            workers=args.workers
        )

    # Eksekusi Stage 2
    if args.stage in ["all", "2"]:
        train_stage_2(
            stage1_weights=stage1_out_weight,
            data_yaml=s2_yaml,
            epochs=args.epochs_s2,
            batch=args.batch,
            device=device_str,
            workers=args.workers,
            output_model_name=args.output_model
        )


if __name__ == "__main__":
    main()
