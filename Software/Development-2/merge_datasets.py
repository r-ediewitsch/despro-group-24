"""
=====================================================================
DEEZPRO - INTEGRATED DATASET PIPELINE (MERGE, DEGRADE & SPLIT)
=====================================================================
File: merge_datasets.py
Deskripsi: Tool terintegrasi untuk mempersiapkan dataset YOLO Deezpro:
           1. Stratified Split Data Mandiri (Positif Api & Negatif Background).
           2. Sensor Degradation pada data publik D-Fire (meniru noise,
              resolusi, dan kompresi JPEG webcam murah).
           3. (Opsional) Copy-Paste sintetis api ke latar belakang lokal.
           4. Pembagian Train-Val cerdas (Pure Local Benchmark pada Val).
           5. Menghasilkan folder 'dataset_combined/' lengkap dengan 'data.yaml'.
=====================================================================
"""

import os
import sys
import glob
import random
import shutil
import argparse
import numpy as np
import cv2


def degrade_image_sensor(
    img: np.ndarray,
    scale_range=(0.35, 0.50),
    jpeg_quality_range=(35, 60),
    noise_sigma_range=(3.0, 7.0),
    blur_prob=0.5
) -> np.ndarray:
    """
    Meniru artefak optik dan sensor webcam berbiaya rendah:
    1. Downscale (35%-50%) lalu upsample kembali ke resolusi asli.
    2. Menambahkan noise Gaussian / sensor ISO.
    3. Gaussian blur halus untuk mensimulasikan lensa plastik webcam.
    4. Kompresi JPEG kualitas rendah (35-60) untuk mensimulasikan USB stream webcam.
    """
    h, w = img.shape[:2]

    # 1. Downscale & Upscale (mengurangi detail mikro DSLR)
    scale = random.uniform(*scale_range)
    sw = max(16, int(w * scale))
    sh = max(16, int(h * scale))
    downsampled = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_AREA)
    upsampled = cv2.resize(downsampled, (w, h), interpolation=cv2.INTER_LINEAR)

    # 2. Gaussian Noise (ISO sensor noise)
    sigma = random.uniform(*noise_sigma_range)
    noise = np.random.normal(0, sigma, upsampled.shape).astype(np.float32)
    noisy = np.clip(upsampled.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # 3. Slight Blur
    if random.random() < blur_prob:
        noisy = cv2.GaussianBlur(noisy, (3, 3), 0)

    # 4. JPEG Compression Artifacts
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
    """
    Membuat sampel sintetis dengan memotong kotak api dari data publik (D-Fire)
    lalu menempelkannya ke foto latar belakang kosong (negatif) arena Deezpro.
    Tepian potongan di-blend dengan Gaussian blur halus agar natural.
    """
    if not bg_pairs or not fg_pairs or num_samples <= 0:
        return 0

    print(f"\n[Copy-Paste] Menghasilkan {num_samples} foto sintetis api di arena Deezpro...")
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

        # Baca label bounding box dari fg
        with open(fg_lbl_path, "r", encoding="utf-8") as f:
            lines = [line.strip().split() for line in f if len(line.strip().split()) >= 5]

        if not lines:
            continue

        new_labels = []
        canvas = bg.copy()

        # Tempelkan 1 hingga 2 api per gambar sintetis
        selected_boxes = random.sample(lines, min(len(lines), random.randint(1, 2)))

        for box in selected_boxes:
            cls_id = int(box[0])
            cx, cy, bw_norm, bh_norm = map(float, box[1:5])

            # Koordinat piksel di gambar asal (fg)
            px1 = max(0, int((cx - bw_norm / 2) * fw))
            py1 = max(0, int((cy - bh_norm / 2) * fh))
            px2 = min(fw, int((cx + bw_norm / 2) * fw))
            py2 = min(fh, int((cy + bh_norm / 2) * fh))

            crop = fg[py1:py2, px1:px2]
            ch, cw = crop.shape[:2]
            if ch < 10 or cw < 10:
                continue

            # Berikan variasi skala acak (0.6x sampai 1.2x)
            scale = random.uniform(0.6, 1.2)
            nw = max(15, min(int(cw * scale), int(bw * 0.45)))
            nh = max(15, min(int(ch * scale), int(bh * 0.45)))
            crop_resized = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_LINEAR)

            # Masker dengan Gaussian blur pada tepian
            mask = np.ones((nh, nw), dtype=np.float32)
            border = max(2, min(nh, nw) // 10)
            mask[:border, :] = 0
            mask[-border:, :] = 0
            mask[:, :border] = 0
            mask[:, -border:] = 0
            mask = cv2.GaussianBlur(mask, (7, 7), 2.0)
            mask_3c = np.repeat(mask[:, :, np.newaxis], 3, axis=2)

            # Tentukan posisi tempel acak di arena latar belakang
            tx = random.randint(10, max(11, bw - nw - 10))
            ty = random.randint(10, max(11, bh - nh - 10))

            roi = canvas[ty:ty+nh, tx:tx+nw]
            blended = (crop_resized.astype(np.float32) * mask_3c + roi.astype(np.float32) * (1.0 - mask_3c)).astype(np.uint8)
            canvas[ty:ty+nh, tx:tx+nw] = blended

            # Simpan koordinat baru dalam format YOLO
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

    print(f"      -> Berhasil membuat {generated_count} sampel sintetis Copy-Paste.")
    return generated_count


def merge_datasets(
    local_dir="dataset_annotated",
    public_dir="../Development/fire-detection/datasets",
    output_dir="dataset_combined",
    split_ratio=0.8,
    max_public_samples=1200,
    degrade_public=True,
    pure_local_val=True,
    num_copy_paste=0,
    clean_output=True
):
    """Pipeline integrasi dataset: Stratified local, Public degradation, Copy-Paste, dan Smart Split."""

    current_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.isabs(local_dir):
        cand = os.path.join(current_dir, local_dir)
        if os.path.exists(cand):
            local_dir = cand
    if not os.path.isabs(public_dir):
        cand = os.path.normpath(os.path.join(current_dir, public_dir))
        if os.path.exists(cand):
            public_dir = cand
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(current_dir, output_dir)

    print("\n" + "=" * 70)
    print("DEEZPRO - INTEGRATED DATASET PIPELINE (MERGE, DEGRADE, & SPLIT)")
    print("=" * 70)
    print(f"  - Dataset Mandiri      : {local_dir}")
    print(f"  - Dataset Publik       : {public_dir}")
    print(f"  - Direktori Output     : {output_dir}")
    print(f"  - Kuota Data Publik    : Maks {max_public_samples} sampel")
    print(f"  - Degradasi Sensor     : {'AKTIF (Downscale, Noise, JPEG Blur)' if degrade_public else 'NONAKTIF'}")
    print(f"  - Mode Set Validasi    : {'100% PURE KAMERA LOKAL (Rekomendasi Terbaik)' if pure_local_val else 'Campuran'}")
    print(f"  - Copy-Paste Sintetis  : {num_copy_paste} sampel")
    print("=" * 70 + "\n")

    train_img = os.path.join(output_dir, "images", "train")
    val_img = os.path.join(output_dir, "images", "val")
    train_lbl = os.path.join(output_dir, "labels", "train")
    val_lbl = os.path.join(output_dir, "labels", "val")

    # Bersihkan direktori lama jika diminta
    if clean_output and os.path.exists(output_dir):
        print(f"[0/4] Membersihkan folder output lama: {output_dir}...")
        for sub in [train_img, val_img, train_lbl, val_lbl]:
            if os.path.exists(sub):
                shutil.rmtree(sub)

    for d in (train_img, val_img, train_lbl, val_lbl):
        os.makedirs(d, exist_ok=True)

    random.seed(42)
    np.random.seed(42)

    # -------------------------------------------------------------
    # 1. Kumpulkan Data Mandiri & Stratifikasi (Positif vs Negatif)
    # -------------------------------------------------------------
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

            # Cek isi label: kosong = latar belakang (negatif), ada isi = ada api (positif)
            is_empty = True
            with open(txt_p, "r", encoding="utf-8") as f:
                if f.read().strip():
                    is_empty = False

            if is_empty:
                local_negative_pairs.append((img_p, txt_p))
            else:
                local_positive_pairs.append((img_p, txt_p))

    total_local = len(local_positive_pairs) + len(local_negative_pairs)
    print(f"[1/4] Dataset Kamera Mandiri:")
    print(f"      - Positif (Api/Asap)        : {len(local_positive_pairs)} foto")
    print(f"      - Negatif (Latar Belakang)  : {len(local_negative_pairs)} foto")
    print(f"      - Total                     : {total_local} foto")

    # Stratified Split Data Mandiri
    random.shuffle(local_positive_pairs)
    random.shuffle(local_negative_pairs)

    pos_split = int(len(local_positive_pairs) * split_ratio)
    neg_split = int(len(local_negative_pairs) * split_ratio)

    loc_train_pos = local_positive_pairs[:pos_split]
    loc_val_pos = local_positive_pairs[pos_split:]

    loc_train_neg = local_negative_pairs[:neg_split]
    loc_val_neg = local_negative_pairs[neg_split:]

    local_train = loc_train_pos + loc_train_neg
    local_val = loc_val_pos + loc_val_neg

    print(f"      -> Pembagian Mandiri : {len(local_train)} Train ({len(loc_train_pos)} pos, {len(loc_train_neg)} neg) | "
          f"{len(local_val)} Val ({len(loc_val_pos)} pos, {len(loc_val_neg)} neg)")

    # -------------------------------------------------------------
    # 2. Kumpulkan Dataset Publik (D-Fire)
    # -------------------------------------------------------------
    public_pairs = []
    if os.path.exists(public_dir):
        pub_img_dir = os.path.join(public_dir, "images")
        pub_lbl_dir = os.path.join(public_dir, "labels")

        pub_imgs = glob.glob(os.path.join(pub_img_dir, "**", "*.jpg"), recursive=True)
        for img_p in pub_imgs:
            base = os.path.splitext(os.path.basename(img_p))[0]
            sub = os.path.basename(os.path.dirname(img_p))
            lbl_candidate = os.path.join(pub_lbl_dir, sub, base + ".txt")
            if not os.path.exists(lbl_candidate):
                lbl_candidate = os.path.join(pub_lbl_dir, base + ".txt")

            if os.path.exists(lbl_candidate):
                public_pairs.append((img_p, lbl_candidate))

    print(f"\n[2/4] Dataset Publik (D-Fire):")
    print(f"      - Ditemukan total : {len(public_pairs)} foto")

    if max_public_samples and len(public_pairs) > max_public_samples:
        random.shuffle(public_pairs)
        public_pairs = public_pairs[:max_public_samples]
        print(f"      -> Disubsample menjadi : {len(public_pairs)} foto (Pencegahan Imbalance)")

    # -------------------------------------------------------------
    # 3. Eksekusi Penyalinan & Sensor Degradation
    # -------------------------------------------------------------
    print(f"\n[3/4] Memproses & Menyalin Berkas ke '{output_dir}'...")

    # Salin data lokal (tanpa degradasi karena sudah asli webcam)
    for img_p, lbl_p in local_train:
        fname = "local_" + os.path.basename(img_p)
        shutil.copy2(img_p, os.path.join(train_img, fname))
        shutil.copy2(lbl_p, os.path.join(train_lbl, os.path.splitext(fname)[0] + ".txt"))

    for img_p, lbl_p in local_val:
        fname = "local_" + os.path.basename(img_p)
        shutil.copy2(img_p, os.path.join(val_img, fname))
        shutil.copy2(lbl_p, os.path.join(val_lbl, os.path.splitext(fname)[0] + ".txt"))

    print(f"      [✓] {len(local_train)} foto lokal tersalin ke train.")
    print(f"      [✓] {len(local_val)} foto lokal tersalin ke val (Pure Benchmark).")

    # Salin data publik (dengan degradasi sensor jika aktif)
    if pure_local_val:
        pub_train = public_pairs
        pub_val = []
    else:
        random.shuffle(public_pairs)
        p_split = int(len(public_pairs) * split_ratio)
        pub_train = public_pairs[:p_split]
        pub_val = public_pairs[p_split:]

    if len(pub_train) > 0:
        print(f"      [-] Memproses & mendegradasi {len(pub_train)} foto publik untuk train...")
        for idx, (img_p, lbl_p) in enumerate(pub_train):
            fname = "pub_" + os.path.basename(img_p)
            dst_img = os.path.join(train_img, fname)
            dst_lbl = os.path.join(train_lbl, os.path.splitext(fname)[0] + ".txt")

            if degrade_public:
                img = cv2.imread(img_p)
                if img is not None:
                    deg = degrade_image_sensor(img)
                    cv2.imwrite(dst_img, deg)
                else:
                    shutil.copy2(img_p, dst_img)
            else:
                shutil.copy2(img_p, dst_img)

            shutil.copy2(lbl_p, dst_lbl)

        for img_p, lbl_p in pub_val:
            fname = "pub_" + os.path.basename(img_p)
            dst_img = os.path.join(val_img, fname)
            dst_lbl = os.path.join(val_lbl, os.path.splitext(fname)[0] + ".txt")
            shutil.copy2(img_p, dst_img)
            shutil.copy2(lbl_p, dst_lbl)

        print(f"      [✓] {len(pub_train)} foto publik berhasil diproses ke train.")
        if pub_val:
            print(f"      [✓] {len(pub_val)} foto publik tersalin ke val.")
    else:
        print("      [i] Tidak ada foto publik yang diproses (hanya menggunakan data lokal).")

    # -------------------------------------------------------------
    # 4. (Opsional) Copy-Paste Sintetis
    # -------------------------------------------------------------
    synth_count = 0
    if num_copy_paste > 0:
        synth_count = generate_copy_paste_samples(
            bg_pairs=loc_train_neg,
            fg_pairs=public_pairs,
            num_samples=num_copy_paste,
            out_img_dir=train_img,
            out_lbl_dir=train_lbl,
            prefix="synth"
        )

    # -------------------------------------------------------------
    # 5. Buat data.yaml Otomatis
    # -------------------------------------------------------------
    yaml_path = os.path.join(output_dir, "data.yaml")
    abs_out = os.path.abspath(output_dir).replace("\\", "/")

    yaml_content = f"""# Deezpro Combined Fire & Smoke Dataset Configuration
# Generated by integrated merge_datasets.py pipeline
# Local Samples: {total_local} ({len(local_train)} train, {len(local_val)} val)
# Public Samples: {len(public_pairs)} (degraded to webcam profile)
# Synthetic Copy-Paste: {synth_count}

path: {abs_out}
train: images/train
val: images/val

names:
  0: smoke
  1: fire
"""
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)

    total_final_train = len(local_train) + len(pub_train) + synth_count
    total_final_val = len(local_val) + len(pub_val)

    print("\n" + "=" * 70)
    print("INTEGRASI DATASET SELESAI DENGAN SUKSES!")
    print("=" * 70)
    print(f"  - Total Foto Training : {total_final_train} foto")
    print(f"      * Lokal Asli      : {len(local_train)} foto ({len(loc_train_pos)} pos, {len(loc_train_neg)} neg)")
    print(f"      * Publik D-Fire   : {len(pub_train)} foto (Degradasi Sensor)")
    if synth_count > 0:
        print(f"      * Sintetis Paste  : {synth_count} foto")
    print(f"  - Total Foto Validasi : {total_final_val} foto")
    print(f"      * Lokal Benchmark : {len(local_val)} foto ({len(loc_val_pos)} pos, {len(loc_val_neg)} neg)")
    if pub_val:
        print(f"      * Publik Val      : {len(pub_val)} foto")
    print(f"  - File data.yaml      : {yaml_path}")
    print("=" * 70 + "\n")
    print("Siap untuk training model:")
    print("  python train_model.py\n")


def main():
    parser = argparse.ArgumentParser(
        description="Tool Integrasi Dataset Deezpro (Degradasi Sensor, Stratifikasi, & Merging)"
    )
    parser.add_argument("--local", default="dataset_annotated",
                        help="Folder dataset mandiri (default: dataset_annotated)")
    parser.add_argument("--public", default="../Development/fire-detection/datasets",
                        help="Folder dataset publik lama (default: ../Development/fire-detection/datasets)")
    parser.add_argument("--out", default="dataset_combined",
                        help="Folder output dataset gabungan (default: dataset_combined)")
    parser.add_argument("--split", type=float, default=0.8,
                        help="Rasio split data lokal train/val (default: 0.8)")
    parser.add_argument("--max-pub", type=int, default=1200,
                        help="Batas maksimal sampel publik (default: 1200)")
    parser.add_argument("--no-degrade", action="store_true",
                        help="Matikan degradasi sensor pada data publik")
    parser.add_argument("--mixed-val", action="store_true",
                        help="Campur data publik ke set validasi (default: False, val murni lokal)")
    parser.add_argument("--copy-paste", type=int, default=0,
                        help="Jumlah sampel sintetis copy-paste yang ingin dibuat (default: 0)")
    parser.add_argument("--no-clean", action="store_true",
                        help="Jangan bersihkan folder output sebelum menyalin")

    args = parser.parse_args()

    merge_datasets(
        local_dir=args.local,
        public_dir=args.public,
        output_dir=args.out,
        split_ratio=args.split,
        max_public_samples=args.max_pub,
        degrade_public=not args.no_degrade,
        pure_local_val=not args.mixed_val,
        num_copy_paste=args.copy_paste,
        clean_output=not args.no_clean
    )


if __name__ == "__main__":
    main()
