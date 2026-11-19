"""
merge_datasets.py
-----------------
Tool penggabungan Dataset Mandiri (Kamera Deezpro) dengan Dataset Publik (D-Fire):
- Mencegah bentrok nama file dengan prefix ('local_' vs 'pub_').
- Memastikan pemetaan kelas seragam (0: smoke, 1: fire).
- Menerapkan pembagian proporsional (80% Train, 20% Val) untuk KEDUA sumber data:
    * 80% lokal + sampel publik masuk ke images/train & labels/train
    * 20% lokal + sampel publik masuk ke images/val & labels/val
- Mendukung subsampling data publik agar data kamera lokal (~300 foto) tidak tenggelam
  oleh ribuan data publik (D-Fire memiliki 14.000+ foto).
- Menghasilkan folder 'dataset_combined/' lengkap dengan 'data.yaml'.
"""

import os
import sys
import glob
import random
import shutil
import argparse


def merge_datasets(local_dir="dataset_annotated",
                   public_dir="../Development/fire-detection/datasets",
                   output_dir="dataset_combined",
                   split_ratio=0.8,
                   max_public_samples=1500):

    print("\n" + "=" * 65)
    print("PENGGABUNGAN DATASET MANDIRI & PUBLIK (DEEZPRO)")
    print(f"  - Dataset Mandiri : {local_dir}")
    print(f"  - Dataset Publik  : {public_dir}")
    print(f"  - Folder Output   : {output_dir}")
    print(f"  - Proporsi Split  : {int(split_ratio*100)}% Train / {int((1-split_ratio)*100)}% Val")
    print(f"  - Kuota Publik    : Maks {max_public_samples} sampel (Pencegahan Imbalance)")
    print("=" * 65 + "\n")

    # Siapkan direktori output
    train_img = os.path.join(output_dir, "images", "train")
    val_img = os.path.join(output_dir, "images", "val")
    train_lbl = os.path.join(output_dir, "labels", "train")
    val_lbl = os.path.join(output_dir, "labels", "val")

    for d in (train_img, val_img, train_lbl, val_lbl):
        os.makedirs(d, exist_ok=True)

    random.seed(42)

    # -------------------------------------------------------------
    # 1. Kumpulkan Dataset Mandiri (Local Camera)
    # -------------------------------------------------------------
    local_pairs = []
    # Cek apakah local_dir adalah dataset_annotated atau dataset_yolo
    loc_lbl_dir = os.path.join(local_dir, "labels")
    loc_img_dir = os.path.join(local_dir, "images")

    if os.path.exists(loc_lbl_dir):
        # Format dataset_annotated (labels/*.txt)
        all_txt = glob.glob(os.path.join(loc_lbl_dir, "**", "*.txt"), recursive=True)
        for txt_p in all_txt:
            base = os.path.splitext(os.path.basename(txt_p))[0]
            # Cari gambar pasangan
            img_candidates = [
                os.path.join(loc_img_dir, base + ".jpg"),
                os.path.join(loc_img_dir, base + ".png"),
                os.path.join(local_dir, base + ".jpg")
            ]
            for c in img_candidates:
                if os.path.exists(c):
                    local_pairs.append((c, txt_p))
                    break

    print(f"[1/3] Ditemukan {len(local_pairs)} pasangan data kamera mandiri.")

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
            # Cari file label di folder labels yang bersesuaian (train/val/test)
            sub = os.path.basename(os.path.dirname(img_p))  # misal 'train' atau 'val'
            lbl_candidate = os.path.join(pub_lbl_dir, sub, base + ".txt")
            if not os.path.exists(lbl_candidate):
                lbl_candidate = os.path.join(pub_lbl_dir, base + ".txt")

            if os.path.exists(lbl_candidate):
                public_pairs.append((img_p, lbl_candidate))

    print(f"[2/3] Ditemukan {len(public_pairs)} pasangan data publik.")

    # Lakukan subsampling jika data publik terlalu mendominasi
    if max_public_samples and len(public_pairs) > max_public_samples:
        random.shuffle(public_pairs)
        public_pairs = public_pairs[:max_public_samples]
        print(f"      -> Dibatasi menjadi {len(public_pairs)} sampel untuk menjaga keseimbangan sensor.")

    # -------------------------------------------------------------
    # 3. Stratified Split (80% Train, 20% Val untuk KEDUA sumber)
    # -------------------------------------------------------------
    print("[3/3] Melakukan stratified train/val split dan penyalinan file...")

    def split_and_copy(pairs, prefix):
        random.shuffle(pairs)
        s_idx = int(len(pairs) * split_ratio)
        train_slice = pairs[:s_idx]
        val_slice = pairs[s_idx:]

        for img_p, lbl_p in train_slice:
            fname = prefix + "_" + os.path.basename(img_p)
            shutil.copy(img_p, os.path.join(train_img, fname))
            shutil.copy(lbl_p, os.path.join(train_lbl, os.path.splitext(fname)[0] + ".txt"))

        for img_p, lbl_p in val_slice:
            fname = prefix + "_" + os.path.basename(img_p)
            shutil.copy(img_p, os.path.join(val_img, fname))
            shutil.copy(lbl_p, os.path.join(val_lbl, os.path.splitext(fname)[0] + ".txt"))

        return len(train_slice), len(val_slice)

    tr_loc, vl_loc = split_and_copy(local_pairs, "local")
    tr_pub, vl_pub = split_and_copy(public_pairs, "pub")

    # Buat file data.yaml otomatis
    yaml_path = os.path.join(output_dir, "data.yaml")
    abs_out = os.path.abspath(output_dir).replace("\\", "/")
    yaml_content = f"""# Deezpro Combined Fire & Smoke Dataset Configuration
# Merged from Local Camera ({len(local_pairs)} samples) + Public Dataset ({len(public_pairs)} samples)

path: {abs_out}
train: images/train
val: images/val

names:
  0: smoke
  1: fire
"""
    with open(yaml_path, "w") as f:
        f.write(yaml_content)

    total_train = tr_loc + tr_pub
    total_val = vl_loc + vl_pub

    print("\n" + "=" * 65)
    print("PENGGABUNGAN SELESAI DENGAN SUKSES!")
    print(f"  - Total Data Training   : {total_train} foto ({tr_loc} mandiri + {tr_pub} publik)")
    print(f"  - Total Data Validasi   : {total_val} foto ({vl_loc} mandiri + {vl_pub} publik)")
    print(f"  - Konfigurasi YOLO      : {yaml_path}")
    print("=" * 65 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Tool Penggabungan Dataset Mandiri & Publik (Deezpro)")
    parser.add_argument("--local", default="dataset_annotated",
                        help="Folder dataset mandiri hasil anotasi (default: dataset_annotated)")
    parser.add_argument("--public", default="../Development/fire-detection/datasets",
                        help="Folder dataset publik lama (default: ../Development/fire-detection/datasets)")
    parser.add_argument("--out", default="dataset_combined",
                        help="Folder output dataset gabungan (default: dataset_combined)")
    parser.add_argument("--split", type=float, default=0.8,
                        help="Rasio split data latih (default: 0.8)")
    parser.add_argument("--max-pub", type=int, default=1500,
                        help="Batas maksimal sampel publik agar tidak menenggelamkan data lokal (default: 1500)")
    args = parser.parse_args()

    merge_datasets(
        local_dir=args.local,
        public_dir=args.public,
        output_dir=args.out,
        split_ratio=args.split,
        max_public_samples=args.max_pub
    )


if __name__ == "__main__":
    main()
