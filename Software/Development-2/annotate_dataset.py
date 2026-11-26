"""
annotate_dataset.py
-------------------
Tool anotasi interaktif & pembentukan dataset YOLO (Fase 2 & Fase 3 STEPS.md):
- Pre-labeling otomatis menggunakan Zero-Shot Foundation Model:
    * Grounding DINO ('IDEA-Research/grounding-dino-tiny')
    * Florence-2 ('microsoft/Florence-2-base')
    * Model YOLO ('models/best.pt')
- Human-in-the-loop:
    * Klik kiri & seret mouse untuk menggambar bounding box baru.
    * Klik kanan pada kotak untuk menghapus (hapus false positive / distractor).
    * Tombol '0': Ubah kelas aktif / kotak menjadi '0: smoke'.
    * Tombol '1': Ubah kelas aktif / kotak menjadi '1: fire'.
    * Tombol 'B': Tandai gambar sebagai background murni (file .txt kosong 0-byte).
    * Tombol 'C': Hapus seluruh bounding box di frame ini.
    * Tombol 'Z': Undo aksi kotak terakhir.
    * Tombol 'SPACE' / 'ENTER': Simpan label dan lanjut ke gambar berikutnya.
    * Tombol 'P': Kembali ke gambar sebelumnya.
    * Tombol 'S': Ekspor & split dataset ke folder standar YOLO (80% train / 20% val) beserta data.yaml!
"""

import os
import sys
import glob
import random
import shutil
import argparse
import numpy as np
import cv2

# Pastikan path modul lokal dikenali
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

CLASSES = {
    0: {"name": "smoke", "color": (255, 200, 0)},   # Cyan / Light Amber (BGR)
    1: {"name": "fire",  "color": (0, 69, 255)}    # Red / Orange-red (BGR)
}


class Box:
    def __init__(self, cls_id, x1, y1, x2, y2, conf=1.0):
        self.cls_id = int(cls_id)
        # Pastikan koordinat min/max terurut
        self.x1 = min(int(x1), int(x2))
        self.y1 = min(int(y1), int(y2))
        self.x2 = max(int(x1), int(x2))
        self.y2 = max(int(y1), int(y2))
        self.conf = float(conf)

    def to_yolo(self, img_w, img_h):
        """Konversi koordinat piksel ke format ternormalisasi YOLO (0.0 - 1.0)."""
        x_center = ((self.x1 + self.x2) / 2.0) / img_w
        y_center = ((self.y1 + self.y2) / 2.0) / img_h
        width = (self.x2 - self.x1) / img_w
        height = (self.y2 - self.y1) / img_h
        return self.cls_id, x_center, y_center, width, height

    @classmethod
    def from_yolo(cls, cls_id, xc, yc, w, h, img_w, img_h, conf=1.0):
        """Konversi format ternormalisasi YOLO ke koordinat piksel absolut."""
        x1 = int((xc - w / 2.0) * img_w)
        y1 = int((yc - h / 2.0) * img_h)
        x2 = int((xc + w / 2.0) * img_w)
        y2 = int((yc + h / 2.0) * img_h)
        return cls(cls_id, x1, y1, x2, y2, conf)

    def contains(self, px, py):
        return self.x1 <= px <= self.x2 and self.y1 <= py <= self.y2


class DatasetAnnotator:
    def __init__(self, raw_dir="dataset_raw", output_dir="dataset_annotated",
                 backend="grounding_dino", model_id=None, conf_thresh=0.30):
        self.raw_dir = raw_dir
        self.output_dir = output_dir
        self.backend = backend.lower() if backend else "none"
        self.model_id = model_id
        self.conf_thresh = conf_thresh

        self.img_dir = os.path.join(output_dir, "images")
        self.lbl_dir = os.path.join(output_dir, "labels")
        os.makedirs(self.img_dir, exist_ok=True)
        os.makedirs(self.lbl_dir, exist_ok=True)

        self.labeler = None
        # Pre-labeler diinisialisasi secara lazy (on-demand) hanya jika ditemukan gambar yang belum memiliki file label .txt

        # Ambil daftar gambar
        self.image_paths = self._gather_images()
        if not self.image_paths:
            print(f"[Annotator] Tidak ada gambar ditemukan di '{raw_dir}' maupun di '{self.img_dir}'!")
            return

        print(f"[Annotator] Total {len(self.image_paths)} gambar siap dianotasi.")

        # Status interaktif
        self.idx = 0
        self.current_boxes = []
        self.active_cls = 1  # Default: 1 (fire)
        self.is_drawing = False
        self.start_pt = None
        self.current_pt = None
        self.selected_box_idx = -1
        self.is_background = False
        self.undo_stack = []

    def _get_labeler(self):
        """Memuat Foundation Model secara on-demand hanya jika ada gambar yang belum berlabel."""
        if self.labeler is None and self.backend != "none":
            try:
                from src.vision.foundation_labeler import FoundationLabeler
                print(f"[Annotator] Gambar baru tanpa label terdeteksi! Menginisialisasi Foundation Model backend='{self.backend}'...")
                self.labeler = FoundationLabeler(backend=self.backend, model_id=self.model_id)
                print(f"[Annotator] Pre-labeler aktif dengan threshold {self.conf_thresh}.")
            except Exception as e:
                print(f"[Annotator WARNING] Gagal menginisialisasi pre-labeler ({e}).")
                print("[Annotator] Berjalan dalam mode manual murni.")
                self.backend = "none"
        return self.labeler

    def _gather_images(self):
        exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
        imgs = []

        # Tangani jika raw_dir berupa list (multiple folders) atau string tunggal
        raw_dirs = self.raw_dir if isinstance(self.raw_dir, list) else [self.raw_dir]

        for r_dir in raw_dirs:
            if os.path.exists(r_dir):
                for ext in exts:
                    imgs.extend(glob.glob(os.path.join(r_dir, "**", ext), recursive=True))

        # 2. Smart Fallback: Jika seluruh raw_dirs kosong, baca dari output_dir/images
        if not imgs and os.path.exists(self.img_dir):
            print(f"[Annotator] Folder sumber kosong. Menggunakan citra yang ada di '{self.img_dir}'...")
            for ext in exts:
                imgs.extend(glob.glob(os.path.join(self.img_dir, "**", ext), recursive=True))

        # Hapus duplikat jika ada dan urutkan
        imgs = sorted(list(set(imgs)))
        return imgs

    def _mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.is_drawing = True
            self.start_pt = (x, y)
            self.current_pt = (x, y)
            self.selected_box_idx = -1

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.is_drawing:
                self.current_pt = (x, y)

        elif event == cv2.EVENT_LBUTTONUP:
            if self.is_drawing and self.start_pt:
                self.is_drawing = False
                x1, y1 = self.start_pt
                x2, y2 = x, y
                if abs(x2 - x1) > 8 and abs(y2 - y1) > 8:
                    self._push_undo()
                    new_box = Box(self.active_cls, x1, y1, x2, y2, conf=1.0)
                    self.current_boxes.append(new_box)
                    self.is_background = False
                self.start_pt = None
                self.current_pt = None

        elif event == cv2.EVENT_RBUTTONDOWN:
            # Hapus kotak yang diklik kanan (Koreksi False Positive)
            for i in reversed(range(len(self.current_boxes))):
                if self.current_boxes[i].contains(x, y):
                    self._push_undo()
                    del self.current_boxes[i]
                    print(f"[Anotasi] Kotak ke-{i+1} dihapus.")
                    break

    def _push_undo(self):
        snapshot = [Box(b.cls_id, b.x1, b.y1, b.x2, b.y2, b.conf) for b in self.current_boxes]
        self.undo_stack.append((snapshot, self.is_background))

    def _undo(self):
        if self.undo_stack:
            snapshot, bg = self.undo_stack.pop()
            self.current_boxes = snapshot
            self.is_background = bg
            print("[Anotasi] Undo berhasil.")

    def _load_annotations_for_current(self, img_bgr, img_w, img_h):
        img_path = self.image_paths[self.idx]
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        lbl_path = os.path.join(self.lbl_dir, base_name + ".txt")

        self.current_boxes = []
        self.undo_stack = []

        # 1. Cek jika sudah ada file label tersimpan sebelumnya
        if os.path.exists(lbl_path):
            if os.path.getsize(lbl_path) == 0:
                self.is_background = True
                return
            self.is_background = False
            with open(lbl_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls_id = int(parts[0])
                        xc, yc, w, h = map(float, parts[1:5])
                        self.current_boxes.append(Box.from_yolo(cls_id, xc, yc, w, h, img_w, img_h))
            return

        # 2. Cek apakah gambar berasal dari subfolder 'negative'
        if "negative" in img_path.lower():
            self.is_background = True
            return

        # 3. Pre-labeling otomatis menggunakan Foundation Model (HANYA JIKA BELUM ADA LABEL)
        self.is_background = False
        labeler = self._get_labeler()
        if labeler is not None:
            try:
                preds = labeler.predict(img_bgr, box_threshold=self.conf_thresh)
                for p in preds:
                    b = p["box"]
                    self.current_boxes.append(Box(p["class_id"], b[0], b[1], b[2], b[3], p["confidence"]))
                if len(preds) == 0:
                    self.is_background = True
            except Exception as e:
                print(f"[Annotator] Pre-labeling gagal untuk {base_name}: {e}")

    def _save_current_annotation(self, img, img_w, img_h):
        img_path = self.image_paths[self.idx]
        base_name = os.path.splitext(os.path.basename(img_path))[0]

        # Salin / simpan gambar ke dataset_annotated/images
        target_img = os.path.join(self.img_dir, base_name + ".jpg")
        cv2.imwrite(target_img, img, [cv2.IMWRITE_JPEG_QUALITY, 95])

        # Simpan file label YOLO ke dataset_annotated/labels
        target_lbl = os.path.join(self.lbl_dir, base_name + ".txt")

        if self.is_background or len(self.current_boxes) == 0:
            # File kosong 0-byte (Negative Background / Hard Negative)
            open(target_lbl, "w").close()
            print(f"[Simpan] {base_name}.txt -> NEGATIVE/BACKGROUND (0 byte)")
        else:
            with open(target_lbl, "w") as f:
                for box in self.current_boxes:
                    cls_id, xc, yc, w, h = box.to_yolo(img_w, img_h)
                    f.write(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")
            print(f"[Simpan] {base_name}.txt -> {len(self.current_boxes)} bounding box")

    def run(self):
        if not self.image_paths:
            return

        win_name = "Anotator Dataset YOLO - Deezpro (Fase 2 STEPS.md)"
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win_name, 1100, 680)
        cv2.setMouseCallback(win_name, self._mouse_callback)

        print("\n" + "=" * 65)
        print("TOOL ANOTASI & HUMAN-IN-THE-LOOP AKTIF")
        print(f"Pre-labeler: {self.backend.upper() if self.labeler else 'MANUAL'}")
        print("Kontrol Keyboard:")
        print("  [Drag Mouse Kiri] : Buat Bounding Box Baru")
        print("  [Klik Kanan Box]  : Hapus Bounding Box")
        print("  [0]               : Pilih Kelas '0: SMOKE'")
        print("  [1]               : Pilih Kelas '1: FIRE'")
        print("  [B]               : Tandai sebagai BACKGROUND MURNI (0 byte .txt)")
        print("  [C]               : Hapus semua kotak (Clear)")
        print("  [Z]               : Undo")
        print("  [SPACE / ENTER]   : Simpan label & lanjut ke gambar berikutnya")
        print("  [P / Panah Kiri]  : Kembali ke gambar sebelumnya")
        print("  [S]               : Ekspor & Split ke folder YOLO (80% train / 20% val)")
        print("  [Q / ESC]         : Keluar")
        print("=" * 65 + "\n")

        last_loaded_idx = -1

        while True:
            img_path = self.image_paths[self.idx]
            raw_img = cv2.imread(img_path)
            if raw_img is None:
                print(f"[Annotator] Gagal membaca {img_path}, lewati...")
                self.idx = min(self.idx + 1, len(self.image_paths) - 1)
                continue

            h, w = raw_img.shape[:2]

            # Muat anotasi saat pindah gambar
            if self.idx != last_loaded_idx:
                self._load_annotations_for_current(raw_img, w, h)
                last_loaded_idx = self.idx

            display = raw_img.copy()

            # Render bounding boxes
            if not self.is_background:
                for b_i, box in enumerate(self.current_boxes):
                    cls_meta = CLASSES.get(box.cls_id, {"name": f"id_{box.cls_id}", "color": (0, 255, 0)})
                    color = cls_meta["color"]

                    # Kotak
                    cv2.rectangle(display, (box.x1, box.y1), (box.x2, box.y2), color, 2)

                    # Label badge
                    label_text = f"{box.cls_id}: {cls_meta['name']}"
                    (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    badge_y1 = max(0, box.y1 - th - 6)
                    cv2.rectangle(display, (box.x1, badge_y1), (box.x1 + tw + 6, box.y1), color, -1)
                    cv2.putText(display, label_text, (box.x1 + 3, box.y1 - 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Gambar draf box saat mouse sedang ditarik
            if self.is_drawing and self.start_pt and self.current_pt:
                cls_color = CLASSES[self.active_cls]["color"]
                cv2.rectangle(display, self.start_pt, self.current_pt, cls_color, 1)

            # Render HUD Overlay
            self._draw_hud(display, img_path, w, h)

            cv2.imshow(win_name, display)
            key = cv2.waitKey(20) & 0xFF

            if key in (27, ord('q'), ord('Q')):
                break
            elif key in (32, 13):  # SPACE or ENTER -> Save & Next
                self._save_current_annotation(raw_img, w, h)
                if self.idx < len(self.image_paths) - 1:
                    self.idx += 1
                else:
                    print("[Annotator] Ini adalah gambar terakhir dalam dataset!")
            elif key in (ord('p'), ord('P')):  # Previous image
                if self.idx > 0:
                    self.idx -= 1
            elif key == ord('0'):  # Class smoke
                self.active_cls = 0
                print("[Kelas Aktif] 0: SMOKE")
            elif key == ord('1'):  # Class fire
                self.active_cls = 1
                print("[Kelas Aktif] 1: FIRE")
            elif key in (ord('b'), ord('B')):  # Mark as background
                self._push_undo()
                self.is_background = not self.is_background
                if self.is_background:
                    self.current_boxes = []
                    print("[Anotasi] Ditandai sebagai BACKGROUND (0-byte)")
            elif key in (ord('c'), ord('C')):  # Clear all boxes
                self._push_undo()
                self.current_boxes = []
                self.is_background = False
                print("[Anotasi] Semua bounding box dibersihkan.")
            elif key in (ord('z'), ord('Z')):  # Undo
                self._undo()
            elif key in (ord('s'), ord('S')):  # Split dataset
                self.export_yolo_dataset()

        cv2.destroyAllWindows()

    def _draw_hud(self, img, img_path, w, h):
        # Top banner
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (w, 65), (20, 20, 20), -1)
        cv2.rectangle(overlay, (0, h - 35), (w, h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.75, img, 0.25, 0, img)

        # Progress info
        filename = os.path.basename(img_path)
        progress_str = f"[{self.idx + 1}/{len(self.image_paths)}] {filename}"
        cv2.putText(img, progress_str, (15, 26), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)

        # Active class badge
        cur_cls = CLASSES[self.active_cls]
        cls_str = f"KELAS: [{self.active_cls}: {cur_cls['name'].upper()}] | MODEL: {self.backend.upper()}"
        cv2.putText(img, cls_str, (15, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.52, cur_cls['color'], 2)

        # Box counter / Status
        if self.is_background:
            status_str = "STATUS: [NEGATIVE / BACKGROUND (0 byte)]"
            status_color = (0, 220, 220)
        else:
            status_str = f"STATUS: {len(self.current_boxes)} Bounding Box"
            status_color = (0, 255, 120)
        cv2.putText(img, status_str, (500, 40), cv2.FONT_HERSHEY_DUPLEX, 0.6, status_color, 2)

        # Bottom help
        help_txt = "[Drag] Box  |  [R-Click] Hapus  |  [0/1] Kelas  |  [B] Background  |  [SPACE] Simpan & Next  |  [S] Export/Split"
        cv2.putText(img, help_txt, (15, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    def export_yolo_dataset(self, split_ratio=0.8, dest_dir="dataset_yolo"):
        """
        Membagi dataset menjadi 80% train dan 20% validation sesuai Fase 3.2 STEPS.md
        dan menghasilkan file data.yaml otomatis.
        """
        print("\n" + "=" * 60)
        print(f"[Export YOLO] Memproses pembagian dataset (Train: {int(split_ratio*100)}%, Val: {int((1-split_ratio)*100)}%)...")

        train_img_dir = os.path.join(dest_dir, "images", "train")
        val_img_dir = os.path.join(dest_dir, "images", "val")
        train_lbl_dir = os.path.join(dest_dir, "labels", "train")
        val_lbl_dir = os.path.join(dest_dir, "labels", "val")

        for d in (train_img_dir, val_img_dir, train_lbl_dir, val_lbl_dir):
            os.makedirs(d, exist_ok=True)

        all_labels = glob.glob(os.path.join(self.lbl_dir, "*.txt"))
        if not all_labels:
            print("[Export YOLO] Belum ada label yang tersimpan di", self.lbl_dir)
            return

        pairs = []
        for lbl_path in all_labels:
            base = os.path.splitext(os.path.basename(lbl_path))[0]
            img_candidate = os.path.join(self.img_dir, base + ".jpg")
            if os.path.exists(img_candidate):
                pairs.append((img_candidate, lbl_path))

        random.seed(42)
        random.shuffle(pairs)

        split_idx = int(len(pairs) * split_ratio)
        train_pairs = pairs[:split_idx]
        val_pairs = pairs[split_idx:]

        for img_p, lbl_p in train_pairs:
            shutil.copy(img_p, train_img_dir)
            shutil.copy(lbl_p, train_lbl_dir)

        for img_p, lbl_p in val_pairs:
            shutil.copy(img_p, val_img_dir)
            shutil.copy(lbl_p, val_lbl_dir)

        yaml_path = os.path.join(dest_dir, "data.yaml")
        abs_dest = os.path.abspath(dest_dir).replace("\\", "/")
        yaml_content = f"""# Deezpro Fire & Smoke Dataset Configuration
# Generated automatically by annotate_dataset.py

path: {abs_dest}
train: images/train
val: images/val

names:
  0: smoke
  1: fire
"""
        with open(yaml_path, "w") as f:
            f.write(yaml_content)

        print(f"[Export YOLO] Selesai!")
        print(f"  - Total Data : {len(pairs)} pasangan gambar & label")
        print(f"  - Training   : {len(train_pairs)} (images/train & labels/train)")
        print(f"  - Validation : {len(val_pairs)} (images/val & labels/val)")
        print(f"  - Konfigurasi: {yaml_path}")
        print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Tool Anotasi Dataset YOLO dengan Zero-Shot Pre-labeling (STEPS.md)")
    parser.add_argument("--raw", nargs="+", default=["dataset_raw"],
                        help="Folder berisi citra mentah (bisa lebih dari satu, contoh: --raw folder1 folder2)")
    parser.add_argument("--out", default="dataset_annotated", help="Folder output citra & label (default: dataset_annotated)")
    parser.add_argument("--backend", default="grounding_dino", choices=["grounding_dino", "florence2", "yolo", "none"],
                        help="Backend pre-labeler (default: grounding_dino)")
    parser.add_argument("--model", default=None, help="Nama repo model HuggingFace atau path checkpoint")
    parser.add_argument("--conf", type=float, default=0.30, help="Confidence threshold pre-labeling (default: 0.30)")
    parser.add_argument("--split-only", action="store_true", help="Hanya eksekusi split dataset tanpa membuka GUI")
    args = parser.parse_args()

    annotator = DatasetAnnotator(
        raw_dir=args.raw,
        output_dir=args.out,
        backend=args.backend,
        model_id=args.model,
        conf_thresh=args.conf
    )

    if args.split_only:
        annotator.export_yolo_dataset()
    else:
        annotator.run()


if __name__ == "__main__":
    main()
