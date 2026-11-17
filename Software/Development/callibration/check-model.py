"""
Skrip Pemeriksa Metadata & Spesifikasi Model YOLO (.pt)
Menganalisis arsitektur, dataset pelatihan, dan daftar kelas objek target.
"""
import os
import sys
import argparse
import zipfile
import pickle


def resolve_model_path(path_str):
    """
    Menyelesaikan path relatif baik dari CWD maupun dari folder skrip ini.
    """
    if os.path.isabs(path_str):
        return path_str
    if os.path.exists(path_str):
        return os.path.abspath(path_str)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidate = os.path.join(script_dir, path_str)
    if os.path.exists(candidate):
        return candidate
    return path_str


def inspect_model_metadata(model_path):
    model_path = resolve_model_path(model_path)

    # Cek jika path berupa folder terekstrak
    if not os.path.isfile(model_path) and os.path.isdir(model_path):
        packed = model_path.rstrip("/\\") + "_packed.pt"
        if not os.path.exists(packed):
            print(f"[INFO] Path berupa direktori, mengemas otomatis menjadi: {packed} ...")
            with zipfile.ZipFile(packed, "w", compression=zipfile.ZIP_STORED) as zf:
                for root, dirs, files in os.walk(model_path):
                    for f in files:
                        fp = os.path.join(root, f)
                        rp = os.path.relpath(fp, model_path)
                        zf.write(fp, rp)
        model_path = packed

    if not os.path.exists(model_path):
        print(f"\n[ERROR] File model tidak ditemukan pada: {model_path}")
        sys.exit(1)

    file_size_mb = os.path.getsize(model_path) / (1024 * 1024)

    # 1. Coba baca via Ultralytics jika terpasang
    data = None
    names = None
    train_args = {}

    try:
        from ultralytics import YOLO
        yolo_obj = YOLO(model_path)
        names = yolo_obj.names
        if hasattr(yolo_obj, "ckpt") and isinstance(yolo_obj.ckpt, dict):
            data = yolo_obj.ckpt
            train_args = data.get("train_args", {})
    except Exception:
        pass

    # 2. Fallback ke torch.load jika Ultralytics tidak mengembalikan ckpt dict
    if data is None:
        try:
            import torch
            try:
                data = torch.load(model_path, map_location="cpu", weights_only=False)
            except TypeError:
                data = torch.load(model_path, map_location="cpu")
            if isinstance(data, dict):
                train_args = data.get("train_args", {})
                if names is None and "model" in data and hasattr(data["model"], "names"):
                    names = data["model"].names
        except Exception:
            pass

    # 3. Fallback membaca data.pkl dari arsip zip langsung
    if data is None and zipfile.is_zipfile(model_path):
        try:
            with zipfile.ZipFile(model_path, "r") as zf:
                pkl_name = [n for n in zf.namelist() if n.endswith("data.pkl")][0]
                with zf.open(pkl_name) as f:
                    class DummyUnpickler(pickle.Unpickler):
                        def find_class(self, module, name):
                            try:
                                return super().find_class(module, name)
                            except Exception:
                                class Dummy:
                                    def __init__(self, *args, **kwargs): pass
                                    def __setstate__(self, state): self.__dict__.update(state if isinstance(state, dict) else {})
                                Dummy.__name__ = name
                                Dummy.__module__ = module
                                return Dummy
                        def persistent_load(self, pid):
                            return ("tensor", pid)
                    data = DummyUnpickler(f).load()
                    if isinstance(data, dict):
                        train_args = data.get("train_args", {})
                        if names is None and "model" in data and hasattr(data["model"], "names"):
                            names = data["model"].names
        except Exception:
            pass

    print("\n" + "=" * 65)
    print(" HASIL PEMERIKSAAN SPESIFIKASI MODEL YOLO")
    print("=" * 65)
    print(f" File Path     : {model_path}")
    print(f" Ukuran File   : {file_size_mb:.2f} MB")

    if isinstance(data, dict):
        print(f" Tanggal Train : {data.get('date', 'Tidak tercatat')}")
        print(f" Versi Library : Ultralytics {data.get('version', 'N/A')}")
        print(f" Task Model    : {train_args.get('task', data.get('task', 'detect'))}")
        print(f" Dataset Asal  : {train_args.get('data', 'Tidak tercatat')}")
        print(f" Project/Run   : {train_args.get('project', '-')}/{train_args.get('name', '-')}")
        print(f" Input Image   : {train_args.get('imgsz', 640)}x{train_args.get('imgsz', 640)}")
        print(f" Epoch Selesai : {data.get('epoch', 'N/A')}")

    print("-" * 65)

    if names:
        total_cls = len(names)
        print(f" Total Kelas   : {total_cls} Kelas")
        print(" Daftar Label  :")
        fire_classes = []
        for cid, cname in sorted(names.items()):
            cname_str = str(cname)
            is_fire_label = any(kw in cname_str.lower() for kw in ("fire", "flame", "smoke", "api")) and "hydrant" not in cname_str.lower()
            if is_fire_label:
                fire_classes.append((cid, cname_str))
                marker = " [TARGET API/ASAP]"
            else:
                marker = ""
            print(f"   [{cid:>2}] : {cname_str:<20}{marker}")

        print("-" * 65)
        print(" KESIMPULAN & ANALISIS:")
        if fire_classes:
            print(f" [OK] Model ini MEMILIKI kelas khusus api: {fire_classes}")
            print("      Model siap digunakan untuk mendeteksi api langsung dengan YOLO.")
        elif total_cls == 80 and names.get(0) == "person":
            print(" [PERHATIAN] Model ini adalah bobot dasar Microsoft COCO (80 objek umum).")
            print("             Kelas 0 adalah 'person' (manusia), BUKAN api.")
            print("             Model ini belum pernah dilatih untuk api/asap.")
            print("             Gunakan mode Computer Vision (CV) pada detect_cam1.py")
            print("             atau lakukan fine-tuning bobot dengan dataset api (D-Fire).")
        else:
            print(f" [INFO] Model memiliki {total_cls} kelas, namun tidak ada label 'fire'/'api'.")
    else:
        print(" [WARNING] Gagal membaca dictionary label kelas dari checkpoint.")

    print("=" * 65 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pemeriksa metadata model YOLO (.pt)")
    parser.add_argument(
        "model",
        nargs="?",
        default="yolo26n.pt",
        help="Path file model yang ingin diperiksa (default: yolo26n.pt)",
    )
    args = parser.parse_args()
    inspect_model_metadata(args.model)
