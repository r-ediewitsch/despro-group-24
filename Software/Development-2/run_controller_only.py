"""
=====================================================================
DEEZPRO - RUN CONTROLLER ONLY (INTERACTIVE TESTING CLI)
=====================================================================
File: run_controller_only.py
Deskripsi: Alat uji interaktif konsol untuk menguji seluruh subsistem
            kontrol antara Python Coordinator dengan 3 modul ESP32
            secara mandiri tanpa memerlukan perangkat keras kamera.
"""

import os
import sys
import yaml
import time

# Pastikan path modul lokal dikenali
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from src.integration.coordinator import CentralCoordinator, SystemState
from src.integration.system_logger import setup_logger, get_logger

logger = setup_logger()


def load_configuration(config_path: str = None) -> dict:
    """Memuat file konfigurasi system_config.yaml."""
    if config_path is None:
        config_path = os.path.join(current_dir, "config", "system_config.yaml")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"File konfigurasi tidak ditemukan: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def print_menu():
    print("\n" + "=" * 70)
    print(" DEEZPRO CONTROLLER TEST CONSOLE (PYTHON <-> 3x ESP32)")
    print("=" * 70)
    print(" Perintah Operasi Berbasis Koordinat:")
    print("   FIRE <x_cm> <y_cm>   : Simulasikan deteksi api (Gerak -> Barrier -> Semprot -> Selesai)")
    print("   GOTO <x_cm> <y_cm>   : Gerak manual ke koordinat fisik tanpa semprot APAR")
    print("   HOME                 : Jalankan Auto-Homing pada seluruh sumbu (Limit Switch)")
    print("\n Perintah Manual & Uji Komponen:")
    print("   JOG X <steps>        : Gerak relatif Sumbu X (misal: 'JOG X 500' atau 'JOG X -200')")
    print("   JOG Y <steps>        : Gerak relatif Sumbu Y (misal: 'JOG Y 500' atau 'JOG Y -200')")
    print("   SPRAY [duration_ms]  : Uji aktivasi servo APAR (default: 3000 ms)")
    print("   STATUS               : Tampilkan telemetri koordinat dan status 3x ESP32")
    print("   ESTOP                : Pemicu Penghentian Darurat Instan (Emergency Stop)")
    print("   RESET                : Reset status dari E-STOP / FAULT kembali ke IDLE")
    print("   HELP / ?             : Tampilkan kembali menu ini")
    print("   QUIT / EXIT          : Keluar dari program")
    print("=" * 70 + "\n")


def on_status_received(node: str, data: dict):
    """Menampilkan pembaruan status real-time dari ESP32."""
    state = data.get("state", "UNKNOWN")
    pos = data.get("pos", "")
    target = data.get("target", "")
    seq = data.get("seq", "")
    msg = data.get("msg", "")

    detail = []
    if pos != "": detail.append(f"pos:{pos}")
    if target != "": detail.append(f"target:{target}")
    if seq: detail.append(f"seq:{seq}")
    if msg: detail.append(f"msg:'{msg}'")

    details_str = " | ".join(detail)
    print(f"\n[FEEDBACK {node.upper()}] Status: {state} ({details_str})\nCommand > ", end="", flush=True)


def main():
    print("\n[INISIALISASI] Memuat konfigurasi dari config/system_config.yaml ...")
    try:
        config = load_configuration()
    except Exception as e:
        print(f"[FATAL] Gagal memuat konfigurasi: {e}")
        return

    coordinator = CentralCoordinator(config)
    coordinator.mqtt.register_status_listener(on_status_received)

    print("[KONEKSI] Menghubungkan ke MQTT Broker...")
    started = coordinator.start(auto_home=False)
    if not started:
        print("[WARNING] Tidak dapat terhubung ke MQTT Broker secara langsung. Membuka konsol offline...")

    print_menu()

    try:
        while True:
            try:
                user_input = input("Command > ").strip()
            except (EOFError, KeyboardInterrupt):
                break

            if not user_input:
                continue

            parts = user_input.split()
            cmd = parts[0].upper()

            if cmd in ["QUIT", "EXIT", "Q"]:
                print("\n[INFO] Menutup koneksi dan keluar dari controller...")
                break

            elif cmd in ["HELP", "?"]:
                print_menu()

            elif cmd == "FIRE":
                if len(parts) < 3:
                    print("[ERROR] Gunakan format: FIRE <x_cm> <y_cm> (Contoh: FIRE 100 120)")
                    continue
                try:
                    x = float(parts[1])
                    y = float(parts[2])
                    print(f"\n[SIMULASI API] Mengirim target koordinat api ({x}, {y}) cm ke sistem...")
                    success = coordinator.dispatch_target(x, y)
                    print(f"[HASIL] Siklus eksekusi target selesai. Status sukses: {success}")
                except ValueError:
                    print("[ERROR] Nilai koordinat harus berupa angka desimal/integer!")

            elif cmd == "GOTO":
                if len(parts) < 3:
                    print("[ERROR] Gunakan format: GOTO <x_cm> <y_cm> (Contoh: GOTO 150 80)")
                    continue
                try:
                    x = float(parts[1])
                    y = float(parts[2])
                    print(f"\n[MANUAL GOTO] Menggerakkan ke ({x}, {y}) cm...")
                    coordinator.manual_goto_cm(x, y)
                except ValueError:
                    print("[ERROR] Nilai koordinat harus berupa angka!")

            elif cmd == "HOME":
                print("\n[MANUAL HOMING] Memulai prosedur auto-homing pada kedua sumbu...")
                coordinator.execute_homing_all(timeout_sec=40.0)

            elif cmd == "JOG":
                if len(parts) < 3:
                    print("[ERROR] Gunakan format: JOG <X|Y> <steps> (Contoh: JOG X 500)")
                    continue
                axis = parts[1].lower()
                if axis not in ["x", "y"]:
                    print("[ERROR] Sumbu harus 'X' atau 'Y'!")
                    continue
                try:
                    steps = int(parts[2])
                    print(f"\n[JOG] Menggerakkan Sumbu {axis.upper()} sejauh {steps} langkah...")
                    coordinator.manual_jog_steps(axis, steps)
                except ValueError:
                    print("[ERROR] Jumlah langkah harus berupa angka bulat integer!")

            elif cmd == "SPRAY":
                dur = 3000
                if len(parts) >= 2:
                    try:
                        dur = int(parts[1])
                    except ValueError:
                        pass
                print(f"\n[UJI SPRAY] Menyemprotkan APAR selama {dur} ms...")
                coordinator.manual_spray_test(duration_ms=dur)

            elif cmd == "STATUS":
                telemetry = coordinator.get_system_telemetry()
                print("\n" + "-" * 55)
                print(" STATUS TELEMETRI SISTEM SAAT INI")
                print("-" * 55)
                print(f" State Sistem   : {telemetry['state']}")
                print(f" MQTT Connected : {'YA' if telemetry['mqtt_connected'] else 'TIDAK'}")
                print(f" Posisi Fisik   : X = {telemetry['current_position_cm']['x']} cm, Y = {telemetry['current_position_cm']['y']} cm")
                print(f" Posisi Langkah : X = {telemetry['current_position_steps']['x']} steps, Y = {telemetry['current_position_steps']['y']} steps")
                print("\n Status Hardware Nodes:")
                for n, info in telemetry['nodes'].items():
                    last_seen = info.get('last_seen', 0)
                    ago = f"{time.time() - last_seen:.1f}s lalu" if last_seen > 0 else "Belum pernah"
                    pos_str = f"| Pos: {info.get('pos', '-')}" if 'pos' in info else ""
                    print(f"   [{n.upper()}] State: {info.get('state', 'UNKNOWN'):<10} {pos_str:<16} | Terakhir: {ago}")
                print("-" * 55)

            elif cmd == "ESTOP":
                print("\n[EMERGENCY STOP] Mematikan pergerakan seluruh motor dan melepas APAR!")
                coordinator.trigger_emergency_stop()

            elif cmd == "RESET":
                print("\n[RESET] Mereset status sistem kembali ke IDLE_MONITORING...")
                coordinator.safety.reset_emergency_stop()
                coordinator.set_state(SystemState.IDLE_MONITORING)

            else:
                print(f"[ERROR] Perintah tidak dikenali: '{user_input}'. Ketik HELP untuk bantuan.")

    finally:
        coordinator.stop()
        print("[INFO] Program selesai.")


if __name__ == "__main__":
    main()
