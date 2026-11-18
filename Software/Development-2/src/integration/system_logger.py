"""
=====================================================================
DEEZPRO INTEGRATION - SYSTEM LOGGER
=====================================================================
File: system_logger.py
Deskripsi: Modul konfigurasi logging standar yang mencatat aktivitas,
            peringatan, dan pesan galat sistem secara terpadu ke
            terminal konsol dan berkas teks logs/system.log.
"""

import os
import sys
import logging
from logging.handlers import RotatingFileHandler

_logger_initialized = False

def setup_logger(log_dir: str = None, log_filename: str = "system.log", level: int = logging.INFO) -> logging.Logger:
    """
    Inisialisasi sistem logging ganda (Konsol + File Teks Rotating).
    """
    global _logger_initialized
    logger = logging.getLogger("Deezpro")

    if _logger_initialized:
        return logger

    logger.setLevel(level)
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)-7s] [%(name)s.%(module)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # 1. Console Handler (Output ke terminal)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 2. File Handler (Output ke logs/system.log)
    if log_dir is None:
        # Default ke folder logs relatif terhadap skrip
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        log_dir = os.path.join(base_dir, "logs")

    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, log_filename)

    file_handler = RotatingFileHandler(
        log_file_path,
        maxBytes=5 * 1024 * 1024,  # Maksimal 5 MB per file
        backupCount=5,              # Simpan hingga 5 file arsip
        encoding="utf-8"
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    _logger_initialized = True
    logger.info(f"System logger aktif. Berkas log tersimpan pada: {log_file_path}")
    return logger

def get_logger() -> logging.Logger:
    """Mengambil instance logger utama Deezpro."""
    global _logger_initialized
    if not _logger_initialized:
        return setup_logger()
    return logging.getLogger("Deezpro")
