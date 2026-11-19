"""
foundation_labeler.py
---------------------
Engine Pre-Labeling Berbasis Zero-Shot Foundation Model (Fase 2.1 STEPS.md):
- Mendukung Grounding DINO (IDEA-Research/grounding-dino-tiny / base)
- Mendukung Florence-2 (microsoft/Florence-2-base / large)
- Mendukung Fallback ke model YOLO (best.pt)

Model visi-bahasa ini mampu mendeteksi api & asap secara zero-shot dengan prompt teks
("fire", "flame", "smoke"), mengatasi keterbatasan dataset publik yang overfitting
dan mengurangi beban anotasi manual secara drastis.
"""

import os
import sys
from typing import List, Dict, Any, Optional
import numpy as np
import cv2

# Safe cache dir for transformers and torch
os.environ.setdefault("HF_HOME", os.path.join(os.path.expanduser("~"), ".cache", "huggingface"))


class FoundationLabeler:
    """
    Wrapper serbaguna untuk pre-labeling zero-shot menggunakan
    Grounding DINO atau Microsoft Florence-2.
    """

    def __init__(self, backend: str = "grounding_dino",
                 model_id: Optional[str] = None,
                 device: Optional[str] = None):
        """
        Args:
            backend: 'grounding_dino', 'florence2', atau 'yolo'
            model_id: Nama model HuggingFace atau path checkpoint
            device: 'cuda', 'cpu', atau None (auto-detect)
        """
        self.backend = backend.lower()
        self.model_id = model_id
        self.device = device
        self.model = None
        self.processor = None

        self._init_backend()

    def _get_device(self):
        if self.device is not None:
            return self.device
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    def _init_backend(self):
        dev = self._get_device()

        if self.backend == "grounding_dino":
            self._init_grounding_dino(dev)
        elif self.backend in ("florence2", "florence-2"):
            self._init_florence2(dev)
        elif self.backend == "yolo":
            self._init_yolo()
        else:
            raise ValueError(f"Backend tidak dikenali: {self.backend}. Gunakan 'grounding_dino', 'florence2', atau 'yolo'.")

    def _init_grounding_dino(self, dev: str):
        try:
            import torch
            from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        except ImportError as e:
            raise ImportError(
                "Pustaka 'transformers' atau 'torch' belum terpasang. "
                "Jalankan: pip install torch transformers"
            ) from e

        target_model = self.model_id or "IDEA-Research/grounding-dino-tiny"
        print(f"[FoundationLabeler] Mengunduh/Memuat Grounding DINO ({target_model}) pada device '{dev}'...")
        self.processor = AutoProcessor.from_pretrained(target_model)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(target_model).to(dev)
        self.model.eval()
        self.device = dev
        print("[FoundationLabeler] Grounding DINO siap digunakan.")

    def _init_florence2(self, dev: str):
        try:
            import torch
            from transformers import AutoProcessor, AutoModelForCausalLM
        except ImportError as e:
            raise ImportError(
                "Pustaka 'transformers' atau 'torch' belum terpasang. "
                "Jalankan: pip install torch transformers timm einops"
            ) from e

        target_model = self.model_id or "microsoft/Florence-2-base"
        print(f"[FoundationLabeler] Mengunduh/Memuat Florence-2 ({target_model}) pada device '{dev}'...")
        torch_dtype = torch.float16 if dev == "cuda" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            target_model,
            torch_dtype=torch_dtype,
            trust_remote_code=True
        ).to(dev)
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(target_model, trust_remote_code=True)
        self.device = dev
        print("[FoundationLabeler] Florence-2 siap digunakan.")

    def _init_yolo(self):
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise ImportError("Pustaka 'ultralytics' belum terpasang. Jalankan: pip install ultralytics") from e

        target_model = self.model_id or "models/best.pt"
        if not os.path.exists(target_model):
            raise FileNotFoundError(f"Checkpoint YOLO tidak ditemukan: {target_model}")
        print(f"[FoundationLabeler] Memuat model YOLO: {target_model}")
        self.model = YOLO(target_model)
        print("[FoundationLabeler] YOLO siap digunakan.")

    @staticmethod
    def _map_label_to_class(label_text: str) -> int:
        """
        Memetakan teks deteksi ke kelas YOLO Deezpro:
        - 0: smoke
        - 1: fire
        """
        lt = label_text.strip().lower()
        if any(w in lt for w in ("smoke", "asap", "vapor", "haze")):
            return 0
        if any(w in lt for w in ("fire", "flame", "blaze", "api")):
            return 1
        return 1  # Default ke fire jika ragu

    def predict(self, image_input, box_threshold: float = 0.30,
                text_threshold: float = 0.25) -> List[Dict[str, Any]]:
        """
        Menjalankan zero-shot detection pada citra.

        Args:
            image_input: Path file string (str) atau array numpy BGR (cv2)
            box_threshold: Ambang keyakinan bounding box (0.0 - 1.0)
            text_threshold: Ambang kesesuaian teks untuk Grounding DINO

        Returns:
            List of dict:
                {
                    "class_id": int (0 atau 1),
                    "class_name": str ("smoke" atau "fire"),
                    "box": [x1, y1, x2, y2], # koordinat piksel absolut
                    "confidence": float
                }
        """
        # Load image via OpenCV
        if isinstance(image_input, str):
            bgr = cv2.imread(image_input)
            if bgr is None:
                raise ValueError(f"Tidak dapat membaca file gambar: {image_input}")
        else:
            bgr = image_input

        h, w = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        if self.backend == "grounding_dino":
            return self._predict_grounding_dino(rgb, w, h, box_threshold, text_threshold)
        elif self.backend in ("florence2", "florence-2"):
            return self._predict_florence2(rgb, w, h)
        elif self.backend == "yolo":
            return self._predict_yolo(bgr, box_threshold)
        return []

    def _predict_grounding_dino(self, rgb_img, w: int, h: int,
                                box_thresh: float, text_thresh: float) -> List[Dict[str, Any]]:
        import torch
        from PIL import Image

        pil_img = Image.fromarray(rgb_img)
        # Grounding DINO prompt dengan separator titik (.)
        prompt = "fire. flame. smoke."

        inputs = self.processor(images=pil_img, text=prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=box_thresh,
            text_threshold=text_thresh,
            target_sizes=[(h, w)]
        )[0]

        detections = []
        boxes = results["boxes"].cpu().numpy()
        scores = results["scores"].cpu().numpy()
        labels = results["labels"]

        for box, score, label in zip(boxes, scores, labels):
            x1, y1, x2, y2 = map(int, box)
            x1 = max(0, min(w - 1, x1))
            y1 = max(0, min(h - 1, y1))
            x2 = max(0, min(w - 1, x2))
            y2 = max(0, min(h - 1, y2))

            cls_id = self._map_label_to_class(label)
            cls_name = "smoke" if cls_id == 0 else "fire"

            detections.append({
                "class_id": cls_id,
                "class_name": cls_name,
                "box": [x1, y1, x2, y2],
                "confidence": float(score),
                "raw_label": label
            })

        return detections

    def _predict_florence2(self, rgb_img, w: int, h: int) -> List[Dict[str, Any]]:
        import torch
        from PIL import Image

        pil_img = Image.fromarray(rgb_img)
        # Task Florence-2: Open Vocabulary Detection / Phrase Grounding
        task_prompt = "<CAPTION_TO_PHRASE_GROUNDING>"
        text_prompt = "fire, flame, smoke"
        full_prompt = task_prompt + " " + text_prompt

        inputs = self.processor(text=full_prompt, images=pil_img, return_tensors="pt").to(self.device)
        if self.device == "cuda":
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16)

        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                num_beams=3
            )

        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed_answer = self.processor.post_process_generation(
            generated_text,
            task=task_prompt,
            image_size=(w, h)
        )

        detections = []
        result_dict = parsed_answer.get(task_prompt, {})
        bboxes = result_dict.get("bboxes", [])
        labels = result_dict.get("labels", [])

        for bbox, label in zip(bboxes, labels):
            x1, y1, x2, y2 = map(int, bbox)
            x1 = max(0, min(w - 1, x1))
            y1 = max(0, min(h - 1, y1))
            x2 = max(0, min(w - 1, x2))
            y2 = max(0, min(h - 1, y2))

            cls_id = self._map_label_to_class(label)
            cls_name = "smoke" if cls_id == 0 else "fire"

            detections.append({
                "class_id": cls_id,
                "class_name": cls_name,
                "box": [x1, y1, x2, y2],
                "confidence": 0.90,  # Florence-2 output doesn't supply raw logits in grounded dict
                "raw_label": label
            })

        return detections

    def _predict_yolo(self, bgr_img, conf_thresh: float) -> List[Dict[str, Any]]:
        results = self.model.predict(bgr_img, conf=conf_thresh, verbose=False)
        detections = []
        if results and len(results) > 0:
            boxes = results[0].boxes
            for box in boxes:
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                xyxy = box.xyxy[0].tolist()
                cls_name = "smoke" if cls_id == 0 else "fire"
                detections.append({
                    "class_id": cls_id,
                    "class_name": cls_name,
                    "box": [int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])],
                    "confidence": conf,
                    "raw_label": cls_name
                })
        return detections
