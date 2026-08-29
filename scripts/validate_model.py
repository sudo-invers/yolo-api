# scripts/validate_model.py

import sys

from ultralytics import YOLO

MAP_THRESHOLD = 0.55  # mAP@0.5 minimo aceitavel
DATASET_YAML = "datasets/validation.yaml"
MODEL_PATH = "models/yolov8n.pt"


def main():
    model = YOLO(MODEL_PATH)

    metrics = model.val(
        data=DATASET_YAML,
        split="val",
        verbose=False,
    )

    map50 = metrics.box.map50

    print(f"mAP@0.5 = {map50:.4f} (limiar: {MAP_THRESHOLD})")

    if map50 < MAP_THRESHOLD:
        print(
            "[FALHA] Modelo abaixo do limiar de qualidade. "
            "Deploy bloqueado."
        )
        sys.exit(1)

    print("[OK] Quality gate aprovado.")


if __name__ == "__main__":
    main()
