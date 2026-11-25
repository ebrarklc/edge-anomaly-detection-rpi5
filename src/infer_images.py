"""
PatchCore (OpenVINO IR) ile bir klasördeki görüntülerde anomali tespiti.

Raspberry Pi 5 üzerinde "dosya modu" için yazılmıştır:
  - Her görüntü için anomali skoru, tahmin (sağlam/hatalı) ve çıkarım süresi ölçülür.
  - Klasör yapısı MVTec gibiyse (test/good, test/<hata_türü>) doğruluk da hesaplanır.
  - Her hata türünden birkaç görüntü için ısı haritası (heatmap) kaydedilir.

Kullanım:
  python src/infer_images.py \
      --model models/bottle/model.xml \
      --data  datasets/MVTecAD/bottle/test \
      --out   outputs/bottle
"""

import argparse
import statistics
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import openvino as ov

IMG_SIZE = 256                      # modelin eğitildiği görüntü boyutu
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


def load_model(xml_path: str, threads: int | None = None):
    """OpenVINO modelini okuyup CPU için derler."""
    core = ov.Core()
    config = {"PERFORMANCE_HINT": "LATENCY"}
    if threads:
        config["INFERENCE_NUM_THREADS"] = threads
    return core.compile_model(core.read_model(xml_path), "CPU", config)


def preprocess(bgr: np.ndarray) -> np.ndarray:
    """BGR (OpenCV) görüntüyü modelin beklediği tensöre çevirir: [1, 3, 256, 256], RGB, 0-1."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    x = rgb.astype(np.float32) / 255.0
    return x.transpose(2, 0, 1)[None]          # HWC -> CHW, batch boyutu ekle


def make_overlay(bgr: np.ndarray, anomaly_map: np.ndarray, pred_mask: np.ndarray,
                 score: float, is_anomaly: bool) -> np.ndarray:
    """Orijinal görüntü | ısı haritası + tahmin edilen hata sınırı."""
    img = cv2.resize(bgr, (IMG_SIZE, IMG_SIZE))
    heat = cv2.applyColorMap((np.clip(anomaly_map, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_JET)
    blended = cv2.addWeighted(img, 0.55, heat, 0.45, 0)
    contours, _ = cv2.findContours(pred_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(blended, contours, -1, (0, 0, 255), 2)

    canvas = np.hstack([img, blended])
    color = (0, 0, 255) if is_anomaly else (0, 170, 0)
    text = f"{'ANOMALI' if is_anomaly else 'SAGLAM'}  skor={score:.2f}"
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 28), (0, 0, 0), -1)
    cv2.putText(canvas, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return canvas


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="model.xml yolu")
    ap.add_argument("--data", required=True, help="görüntü klasörü (alt klasörler dahil taranır)")
    ap.add_argument("--out", default="outputs", help="ısı haritalarının kaydedileceği klasör")
    ap.add_argument("--save-per-type", type=int, default=2, help="her hata türünden kaç görsel kaydedilsin")
    ap.add_argument("--threshold", type=float, default=0.5, help="eşik (model skoru 0-1 arası, varsayılan 0.5)")
    ap.add_argument("--threads", type=int, default=None, help="CPU iş parçacığı sayısı (boş = otomatik)")
    args = ap.parse_args()

    files = sorted(p for p in Path(args.data).rglob("*") if p.suffix.lower() in IMG_EXTS)
    if not files:
        raise SystemExit(f"'{args.data}' içinde görüntü bulunamadı.")

    t0 = time.perf_counter()
    model = load_model(args.model, args.threads)
    print(f"Model yüklendi: {args.model}  ({time.perf_counter() - t0:.1f} sn)")

    # Isınma (warm-up): ilk çalıştırmalar bellek ayırma yüzünden yavaştır, ölçüme katmıyoruz
    dummy = np.zeros((1, 3, IMG_SIZE, IMG_SIZE), np.float32)
    for _ in range(3):
        model(dummy)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    latencies_ms, saved = [], defaultdict(int)
    per_type = defaultdict(lambda: [0, 0])     # tür -> [toplam, "anomali" denen]
    tp = fp = tn = fn = 0

    print(f"\n{len(files)} görüntü işleniyor...\n")
    for path in files:
        bgr = cv2.imread(str(path))
        x = preprocess(bgr)

        t = time.perf_counter()
        r = model(x)
        latencies_ms.append((time.perf_counter() - t) * 1000)

        score = float(r["pred_score"][0])
        is_anomaly = score > args.threshold
        defect_type = path.parent.name
        per_type[defect_type][0] += 1
        per_type[defect_type][1] += int(is_anomaly)

        truth = defect_type != "good"           # MVTec yapısı: 'good' dışındaki her klasör hatalı
        tp += truth and is_anomaly
        fn += truth and not is_anomaly
        fp += (not truth) and is_anomaly
        tn += (not truth) and not is_anomaly

        if saved[defect_type] < args.save_per_type:
            overlay = make_overlay(bgr, r["anomaly_map"][0, 0], r["pred_mask"][0, 0], score, is_anomaly)
            cv2.imwrite(str(out_dir / f"{defect_type}_{path.stem}.png"), overlay)
            saved[defect_type] += 1

    # --- Rapor ---
    print(f"{'Tür':<20} {'Adet':>5} {'Anomali denen':>14}")
    print("-" * 41)
    for t_name, (n, n_anom) in sorted(per_type.items()):
        print(f"{t_name:<20} {n:>5} {n_anom:>10} ({n_anom / n:.0%})")

    total = tp + tn + fp + fn
    print("\nKarışıklık matrisi:  TP={}  FN={}  FP={}  TN={}".format(tp, fn, fp, tn))
    print(f"Doğruluk : {(tp + tn) / total:.3f}")
    if tp + fp:
        print(f"Precision: {tp / (tp + fp):.3f}")
    if tp + fn:
        print(f"Recall   : {tp / (tp + fn):.3f}")

    mean_ms = statistics.mean(latencies_ms)
    print(f"\n⏱️  Çıkarım süresi (görüntü başına): ortalama {mean_ms:.0f} ms | "
          f"medyan {statistics.median(latencies_ms):.0f} ms | "
          f"min {min(latencies_ms):.0f} ms | max {max(latencies_ms):.0f} ms")
    print(f"⚡ Yaklaşık hız: {1000 / mean_ms:.2f} FPS")
    print(f"🖼️  Isı haritaları: {out_dir}/")


if __name__ == "__main__":
    main()
