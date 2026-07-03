"""
Etap 0: Proof of concept — process a recorded video file.

Runs YOLO11 on sampled frames, detects persons and vehicles,
flags dangerous proximity, outputs annotated video.

Usage:
    python etap0/detect_video.py --input video.mp4 --output annotated.mp4
    python etap0/detect_video.py --input video.mp4 --show
"""
import argparse
import sys
import os
import time
from collections import defaultdict

import cv2
import numpy as np
from ultralytics import YOLO

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CONFIG

PERSON_CLASSES = [0]
VEHICLE_CLASSES = [2, 5, 7]  # car, bus, truck
ALL_CLASSES = PERSON_CLASSES + VEHICLE_CLASSES

COLOR_PERSON = (0, 255, 0)       # green
COLOR_VEHICLE = (255, 136, 0)    # blue-ish (BGR)
COLOR_DANGER = (0, 0, 255)       # red
COLOR_WARNING = (0, 165, 255)    # orange


def bbox_iou(a, b):
    xi1 = max(a[0], b[0])
    yi1 = max(a[1], b[1])
    xi2 = min(a[2], b[2])
    yi2 = min(a[3], b[3])
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def bbox_min_distance(a, b):
    dx = max(0, max(a[0] - b[2], b[0] - a[2]))
    dy = max(0, max(a[1] - b[3], b[1] - a[3]))
    return (dx**2 + dy**2) ** 0.5


def box_center(box):
    return ((box[0] + box[2]) // 2, (box[1] + box[3]) // 2)


class TemporalFilter:
    def __init__(self, required=3, cooldown_sec=10.0):
        self.required = required
        self.cooldown_sec = cooldown_sec
        self._streak = defaultdict(int)
        self._last_alert = {}

    @staticmethod
    def _pair_key(person_box, vehicle_box):
        pc = ((person_box[0] + person_box[2]) // 64,
              (person_box[1] + person_box[3]) // 64)
        vc = ((vehicle_box[0] + vehicle_box[2]) // 64,
              (vehicle_box[1] + vehicle_box[3]) // 64)
        return (pc, vc)

    def update(self, danger_pairs, now):
        """
        danger_pairs: list of (person_det, vehicle_det, iou, dist, severity)
        Returns list of confirmed pairs.
        """
        current_keys = set()
        confirmed = []
        for pair in danger_pairs:
            p_det, v_det = pair[0], pair[1]
            key = self._pair_key(p_det["box"], v_det["box"])
            current_keys.add(key)
            self._streak[key] += 1
            if self._streak[key] >= self.required:
                last = self._last_alert.get(key, float("-inf"))
                if now - last >= self.cooldown_sec:
                    confirmed.append(pair)
                    self._last_alert[key] = now

        dead = [k for k in self._streak if k not in current_keys]
        for k in dead:
            del self._streak[k]
        return confirmed


def parse_detections(results, model):
    persons = []
    vehicles = []
    all_dets = []
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            xyxy = [int(v) for v in box.xyxy[0].tolist()]
            det = {
                "box": xyxy,
                "cls_id": cls_id,
                "cls_name": model.names[cls_id],
                "conf": conf,
            }
            all_dets.append(det)
            if cls_id in PERSON_CLASSES:
                persons.append(det)
            elif cls_id in VEHICLE_CLASSES:
                vehicles.append(det)
    return all_dets, persons, vehicles


def find_dangers(persons, vehicles, cfg):
    dangers = []
    for p in persons:
        for v in vehicles:
            iou = bbox_iou(p["box"], v["box"])
            dist = bbox_min_distance(p["box"], v["box"])
            if iou > cfg.overlap_iou:
                dangers.append((p, v, iou, 0.0, "DANGER"))
            elif dist < cfg.proximity_px:
                dangers.append((p, v, 0.0, dist, "WARNING"))
    return dangers


def annotate_frame(frame, all_dets, raw_dangers, confirmed_dangers):
    out = frame.copy()

    for d in all_dets:
        x1, y1, x2, y2 = d["box"]
        color = COLOR_PERSON if d["cls_id"] in PERSON_CLASSES else COLOR_VEHICLE
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f'{d["cls_name"]} {d["conf"]:.0%}'
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw, y1), color, -1)
        cv2.putText(out, label, (x1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    for p, v, iou, dist, severity in raw_dangers:
        pc = box_center(p["box"])
        vc = box_center(v["box"])
        color = COLOR_DANGER if severity == "DANGER" else COLOR_WARNING
        cv2.line(out, pc, vc, color, 1, cv2.LINE_AA)

    for p, v, iou, dist, severity in confirmed_dangers:
        pc = box_center(p["box"])
        vc = box_center(v["box"])

        overlay = out.copy()
        cv2.rectangle(overlay, (p["box"][0], p["box"][1]),
                      (p["box"][2], p["box"][3]), COLOR_DANGER, -1)
        cv2.addWeighted(overlay, 0.3, out, 0.7, 0, out)

        cv2.line(out, pc, vc, COLOR_DANGER, 3, cv2.LINE_AA)

        mid = ((pc[0] + vc[0]) // 2, (pc[1] + vc[1]) // 2)
        cv2.putText(out, f"ALARM: {severity}", (mid[0] - 40, mid[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_DANGER, 2)

    if confirmed_dangers:
        cv2.rectangle(out, (0, 0), (out.shape[1], 40), COLOR_DANGER, -1)
        cv2.putText(out, f"ALARM — {len(confirmed_dangers)} danger(s) detected",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    return out


def main():
    parser = argparse.ArgumentParser(description="Etap 0: PoC video detection")
    parser.add_argument("--input", required=True, help="Input video file")
    parser.add_argument("--output", default="output_annotated.mp4",
                        help="Output annotated video file")
    parser.add_argument("--sample-fps", type=float, default=3.0,
                        help="Frames per second to sample (default: 3)")
    parser.add_argument("--model", default=CONFIG.yolo.model_name,
                        help="YOLO model file")
    parser.add_argument("--show", action="store_true",
                        help="Display live preview")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: input file '{args.input}' not found")
        sys.exit(1)

    print(f"Loading model: {args.model}")
    model = YOLO(args.model)

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        print(f"Error: cannot open video '{args.input}'")
        sys.exit(1)

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_interval = max(1, int(src_fps / args.sample_fps))

    print(f"Video: {width}x{height} @ {src_fps:.1f}fps, {total_frames} frames")
    print(f"Sampling every {frame_interval} frames ({args.sample_fps:.1f} fps)")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_writer = cv2.VideoWriter(args.output, fourcc, args.sample_fps,
                                 (width, height))

    temporal = TemporalFilter(
        required=CONFIG.danger.consecutive_frames_required,
        cooldown_sec=CONFIG.danger.cooldown_seconds,
    )

    frame_idx = 0
    processed = 0
    total_alerts = 0
    t_start = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if frame_idx % frame_interval != 0:
            continue

        results = model.predict(
            frame,
            conf=CONFIG.yolo.confidence_threshold,
            iou=CONFIG.yolo.iou_threshold,
            device=CONFIG.yolo.device,
            imgsz=CONFIG.yolo.img_size,
            classes=ALL_CLASSES,
            verbose=False,
        )

        all_dets, persons, vehicles = parse_detections(results, model)
        raw_dangers = find_dangers(persons, vehicles, CONFIG.danger)
        now = time.time()
        confirmed = temporal.update(raw_dangers, now)

        annotated = annotate_frame(frame, all_dets, raw_dangers, confirmed)
        out_writer.write(annotated)

        processed += 1
        total_alerts += len(confirmed)

        if args.show:
            cv2.imshow("Salamono Safety — Etap 0", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        if processed % 10 == 0:
            elapsed = time.time() - t_start
            fps = processed / elapsed if elapsed > 0 else 0
            print(f"  Frame {frame_idx}/{total_frames} | "
                  f"Processed {processed} | "
                  f"{fps:.1f} fps | "
                  f"Persons: {len(persons)} | Vehicles: {len(vehicles)} | "
                  f"Alerts: {total_alerts}")

    cap.release()
    out_writer.release()
    if args.show:
        cv2.destroyAllWindows()

    elapsed = time.time() - t_start
    print(f"\nDone. Processed {processed} frames in {elapsed:.1f}s "
          f"({processed/elapsed:.1f} fps)")
    print(f"Total alerts: {total_alerts}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
