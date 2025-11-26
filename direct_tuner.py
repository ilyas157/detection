#!/usr/bin/env python3
"""
Direct-method tuning helper.

Usage example:
    python direct_tuner.py --image image/boss.jpg --circle-tol 0.25 --intersection 0.8

The script draws every candidate rectangle and highlights the best direct match
so you can iteratively find good parameter values before wiring them into the UI.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

RED_HSV_BOUNDS = (
    (np.array([0, 80, 80]), np.array([10, 255, 255])),
    (np.array([160, 80, 80]), np.array([179, 255, 255])),
)
GREEN_HSV_BOUNDS = (np.array([35, 50, 50]), np.array([90, 255, 255]))


@dataclass
class Rectangle:
    x: int
    y: int
    w: int
    h: int


@dataclass
class Blob:
    bbox: Tuple[int, int, int, int]
    contour: np.ndarray
    color: str  # "red" | "green"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine tune direct detection parameters.")
    parser.add_argument("--image", required=True, help="Path to image to analyse.")
    parser.add_argument("--circle-tol", type=float, default=0.3, help="Max |1 - circularity| allowed (direct).")
    parser.add_argument(
        "--intersection",
        type=float,
        default=0.7,
        help="Required overlap ratio between rect and blob (direct).",
    )
    parser.add_argument("--min-rect-area", type=int, default=1000, help="Minimum area for rectangle candidates.")
    parser.add_argument(
        "--min-rect-aspect",
        type=float,
        default=0.75,
        help="Minimum aspect ratio (width / height) for rectangle candidates.",
    )
    parser.add_argument(
        "--max-rect-aspect",
        type=float,
        default=1.33,
        help="Maximum aspect ratio (width / height) for rectangle candidates.",
    )
    parser.add_argument(
        "--min-blob-area",
        type=int,
        default=30,
        help="Minimum contour area for blobs (after color masking).",
    )
    parser.add_argument(
        "--save",
        type=Path,
        help="Optional path to save the visualization instead of (or in addition to) showing it.",
    )
    parser.add_argument(
        "--no-window",
        action="store_true",
        help="Skip cv2.imshow (useful when running headless, rely on --save).",
    )
    return parser.parse_args()


def clean_mask(mask: np.ndarray) -> np.ndarray:
    kernel = np.ones((3, 3), np.uint8)
    return cv2.morphologyEx(cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel), cv2.MORPH_CLOSE, kernel)


def find_rectangles(
    image: np.ndarray,
    min_area: int,
    min_aspect: float,
    max_aspect: float,
) -> List[Rectangle]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    rectangles: List[Rectangle] = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        if area < min_area:
            continue
        aspect_ratio = w / float(h) if h else 0
        if aspect_ratio < min_aspect or aspect_ratio > max_aspect:
            continue
        rectangles.append(Rectangle(x, y, w, h))
    return rectangles


def find_color_blobs(image: np.ndarray, min_blob_area: int) -> List[Blob]:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    red_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lower, upper in RED_HSV_BOUNDS:
        red_mask = cv2.bitwise_or(red_mask, cv2.inRange(hsv, lower, upper))
    green_mask = cv2.inRange(hsv, GREEN_HSV_BOUNDS[0], GREEN_HSV_BOUNDS[1])

    blobs: List[Blob] = []

    for mask, color in ((red_mask, "red"), (green_mask, "green")):
        clean = clean_mask(mask)
        cnts, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in cnts:
            if cv2.contourArea(cnt) < min_blob_area:
                continue
            blobs.append(Blob(bbox=cv2.boundingRect(cnt), contour=cnt, color=color))
    return blobs


def evaluate_circularity(contour: np.ndarray) -> float:
    perimeter = cv2.arcLength(contour, True)
    area = cv2.contourArea(contour)
    if perimeter == 0 or area == 0:
        return 0.0
    return 4 * np.pi * area / (perimeter * perimeter)


def run_direct_matching(
    rectangles: Sequence[Rectangle],
    blobs: Sequence[Blob],
    circle_tolerance: float,
    intersection_ratio: float,
) -> Tuple[Optional[dict], List[Rectangle]]:
    direct_match = None
    candidates: List[Rectangle] = []

    for rect in rectangles:
        rx, ry, rw, rh = rect.x, rect.y, rect.w, rect.h
        ratio = rw / float(rh)
        if ratio < 0.75 or ratio > 1.33:
            continue
        candidates.append(rect)

        rect_area = rw * rh

        for blob in blobs:
            circ = evaluate_circularity(blob.contour)
            if abs(1 - circ) > circle_tolerance:
                continue

            bx, by, bw, bh = blob.bbox
            blob_area = bw * bh
            xi1 = max(rx, bx)
            yi1 = max(ry, by)
            xi2 = min(rx + rw, bx + bw)
            yi2 = min(ry + rh, by + bh)
            inter_w = max(0, xi2 - xi1)
            inter_h = max(0, yi2 - yi1)
            inter_area = inter_w * inter_h

            if inter_area > intersection_ratio * rect_area and inter_area > intersection_ratio * blob_area:
                direct_match = {
                    "rectangle": rect,
                    "blob": blob,
                    "overlap": inter_area / max(rect_area, blob_area),
                    "circularity": circ,
                }
                return direct_match, candidates

    return direct_match, candidates


def color_for_rect(rect: Rectangle, blobs: Sequence[Blob]) -> Tuple[int, int, int]:
    rx, ry, rw, rh = rect.x, rect.y, rect.w, rect.h
    best_color = (180, 180, 180)
    best_overlap = 0.0
    for blob in blobs:
        bx, by, bw, bh = blob.bbox
        xi1 = max(rx, bx)
        yi1 = max(ry, by)
        xi2 = min(rx + rw, bx + bw)
        yi2 = min(ry + rh, by + bh)
        overlap = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        if overlap > best_overlap:
            best_overlap = overlap
            best_color = (0, 0, 255) if blob.color == "red" else (0, 255, 0)
    return best_color


def visualize(image: np.ndarray, blobs: Sequence[Blob], rectangles: Sequence[Rectangle], direct_match: Optional[dict]):
    viz = image.copy()
    for rect in rectangles:
        cv2.rectangle(
            viz,
            (rect.x, rect.y),
            (rect.x + rect.w, rect.y + rect.h),
            color_for_rect(rect, blobs),
            2,
        )
    for blob in blobs:
        color = (0, 0, 255) if blob.color == "red" else (0, 255, 0)
        cv2.drawContours(viz, [blob.contour], -1, color, 2)

    if direct_match:
        rect = direct_match["rectangle"]
        blob = direct_match["blob"]
        cv2.rectangle(
            viz,
            (rect.x, rect.y),
            (rect.x + rect.w, rect.y + rect.h),
            (255, 255, 0),
            3,
        )
        cv2.drawContours(viz, [blob.contour], -1, (0, 255, 255), 3)
        cv2.putText(
            viz,
            f"Match {blob.color.upper()}  circ={direct_match['circularity']:.2f}",
            (rect.x, max(0, rect.y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2,
        )

    return viz


def main() -> None:
    args = parse_args()
    image_path = Path(args.image)
    image = cv2.imread(str(image_path))
    if image is None:
        raise SystemExit(f"Impossible de lire l'image {image_path}")

    rectangles = find_rectangles(
        image,
        min_area=args.min_rect_area,
        min_aspect=args.min_rect_aspect,
        max_aspect=args.max_rect_aspect,
    )
    blobs = find_color_blobs(image, min_blob_area=args.min_blob_area)
    match, candidates = run_direct_matching(
        rectangles,
        blobs,
        circle_tolerance=args.circle_tol,
        intersection_ratio=args.intersection,
    )

    print(f"Rectangles retenus : {len(candidates)} / {len(rectangles)} totaux")
    print(f"Blobs valides : {len(blobs)}")
    if match:
        print(
            f"✅ Match direct trouvé ({match['blob'].color.upper()}) – "
            f"circularité={match['circularity']:.3f}, overlap~{match['overlap']:.2f}"
        )
    else:
        print("❌ Aucun match direct avec ces paramètres.")

    viz = visualize(image, blobs, candidates, match)

    if args.save:
        cv2.imwrite(str(args.save), viz)
        print(f"Visualisation sauvegardée dans {args.save}")

    if not args.no_window:
        cv2.imshow("Direct Method Tuner", viz)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()





