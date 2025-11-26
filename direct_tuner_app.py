from pathlib import Path
from typing import List, Sequence, Tuple

import cv2
import numpy as np
import streamlit as st

IMAGE_DIR = Path(__file__).parent / "image"

RED_HSV_BOUNDS = (
    (np.array([0, 80, 80]), np.array([10, 255, 255])),
    (np.array([160, 80, 80]), np.array([179, 255, 255])),
)
GREEN_HSV_BOUNDS = (np.array([35, 50, 50]), np.array([90, 255, 255]))


def list_images() -> List[Path]:
    if not IMAGE_DIR.exists():
        return []
    return sorted([p for p in IMAGE_DIR.iterdir() if p.is_file()])


def clean_mask(mask: np.ndarray) -> np.ndarray:
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def find_rectangles(image: np.ndarray, min_area: int, min_aspect: float, max_aspect: float):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    rects = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w * h < min_area:
            continue
        aspect = w / float(h) if h else 0
        if not (min_aspect <= aspect <= max_aspect):
            continue
        rects.append((x, y, w, h))
    return rects


def find_blobs(image: np.ndarray, min_blob_area: int):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    blobs = []
    mask_red = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lower, upper in RED_HSV_BOUNDS:
        mask_red = cv2.bitwise_or(mask_red, cv2.inRange(hsv, lower, upper))
    mask_green = cv2.inRange(hsv, GREEN_HSV_BOUNDS[0], GREEN_HSV_BOUNDS[1])
    for mask, color in ((mask_red, "red"), (mask_green, "green")):
        clean = clean_mask(mask)
        contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) < min_blob_area:
                continue
            blobs.append({"bbox": cv2.boundingRect(cnt), "contour": cnt, "color": color})
    return blobs


def evaluate_circularity(contour: np.ndarray) -> float:
    perimeter = cv2.arcLength(contour, True)
    area = cv2.contourArea(contour)
    if perimeter == 0 or area == 0:
        return 0.0
    return 4 * np.pi * area / (perimeter * perimeter)


def color_for_rect(rect: Tuple[int, int, int, int], blobs: Sequence[dict]) -> Tuple[int, int, int]:
    rx, ry, rw, rh = rect
    best_color = (180, 180, 180)
    best_overlap = 0.0
    for blob in blobs:
        bx, by, bw, bh = blob["bbox"]
        xi1 = max(rx, bx)
        yi1 = max(ry, by)
        xi2 = min(rx + rw, bx + bw)
        yi2 = min(ry + rh, by + bh)
        overlap = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        if overlap > best_overlap:
            best_overlap = overlap
            best_color = (0, 0, 255) if blob["color"] == "red" else (0, 255, 0)
    return best_color


def compute_direct_match(
    rects,
    blobs,
    circle_tol: float,
    intersection_ratio: float,
):
    for rect in rects:
        rx, ry, rw, rh = rect
        rect_area = rw * rh
        for blob in blobs:
            circularity = evaluate_circularity(blob["contour"])
            if abs(1 - circularity) > circle_tol:
                continue
            bx, by, bw, bh = blob["bbox"]
            blob_area = bw * bh
            xi1 = max(rx, bx)
            yi1 = max(ry, by)
            xi2 = min(rx + rw, bx + bw)
            yi2 = min(ry + rh, by + bh)
            inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
            if inter_area > intersection_ratio * rect_area and inter_area > intersection_ratio * blob_area:
                return rect, blob, circularity, inter_area / max(rect_area, blob_area)
    return None, None, None, None


def render(image: np.ndarray, rects, blobs, match_rect):
    canvas = image.copy()
    for rect in rects:
        color = color_for_rect(rect, blobs)
        cv2.rectangle(canvas, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), color, 2)
    for blob in blobs:
        color = (0, 0, 255) if blob["color"] == "red" else (0, 255, 0)
        cv2.drawContours(canvas, [blob["contour"]], -1, color, 2)
    if match_rect:
        rect, blob = match_rect
        cv2.rectangle(
            canvas,
            (rect[0], rect[1]),
            (rect[0] + rect[2], rect[1] + rect[3]),
            (255, 255, 0),
            3,
        )
        cv2.drawContours(canvas, [blob["contour"]], -1, (0, 255, 255), 3)
    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)


def main():
    st.set_page_config(page_title="Tuner Direct", page_icon="🧪", layout="wide")
    st.title("Tuner méthode directe")
    st.caption("Ajuste les paramètres pour trouver un bon couple cercle/intersection.")

    images = list_images()
    if not images:
        st.error("Aucune image disponible dans `image/`.")
        return

    selected = st.selectbox("Image", images, format_func=lambda p: p.name)
    image_bgr = cv2.imread(str(selected))
    if image_bgr is None:
        st.error("Impossible de charger cette image.")
        return

    with st.sidebar:
        st.header("Paramètres")
        circle_tol = st.slider("Tolérance cercle direct", 0.0, 0.6, 0.3, 0.01)
        intersection = st.slider("Intersection minimale", 0.3, 0.95, 0.7, 0.05)
        min_area = st.slider("Aire minimale rectangle", 300, 4000, 1000, 100)
        min_aspect = st.slider("Aspect min (w/h)", 0.5, 1.0, 0.75, 0.05)
        max_aspect = st.slider("Aspect max (w/h)", 1.0, 3.0, 1.33, 0.05)
        min_blob_area = st.slider("Aire min blob", 10, 300, 30, 10)

    rects = find_rectangles(image_bgr, min_area, min_aspect, max_aspect)
    blobs = find_blobs(image_bgr, min_blob_area)
    rect, blob, circularity, overlap = compute_direct_match(rects, blobs, circle_tol, intersection)

    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Visualisation")
        match_pair = (rect, blob) if rect is not None else None
        st.image(
            render(image_bgr, rects, blobs, match_pair),
            caption="Rectangles candidats + blobs",
            use_container_width=True,
        )
    with col2:
        st.subheader("Statistiques")
        st.write(f"Rectangles retenus : **{len(rects)}**")
        st.write(f"Blobs valides : **{len(blobs)}**")
        if rect is None:
            st.error("Aucun match direct avec ces paramètres.")
        else:
            st.success("Match direct trouvé ✅")
            st.write(f"• Couleur : **{blob['color']}**")
            st.write(f"• Circularité : **{circularity:.3f}**")
            st.write(f"• Overlap relatif : **{overlap:.2f}**")


if __name__ == "__main__":
    main()





