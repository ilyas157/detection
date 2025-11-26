from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import streamlit as st

st.set_page_config(
    page_title="Analyseur de feux",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

IMAGE_DIR = Path(__file__).parent / "image"
SUPPORTED_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
DEFAULT_CIRCLE_TOLERANCE = 0.3
DEFAULT_DIRECT_CIRCLE_TOLERANCE = 0.3
DEFAULT_DIRECT_INTERSECTION_RATIO = 0.7


def list_local_images() -> List[Path]:
    if not IMAGE_DIR.exists():
        return []
    return sorted(
        [
            path
            for path in IMAGE_DIR.iterdir()
            if path.suffix.lower() in SUPPORTED_EXT and path.is_file()
        ],
        key=lambda p: p.name.lower(),
    )


def _clean_mask(mask: np.ndarray) -> np.ndarray:
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def _find_blobs(mask: np.ndarray, color_name: str, hsv_image: np.ndarray, min_brightness=120):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blobs = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 30:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        cx = int(x + w / 2)
        cy = int(y + h / 2)
        roi = hsv_image[y : y + h, x : x + w]
        mean_v = np.mean(roi[:, :, 2]) if roi.size else 0
        if mean_v < min_brightness:
            continue
        blobs.append(
            {"cx": cx, "cy": cy, "bbox": (x, y, w, h), "color": color_name, "contour": cnt}
        )
    return blobs


def _to_rgb(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _find_tuner_rectangles(image: np.ndarray, min_area: int, min_aspect: float, max_aspect: float):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    rects = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        if area < min_area:
            continue
        aspect = w / float(h) if h else 0
        if not (min_aspect <= aspect <= max_aspect):
            continue
        rects.append((x, y, w, h))
    return rects


def _find_tuner_blobs(image: np.ndarray, min_blob_area: int):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    blobs = []
    lower_red1, upper_red1 = np.array([0, 80, 80]), np.array([10, 255, 255])
    lower_red2, upper_red2 = np.array([160, 80, 80]), np.array([179, 255, 255])
    mask_red = cv2.bitwise_or(cv2.inRange(hsv, lower_red1, upper_red1), cv2.inRange(hsv, lower_red2, upper_red2))
    mask_green = cv2.inRange(hsv, np.array([35, 50, 50]), np.array([90, 255, 255]))

    for mask, color in ((mask_red, "red"), (mask_green, "green")):
        clean = _clean_mask(mask)
        contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) < min_blob_area:
                continue
            blobs.append({"bbox": cv2.boundingRect(cnt), "contour": cnt, "color": color})
    return blobs


def _compute_tuner_match(rects, blobs, circle_tol, intersection):
    for rect in rects:
        rx, ry, rw, rh = rect
        rect_area = rw * rh
        for blob in blobs:
            perimeter = cv2.arcLength(blob["contour"], True)
            area = cv2.contourArea(blob["contour"])
            if perimeter == 0 or area == 0:
                continue
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            if abs(1 - circularity) > circle_tol:
                continue
            bx, by, bw, bh = blob["bbox"]
            blob_area = bw * bh
            xi1 = max(rx, bx)
            yi1 = max(ry, by)
            xi2 = min(rx + rw, bx + bw)
            yi2 = min(ry + rh, by + bh)
            inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
            if inter_area > intersection * rect_area and inter_area > intersection * blob_area:
                return rect, blob, circularity, inter_area / max(rect_area, blob_area)
    return None, None, None, None


def _color_for_tuner_rect(rect, blobs):
    rx, ry, rw, rh = rect
    best_overlap = 0.0
    best_color = (180, 180, 180)
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


def _render_tuner_view(image_bgr, rects, blobs, match_pair):
    canvas = image_bgr.copy()
    for rect in rects:
        color = _color_for_tuner_rect(rect, blobs)
        cv2.rectangle(canvas, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), color, 2)
    for blob in blobs:
        color = (0, 0, 255) if blob["color"] == "red" else (0, 255, 0)
        cv2.drawContours(canvas, [blob["contour"]], -1, color, 2)
    if match_pair is not None:
        rect, blob = match_pair
        cv2.rectangle(canvas, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (255, 255, 0), 3)
        cv2.drawContours(canvas, [blob["contour"]], -1, (0, 255, 255), 3)
    return _to_rgb(canvas)


def analyze_image(
    image: np.ndarray,
    circle_tolerance: float = DEFAULT_CIRCLE_TOLERANCE,
    direct_circle_tolerance: float = DEFAULT_DIRECT_CIRCLE_TOLERANCE,
    direct_intersection_ratio: float = DEFAULT_DIRECT_INTERSECTION_RATIO,
) -> Dict[str, Optional[str]]:
    output = image.copy()
    candidate_viz = image.copy()

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    filtered_rectangles = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        aspect_ratio = h / float(w)
        if area < 1000:
            continue
        if not (0.8 <= aspect_ratio <= 10):
            continue
        filtered_rectangles.append({"x": x, "y": y, "w": w, "h": h})

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_red1, upper_red1 = np.array([0, 80, 80]), np.array([10, 255, 255])
    lower_red2, upper_red2 = np.array([160, 80, 80]), np.array([179, 255, 255])

    mask_red = cv2.bitwise_or(
        cv2.inRange(hsv, lower_red1, upper_red1),
        cv2.inRange(hsv, lower_red2, upper_red2),
    )
    mask_green = cv2.inRange(hsv, np.array([35, 50, 50]), np.array([90, 255, 255]))

    mask_red_clean = _clean_mask(mask_red)
    mask_green_clean = _clean_mask(mask_green)

    blobs_red = _find_blobs(mask_red_clean.copy(), "red", hsv)
    blobs_green = _find_blobs(mask_green_clean.copy(), "green", hsv)
    all_blobs = blobs_red + blobs_green

    direct_match_found = None
    candidate_rects_to_draw = []

    def rect_color_from_blobs(rect, blobs):
        rx, ry, rw, rh = rect["x"], rect["y"], rect["w"], rect["h"]
        best_blob = None
        best_overlap = 0.0
        for blob in blobs:
            bx, by, bw, bh = blob["bbox"]
            xi1 = max(rx, bx)
            yi1 = max(ry, by)
            xi2 = min(rx + rw, bx + bw)
            yi2 = min(ry + rh, by + bh)
            inter_w = max(0, xi2 - xi1)
            inter_h = max(0, yi2 - yi1)
            overlap = inter_w * inter_h
            if overlap > best_overlap:
                best_overlap = overlap
                best_blob = blob
        if best_blob:
            return (0, 0, 255) if best_blob["color"] == "red" else (0, 255, 0)
        return (180, 180, 180)

    for rect in filtered_rectangles:
        rx, ry, rw, rh = rect["x"], rect["y"], rect["w"], rect["h"]
        ratio = float(rw) / rh
        if ratio < 0.75 or ratio > 1.33:
            continue
        candidate_rects_to_draw.append(rect)

        rect_area = rw * rh

        for blob in all_blobs:
            contour = blob["contour"]
            perimeter = cv2.arcLength(contour, True)
            area = cv2.contourArea(contour)
            if perimeter == 0 or area == 0:
                continue
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            if abs(1 - circularity) > direct_circle_tolerance:
                continue

            bx, by, bw, bh = blob["bbox"]
            blob_area = bw * bh

            xi1 = max(rx, bx)
            yi1 = max(ry, by)
            xi2 = min(rx + rw, bx + bw)
            yi2 = min(ry + rh, by + bh)

            inter_w = max(0, xi2 - xi1)
            inter_h = max(0, yi2 - yi1)
            inter_area = inter_w * inter_h

            if inter_area > direct_intersection_ratio * rect_area and inter_area > direct_intersection_ratio * blob_area:
                direct_match_found = {
                    "rectangle": rect,
                    "blob": blob,
                    "color": blob["color"],
                    "contour": blob["contour"],
                }
                break
        if direct_match_found:
            break

    for rect in candidate_rects_to_draw:
        cv2.rectangle(
            candidate_viz,
            (rect["x"], rect["y"]),
            (rect["x"] + rect["w"], rect["y"] + rect["h"]),
            rect_color_from_blobs(rect, all_blobs),
            2,
        )

    for blob in blobs_red:
        cv2.drawContours(candidate_viz, [blob["contour"]], -1, (0, 0, 255), 2)
    for blob in blobs_green:
        cv2.drawContours(candidate_viz, [blob["contour"]], -1, (0, 255, 0), 2)

    message = None

    color_label = {"red": "rouge", "green": "vert"}

    if direct_match_found:
        res = direct_match_found
        x, y, w, h = res["rectangle"]["x"], res["rectangle"]["y"], res["rectangle"]["w"], res["rectangle"]["h"]
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 3)
        cv2.drawContours(output, [res["contour"]], -1, (0, 255, 255), 2)
        cv2.putText(
            output,
            f"Direct : {color_label.get(res['color'], res['color']).upper()}",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2,
        )
        color_fr = color_label.get(res["color"], res["color"])
        message = f" Détection directe : {color_fr.upper()}"
    else:

        def match_blobs_to_rectangles(blobs, rectangles):
            results = []
            for rect in rectangles:
                x, y, w, h = rect["x"], rect["y"], rect["w"], rect["h"]
                x2, y2 = x + w, y + h
                blobs_inside = []
                for blob in blobs:
                    cx, cy = blob["cx"], blob["cy"]
                    if not (x <= cx <= x2 and y <= cy <= y2):
                        continue
                    in_expected_position = False
                    if blob["color"] == "red" and y <= cy <= y + 0.4 * h:
                        in_expected_position = True
                    #elif blob["color"] in ("orange", "yellow") and y + 0.4 * h <= cy <= y + 0.6 * h:
                    #   in_expected_position = True
                    elif blob["color"] == "green" and y + 0.6 * h <= cy <= y + h:
                        in_expected_position = True

                    blob_copy = blob.copy()
                    blob_copy["in_expected_position"] = in_expected_position
                    blobs_inside.append(blob_copy)

                if blobs_inside:
                    results.append({"rectangle": rect, "blobs": blobs_inside})
            return results

        matches = match_blobs_to_rectangles(all_blobs, filtered_rectangles)
        active_light = None
        max_area = -1

        for m in matches:
            rect = m["rectangle"]
            x, y, w, h = rect["x"], rect["y"], rect["w"], rect["h"]
            for b in m["blobs"]:
                if not b.get("in_expected_position", False):
                    continue
                color = b["color"]
                #if color == "orange":
                #    continue

                x0, y0, bw, bh = b["bbox"]
                mask = mask_red_clean if color == "red" else mask_green_clean

                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for cnt in contours:
                    perimeter = cv2.arcLength(cnt, True)
                    area = cv2.contourArea(cnt)
                    if perimeter == 0 or area == 0:
                        continue
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    if abs(1 - circularity) > circle_tolerance:
                        continue
                    bx, by, bwc, bhc = cv2.boundingRect(cnt)
                    if not (bx >= x0 and by >= y0 and bx + bwc <= x0 + bw and by + bhc <= y0 + bh):
                        continue
                    if area > max_area:
                        max_area = area
                        active_light = {"rectangle": rect, "blob": b, "contour": cnt, "color": color}

        if active_light:
            rect = active_light["rectangle"]
            x, y, w, h = rect["x"], rect["y"], rect["w"], rect["h"]
            cv2.rectangle(output, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.drawContours(output, [active_light["contour"]], -5, (0, 255, 255), 2)
            color_fr = color_label.get(active_light["color"], active_light["color"])
            message = f"🔍 Détection indirecte : {color_fr}"
        else:
            message = "Aucun feu détecté."

    return {
        "message": message,
        "output": _to_rgb(output),
        "candidate_viz": _to_rgb(candidate_viz),
    }


def sidebar_library(image_paths: List[Path]):
    st.sidebar.header("Bibliothèque locale")
    if not image_paths:
        st.sidebar.info("Ajoute tes images dans le dossier `image/` pour les voir ici.")
        return

    if "selected_image" not in st.session_state and image_paths:
        st.session_state["selected_image"] = str(image_paths[0])

    for path in image_paths:
        with st.sidebar.expander(path.name, expanded=str(path) == st.session_state.get("selected_image")):
            st.image(str(path),use_container_width=True)
            if st.button("Choisir", key=f"select_{path.name}"):
                st.session_state["selected_image"] = str(path)


def main():
    image_paths = list_local_images()
    sidebar_library(image_paths)

    st.title("Bibliothèque & Analyseur de feux")
    st.caption(
        ""
    )

    selected = st.session_state.get("selected_image")
    if not selected:
        st.warning("Aucune image disponible. Ajoute des fichiers dans `image/`.")
        return

    selected_path = Path(selected)
    if not selected_path.exists():
        st.error("L'image sélectionnée est introuvable. Recharge la page.")
        return

    image_bgr = cv2.imread(str(selected_path))
    if image_bgr is None:
        st.error("Impossible de lire cette image.")
        return

    tab_main, tab_tuner = st.tabs(["Analyse principale", "Tuner direct"])

    with tab_main:
        slider_col1, slider_col2 = st.columns(2)
        with slider_col1:
            direct_circle_tol = st.slider(
                "Tolérance cercle (direct)",
                min_value=0.0,
                max_value=0.6,
                value=DEFAULT_DIRECT_CIRCLE_TOLERANCE,
                step=0.01,
                help="Plus la tolérance est faible, plus la forme doit être circulaire.",
            )
        with slider_col2:
            direct_intersection_ratio = st.slider(
                "Intersection requise (direct)",
                min_value=0.3,
                max_value=0.95,
                value=DEFAULT_DIRECT_INTERSECTION_RATIO,
                step=0.05,
                help="Pourcentage de recouvrement rectangle/blob exigé pour le match direct.",
            )

        st.subheader(f"Image sélectionnée : {selected_path.name}")
        st.image(_to_rgb(image_bgr), caption=selected_path.name,use_container_width=True)

        if st.button("Analyser", type="primary"):
            with st.spinner("Analyse en cours…"):
                result = analyze_image(
                    image_bgr,
                    circle_tolerance=DEFAULT_CIRCLE_TOLERANCE,
                    direct_circle_tolerance=direct_circle_tol,
                    direct_intersection_ratio=direct_intersection_ratio,
                )
            st.success(result["message"])
            col1, col2 = st.columns(2)
            with col1:
                st.image(result["candidate_viz"], caption="Candidats & contours")
            with col2:
                st.image(result["output"], caption="Résultat final")
        else:
            st.info("Clique sur “Analyser” pour exécuter l'algorithme. Rien n'est sauvegardé.")

    with tab_tuner:
        st.subheader("Tuner direct (expérimentation)")
        tuner_image = st.selectbox(
            "Image pour le tuner",
            image_paths,
            index=image_paths.index(selected_path) if selected_path in image_paths else 0,
            format_func=lambda p: p.name,
        )
        tuner_bgr = cv2.imread(str(tuner_image))
        if tuner_bgr is None:
            st.error("Impossible de lire cette image.")
        else:
            col_params1, col_params2 = st.columns(2)
            with col_params1:
                tuner_circle_tol = st.slider("Tolérance cercle direct", 0.0, 0.6, 0.3, 0.01, key="tuner_circle")
                tuner_min_area = st.slider("Aire minimale rectangle", 300, 4000, 1000, 100, key="tuner_area")
                tuner_min_blob_area = st.slider("Aire minimale blob", 10, 300, 30, 10, key="tuner_blob_area")
            with col_params2:
                tuner_intersection = st.slider("Intersection minimale", 0.3, 0.95, 0.7, 0.05, key="tuner_intersection")
                tuner_min_aspect = st.slider("Aspect min (w/h)", 0.5, 1.0, 0.75, 0.05, key="tuner_min_aspect")
                tuner_max_aspect = st.slider("Aspect max (w/h)", 1.0, 3.0, 1.33, 0.05, key="tuner_max_aspect")

            rects = _find_tuner_rectangles(tuner_bgr, tuner_min_area, tuner_min_aspect, tuner_max_aspect)
            blobs = _find_tuner_blobs(tuner_bgr, tuner_min_blob_area)
            match_rect, match_blob, circ, overlap = _compute_tuner_match(rects, blobs, tuner_circle_tol, tuner_intersection)

            col_viz, col_stats = st.columns([2, 1])
            with col_viz:
                st.image(
                    _render_tuner_view(
                        tuner_bgr,
                        rects,
                        blobs,
                        (match_rect, match_blob) if match_rect is not None else None,
                    ),
                    caption="Rectangles candidats & blobs",
                    use_container_width=True,
                )
            with col_stats:
                st.write(f"Rectangles retenus : **{len(rects)}**")
                st.write(f"Blobs valides : **{len(blobs)}**")
                if match_rect is None:
                    st.error("Aucun match direct avec ces paramètres.")
                else:
                    st.success("Match direct trouvé ✅")
                    st.write(f"Couleur : **{match_blob['color']}**")
                    st.write(f"Circularité : **{circ:.3f}**")
                    st.write(f"Overlap relatif : **{overlap:.2f}**")


if __name__ == "__main__":
    main()