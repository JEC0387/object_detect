import colorsys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image


@dataclass(frozen=True)
class Detection:
    class_id: int
    class_name: str
    confidence: float
    xyxy: Tuple[int, int, int, int]


def _stable_color(class_id: int) -> Tuple[int, int, int]:
    # Deterministic vivid-ish colors per class id (BGR for OpenCV)
    h = (class_id * 0.61803398875) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.75, 0.95)
    return int(b * 255), int(g * 255), int(r * 255)


@st.cache_resource(show_spinner=False)
def load_model():
    from ultralytics import YOLO

    model = YOLO("yolov8n.pt")  # auto-download on first run
    return model


def preprocess_image(pil_img: Image.Image, max_side: int = 1280) -> np.ndarray:
    pil_img = pil_img.convert("RGB")
    w, h = pil_img.size
    scale = min(1.0, max_side / max(w, h))
    if scale < 1.0:
        pil_img = pil_img.resize((int(w * scale), int(h * scale)), Image.BILINEAR)
    rgb = np.array(pil_img)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return bgr


def run_detection(
    model: Any,
    bgr_img: np.ndarray,
    conf: float = 0.25,
    iou: float = 0.45,
    imgsz: int = 640,
) -> List[Detection]:
    results = model.predict(
        source=bgr_img,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        device="cpu",
        verbose=False,
    )
    r0 = results[0]

    names: Dict[int, str] = getattr(r0, "names", {}) or {}
    boxes = getattr(r0, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return []

    xyxy = boxes.xyxy.detach().cpu().numpy()
    cls = boxes.cls.detach().cpu().numpy().astype(int)
    confs = boxes.conf.detach().cpu().numpy()

    dets: List[Detection] = []
    for (x1, y1, x2, y2), cid, c in zip(xyxy, cls, confs):
        dets.append(
            Detection(
                class_id=int(cid),
                class_name=str(names.get(int(cid), f"class_{int(cid)}")),
                confidence=float(c),
                xyxy=(int(x1), int(y1), int(x2), int(y2)),
            )
        )
    return dets


def draw_boxes(bgr_img: np.ndarray, detections: List[Detection]) -> np.ndarray:
    out = bgr_img.copy()
    for d in detections:
        x1, y1, x2, y2 = d.xyxy
        color = _stable_color(d.class_id)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        label = f"{d.class_name} {d.confidence:.2f}"
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        y_text = max(0, y1 - th - baseline - 6)
        cv2.rectangle(out, (x1, y_text), (x1 + tw + 6, y_text + th + baseline + 6), color, -1)
        cv2.putText(
            out,
            label,
            (x1 + 3, y_text + th + 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
    return out


def display_results(
    pil_original: Image.Image,
    bgr_with_boxes: np.ndarray,
    detections: List[Detection],
):
    rgb_with_boxes = cv2.cvtColor(bgr_with_boxes, cv2.COLOR_BGR2RGB)

    left, right = st.columns([1.2, 1.0], gap="large")
    with left:
        st.subheader("결과 이미지")
        st.image(rgb_with_boxes, use_container_width=True)

    with right:
        st.subheader("탐지 결과")
        st.metric("총 객체 수", len(detections))

        counts: Dict[str, int] = {}
        for d in detections:
            counts[d.class_name] = counts.get(d.class_name, 0) + 1

        if counts:
            st.write("객체 개수(클래스별)")
            st.dataframe(
                pd.DataFrame(
                    [{"class": k, "count": v} for k, v in sorted(counts.items(), key=lambda x: (-x[1], x[0]))]
                ),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("탐지된 객체가 없습니다.")

    st.subheader("객체 리스트")
    if detections:
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "class": d.class_name,
                        "confidence": round(d.confidence, 4),
                        "x1": d.xyxy[0],
                        "y1": d.xyxy[1],
                        "x2": d.xyxy[2],
                        "y2": d.xyxy[3],
                    }
                    for d in detections
                ]
            ).sort_values(["class", "confidence"], ascending=[True, False]),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.empty()


def _get_input_image() -> Optional[Image.Image]:
    st.write("입력 방식 선택")
    tab1, tab2 = st.tabs(["모바일 촬영", "이미지 업로드"])

    pil_img: Optional[Image.Image] = None
    with tab1:
        cam = st.camera_input("카메라로 촬영")
        if cam is not None:
            pil_img = Image.open(cam).convert("RGB")

    with tab2:
        up = st.file_uploader("이미지 파일 업로드", type=["png", "jpg", "jpeg", "webp"])
        if up is not None:
            pil_img = Image.open(up).convert("RGB")

    return pil_img


def main():
    st.set_page_config(page_title="모바일 이미지 객체 탐지 (YOLOv8)", layout="wide")

    st.title("모바일 이미지 객체 탐지 앱 (YOLOv8)")
    st.caption("카메라 촬영 또는 이미지 업로드 → YOLOv8 추론 → 바운딩 박스/클래스/신뢰도/통계 출력")

    with st.expander("설정", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            conf = st.slider("Confidence Threshold", 0.05, 0.95, 0.25, 0.05)
        with col2:
            iou = st.slider("IoU Threshold", 0.05, 0.95, 0.45, 0.05)
        with col3:
            imgsz = st.select_slider("Image Size (imgsz)", options=[320, 416, 512, 640, 768, 896, 1024], value=640)

        max_side = st.select_slider("입력 이미지 리사이즈(max side)", options=[640, 960, 1280, 1600, 1920], value=1280)

    pil_img = _get_input_image()

    action_col1, action_col2 = st.columns([1, 3])
    with action_col1:
        analyze = st.button("분석 실행", type="primary", use_container_width=True, disabled=(pil_img is None))
    with action_col2:
        if pil_img is None:
            st.info("촬영하거나 이미지를 업로드한 뒤 분석을 실행하세요.")

    if not analyze:
        if pil_img is not None:
            st.subheader("입력 이미지")
            st.image(pil_img, use_container_width=True)
        return

    with st.spinner("모델 로딩 및 추론 중... (첫 실행은 모델 다운로드로 시간이 걸릴 수 있어요)"):
        model = load_model()
        bgr = preprocess_image(pil_img, max_side=max_side)
        dets = run_detection(model, bgr, conf=conf, iou=iou, imgsz=imgsz)
        boxed = draw_boxes(bgr, dets)

    display_results(pil_img, boxed, dets)


if __name__ == "__main__":
    main()

import colorsys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image


@dataclass(frozen=True)
class Detection:
    class_id: int
    class_name: str
    confidence: float
    xyxy: Tuple[int, int, int, int]


def _stable_color(class_id: int) -> Tuple[int, int, int]:
    # Deterministic vivid-ish colors per class id (BGR for OpenCV)
    h = (class_id * 0.61803398875) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.75, 0.95)
    return int(b * 255), int(g * 255), int(r * 255)


@st.cache_resource(show_spinner=False)
def load_model():
    from ultralytics import YOLO

    model = YOLO("yolov8n.pt")  # auto-download on first run
    return model


def preprocess_image(pil_img: Image.Image, max_side: int = 1280) -> np.ndarray:
    pil_img = pil_img.convert("RGB")
    w, h = pil_img.size
    scale = min(1.0, max_side / max(w, h))
    if scale < 1.0:
        pil_img = pil_img.resize((int(w * scale), int(h * scale)), Image.BILINEAR)
    rgb = np.array(pil_img)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return bgr


def run_detection(
    model: Any,
    bgr_img: np.ndarray,
    conf: float = 0.25,
    iou: float = 0.45,
    imgsz: int = 640,
) -> List[Detection]:
    results = model.predict(
        source=bgr_img,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        device="cpu",
        verbose=False,
    )
    r0 = results[0]

    names: Dict[int, str] = getattr(r0, "names", {}) or {}
    boxes = getattr(r0, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return []

    xyxy = boxes.xyxy.detach().cpu().numpy()
    cls = boxes.cls.detach().cpu().numpy().astype(int)
    confs = boxes.conf.detach().cpu().numpy()

    dets: List[Detection] = []
    for (x1, y1, x2, y2), cid, c in zip(xyxy, cls, confs):
        dets.append(
            Detection(
                class_id=int(cid),
                class_name=str(names.get(int(cid), f"class_{int(cid)}")),
                confidence=float(c),
                xyxy=(int(x1), int(y1), int(x2), int(y2)),
            )
        )
    return dets


def draw_boxes(bgr_img: np.ndarray, detections: List[Detection]) -> np.ndarray:
    out = bgr_img.copy()
    for d in detections:
        x1, y1, x2, y2 = d.xyxy
        color = _stable_color(d.class_id)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        label = f"{d.class_name} {d.confidence:.2f}"
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        y_text = max(0, y1 - th - baseline - 6)
        cv2.rectangle(out, (x1, y_text), (x1 + tw + 6, y_text + th + baseline + 6), color, -1)
        cv2.putText(
            out,
            label,
            (x1 + 3, y_text + th + 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
    return out


def display_results(
    pil_original: Image.Image,
    bgr_with_boxes: np.ndarray,
    detections: List[Detection],
):
    rgb_with_boxes = cv2.cvtColor(bgr_with_boxes, cv2.COLOR_BGR2RGB)

    left, right = st.columns([1.2, 1.0], gap="large")
    with left:
        st.subheader("결과 이미지")
        st.image(rgb_with_boxes, use_container_width=True)

    with right:
        st.subheader("탐지 결과")
        st.metric("총 객체 수", len(detections))

        counts: Dict[str, int] = {}
        for d in detections:
            counts[d.class_name] = counts.get(d.class_name, 0) + 1

        if counts:
            st.write("객체 개수(클래스별)")
            st.dataframe(
                pd.DataFrame(
                    [{"class": k, "count": v} for k, v in sorted(counts.items(), key=lambda x: (-x[1], x[0]))]
                ),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("탐지된 객체가 없습니다.")

    st.subheader("객체 리스트")
    if detections:
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "class": d.class_name,
                        "confidence": round(d.confidence, 4),
                        "x1": d.xyxy[0],
                        "y1": d.xyxy[1],
                        "x2": d.xyxy[2],
                        "y2": d.xyxy[3],
                    }
                    for d in detections
                ]
            ).sort_values(["class", "confidence"], ascending=[True, False]),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.empty()


def _get_input_image() -> Optional[Image.Image]:
    st.write("입력 방식 선택")
    tab1, tab2 = st.tabs(["모바일 촬영", "이미지 업로드"])

    pil_img: Optional[Image.Image] = None
    with tab1:
        cam = st.camera_input("카메라로 촬영")
        if cam is not None:
            pil_img = Image.open(cam).convert("RGB")

    with tab2:
        up = st.file_uploader("이미지 파일 업로드", type=["png", "jpg", "jpeg", "webp"])
        if up is not None:
            pil_img = Image.open(up).convert("RGB")

    return pil_img


def main():
    st.set_page_config(page_title="모바일 이미지 객체 탐지 (YOLOv8)", layout="wide")

    st.title("모바일 이미지 객체 탐지 앱 (YOLOv8)")
    st.caption("카메라 촬영 또는 이미지 업로드 → YOLOv8 추론 → 바운딩 박스/클래스/신뢰도/통계 출력")

    with st.expander("설정", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            conf = st.slider("Confidence Threshold", 0.05, 0.95, 0.25, 0.05)
        with col2:
            iou = st.slider("IoU Threshold", 0.05, 0.95, 0.45, 0.05)
        with col3:
            imgsz = st.select_slider("Image Size (imgsz)", options=[320, 416, 512, 640, 768, 896, 1024], value=640)

        max_side = st.select_slider("입력 이미지 리사이즈(max side)", options=[640, 960, 1280, 1600, 1920], value=1280)

    pil_img = _get_input_image()

    action_col1, action_col2 = st.columns([1, 3])
    with action_col1:
        analyze = st.button("분석 실행", type="primary", use_container_width=True, disabled=(pil_img is None))
    with action_col2:
        if pil_img is None:
            st.info("촬영하거나 이미지를 업로드한 뒤 분석을 실행하세요.")

    if not analyze:
        if pil_img is not None:
            st.subheader("입력 이미지")
            st.image(pil_img, use_container_width=True)
        return

    with st.spinner("모델 로딩 및 추론 중... (첫 실행은 모델 다운로드로 시간이 걸릴 수 있어요)"):
        model = load_model()
        bgr = preprocess_image(pil_img, max_side=max_side)
        dets = run_detection(model, bgr, conf=conf, iou=iou, imgsz=imgsz)
        boxed = draw_boxes(bgr, dets)

    display_results(pil_img, boxed, dets)


if __name__ == "__main__":
    main()

