import gradio as gr
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
from ultralytics import YOLO


class AnalyzeFaceModel(nn.Module):
    """ResNet‑18 기반 얼굴 정치 성향 분류 모델 (1‑채널 입력, 2‑클래스 출력)"""

    def __init__(self):
        super().__init__()
        self.model = models.resnet18(pretrained=False)
        # RGB(3) → Gray(1) 채널 교체
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # 2‑클래스(진보/보수) 분류기 교체
        self.model.fc = nn.Linear(self.model.fc.in_features, 2)
        # 학습된 파라미터 로드
        self.model.load_state_dict(
            torch.load("analyze_face.pth", map_location=torch.device("cpu"))
        )
        self.model.eval()

    def forward(self, x):  # 型 힌트 생략 → PyTorch 원형 유지
        return self.model(x)


# ────────────────────────────────────────────────────────────────────────────────
# 모델 및 전처리 객체 초기화
# ────────────────────────────────────────────────────────────────────────────────

yolo_detector = YOLO("./yolov8n-face-lindevs.pt")  # YOLOv8 얼굴 탐지 가중치
classifier = AnalyzeFaceModel()

transform = transforms.Compose(
    [
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ]
)


# ────────────────────────────────────────────────────────────────────────────────
# YOLO 얼굴 감지 + 정사각형 크롭 함수
# ────────────────────────────────────────────────────────────────────────────────

def crop_face_with_yolo(pil_img: Image.Image) -> Image.Image | None:
    """PIL 이미지를 받아 YOLO로 가장 큰 얼굴을 정사각형 크롭 후 PIL 반환.

    얼굴이 없으면 None 반환.
    """

    # Ultralytics YOLO는 RGB 또는 BGR numpy 모두 허용됨. BGR 사용 시 cv2 관습 부합.
    bgr_img = np.array(pil_img)[:, :, ::-1]  # PIL(RGB) → numpy(BGR)

    results = yolo_detector.predict(bgr_img)
    if len(results[0].boxes) == 0:  # 얼굴 없음
        return None

    # 모든 박스(x_center, y_center, w, h) → numpy(절대 좌표)
    boxes = results[0].boxes.xywh.cpu().numpy()

    # 가장 큰 얼굴(면적 최대) 선택
    areas = boxes[:, 2] * boxes[:, 3]
    x_center, y_center, w, h = boxes[np.argmax(areas)].astype(int)

    # 정사각형 범위 계산
    size = max(w, h)
    x1 = max(0, x_center - size // 2)
    y1 = max(0, y_center - size // 2)
    x2 = min(bgr_img.shape[1], x1 + size)
    y2 = min(bgr_img.shape[0], y1 + size)

    cropped_bgr = bgr_img[y1:y2, x1:x2]
    if cropped_bgr.size == 0:  # 경계 밖 잘림 등 예외
        return None

    cropped_rgb = cropped_bgr[:, :, ::-1]  # BGR → RGB
    return Image.fromarray(cropped_rgb)


# ────────────────────────────────────────────────────────────────────────────────
# Gradio용 추론 함수
# ────────────────────────────────────────────────────────────────────────────────

def predict(pil_img: Image.Image):
    """Gradio 인터페이스에서 호출되는 추론 함수"""

    face_img = crop_face_with_yolo(pil_img)
    if face_img is None:
        return "얼굴을 찾지 못했습니다"

    input_tensor = transform(face_img).unsqueeze(0)  # [1, 1, 224, 224]

    with torch.no_grad():
        outputs = classifier(input_tensor)
        _, pred_idx = torch.max(outputs, 1)

    return "진보" if pred_idx.item() == 0 else "보수"


# ────────────────────────────────────────────────────────────────────────────────
# Gradio UI 정의 및 실행
# ────────────────────────────────────────────────────────────────────────────────

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="이미지를 업로드하세요"),
    outputs=gr.Label(label="분류 결과"),
    title="정치 성향 얼굴 분류기 (YOLO 자동 크롭)",
    description="얼굴이 포함된 사진을 업로드하면, YOLOv8로 얼굴을 자동 감지·크롭 후 진보/보수를 분류합니다.",
)

demo.launch(share=True)
