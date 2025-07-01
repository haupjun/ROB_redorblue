import torch
from torchvision import transforms, models
import torch.nn as nn
from PIL import Image

# 환경변수 설정 (libiomp5md 에러 방지용)
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 1. 모델 구조 정의 (1채널 입력, 2클래스 출력)
model = models.resnet18(weights=None)
model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
model.fc = nn.Linear(model.fc.in_features, 2)

# 2. 저장된 가중치 불러오기
model.load_state_dict(torch.load("analyze_face.pth", map_location=torch.device("cpu")))
model.eval()

# 3. 이미지 전처리 (학습 시 사용한 것과 동일하게)
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# 4. 예측 함수
def predict_image(image: Image.Image):
    img_tensor = transform(image).unsqueeze(0)  # (1, 1, 224, 224)
    with torch.no_grad():
        outputs = model(img_tensor)
        predicted = torch.argmax(outputs, dim=1).item()
    return "진보" if predicted == 0 else "보수"
