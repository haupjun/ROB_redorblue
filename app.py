import gradio as gr
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

class AnalyzeFaceModel(nn.Module):
    def __init__(self):
        super(AnalyzeFaceModel, self).__init__()
        self.model = models.resnet18(pretrained=False)
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, 2)  # 진보/보수
        self.model.load_state_dict(torch.load("analyze_face.pth", map_location=torch.device('cpu')))
        self.model.eval()

    def forward(self, x):
        return self.model(x)


model = AnalyzeFaceModel()


transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

def predict(image):
    image = transform(image).unsqueeze(0)  # [1, 1, 224, 224]
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        label = "진보" if predicted.item() == 0 else "보수"
    return label


demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="이미지를 업로드하세요"),
    outputs=gr.Label(label="분류 결과"),
    title="정치 성향 얼굴 분류기",
    description="얼굴 이미지를 업로드하면 진보 또는 보수로 분류합니다."
)


demo.launch(share=True)
