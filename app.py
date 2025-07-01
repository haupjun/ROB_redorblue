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
        self.model.fc = nn.Linear(self.model.fc.in_features, 2)  # ����/����
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
        label = "����" if predicted.item() == 0 else "����"
    return label


demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="�̹����� ���ε��ϼ���"),
    outputs=gr.Label(label="�з� ���"),
    title="��ġ ���� �� �з���",
    description="�� �̹����� ���ε��ϸ� ���� �Ǵ� ������ �з��մϴ�."
)


demo.launch(share=True)
