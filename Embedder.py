import torch
from torchvision import models
from torchvision.transforms import Compose
import torchvision.transforms as transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Embedder():
    transform = None
    def __init__(self):
        pass

    def calculate_batch_features(self, x):
        pass


resnet_transform = Compose([transforms.Resize((224, 224)),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                            ])


class ResnetEmbedder(Embedder):
    transform = resnet_transform
    def __init__(self, embedder_dim: int):
        super().__init__()
        self.embedder_dim = embedder_dim
        model = models.resnet50(pretrained=True)
        self.features_model = torch.nn.Sequential(*list(model.children())[:-1])
        self.features_model.eval()
        self.features_model = self.features_model.to(device)

    def calculate_batch_features(self, data_batch):
        with torch.no_grad():
            outputs = self.features_model(data_batch.squeeze(dim=1).to(device)).detach()
            outputs = outputs.flatten(1)
        return outputs
