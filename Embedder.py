import torch
from torchvision import models

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class Embedder():
    def __init__(self):
        pass

    def calculate_batch_features_and_labels(self, x):
        pass

class Resnet_Embedder(Embedder):
    def __init__(self):
        model = models.resnet50(pretrained=True)
        self.features_model = torch.nn.Sequential(*list(model.children())[:-1])
        self.features_model.eval()
        self.features_model = self.features_model.to(device)

    def calculate_batch_features_and_labels(self, data_batch):
        with torch.no_grad():
            outputs = self.features_model(data_batch[0].squeeze(dim=1).to(device)).detach()
            outputs = outputs.flatten(1)
        labels = data_batch[1]
        return outputs, labels