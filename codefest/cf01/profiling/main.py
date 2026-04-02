from torchinfo import summary
from torchvision.models import resnet18

model = resnet18()

batch_size: int = 1

s = summary(model, input_size=(batch_size, 3, 224, 244))
