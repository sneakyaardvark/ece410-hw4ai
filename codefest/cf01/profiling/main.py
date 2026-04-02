from pathlib import Path

from torchinfo import summary
from torchvision.models import resnet18

OUTPUT_FILE = Path(__file__).parent / "resnet18_profile.txt"


def main():
    model = resnet18()
    batch_size: int = 1

    s = summary(model, input_size=(batch_size, 3, 224, 224))

    OUTPUT_FILE.write_text(str(s))
    print(f"Summary saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
