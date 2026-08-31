from models.vgg11 import VGG11Model
from test import main

if __name__ == "__main__":

    model = VGG11Model()

    main(
        model=model,
        model_name="vgg11"
    )