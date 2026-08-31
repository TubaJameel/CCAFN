from models.inceptionv3 import InceptionV3Model
from test import main

if __name__ == "__main__":

    model = InceptionV3Model()

    main(
        model=model,
        model_name="inception"
    )