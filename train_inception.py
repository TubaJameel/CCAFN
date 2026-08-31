import train

from models.inceptionv3 import InceptionV3Model


if __name__ == "__main__":

    train.main(

        model=InceptionV3Model(),

        model_name="inception"

    )