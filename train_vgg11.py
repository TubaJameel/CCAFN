import train

from models.vgg11 import VGG11Model


if __name__ == "__main__":

    train.main(

        model=VGG11Model(),

        model_name="vgg11"

    )