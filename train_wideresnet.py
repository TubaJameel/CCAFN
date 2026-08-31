import train

from models.wideresnet_se import WideResNetSE


if __name__ == "__main__":

    train.main(

        model=WideResNetSE(),

        model_name="wideresnet"

    )