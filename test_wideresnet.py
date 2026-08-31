from models.wideresnet_se import WideResNetSE
from test import main

if __name__ == "__main__":

    model = WideResNetSE()

    main(
        model=model,
        model_name="wideresnet"
    )