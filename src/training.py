from utils.common import readConfig
from utils.data_mgmt import get_data
from utils.model import create_model, saveModel
import argparse
import os


def training(config_path):
    config = readConfig(config_path)
    validation_data_size = config['params']['validation_data_size']
    LOSS_FUNCTION = config['params']['loss_function']
    METRICS = config['params']['metrics']
    OPTIMIZER = config['params']['optimizer']
    (X_train, y_train), (X_valid, y_valid), (X_test, y_test) = get_data(validation_data_size)

    model = create_model(LOSS_FUNCTION, OPTIMIZER, METRICS)

    EPOCHS = config['params']['epochs']
    VALIDATION_SET = (X_valid, y_valid)
    history = model.fit(X_train, y_train, epochs=EPOCHS,
                        validation_data=VALIDATION_SET)

    artifacts_dir=config['artifacts']['artifacts_dir']
    model_dir = config['artifacts']['model_dir']
    model_dir_path=os.path.join(artifacts_dir,model_dir)
    os.makedirs(model_dir_path,exist_ok=True)
    save = input("Do you want to Save the model(y/n)")
    if (save == 'y'):

        saveModel(model_dir_path, model)


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--config', '-c', default='config.yaml')
    parsed_args = args.parse_args()
    training(config_path=parsed_args.config)
