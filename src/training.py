from utils.common import readConfig
from utils.data_mgmt import get_data
from utils.model import create_model, saveModel, save_plot
from utils.callbacks import get_callbacks
import argparse
import os
import time


def training(config_path):
    config = readConfig(config_path)
    validation_data_size = config['params']['validation_data_size']
    LOSS_FUNCTION = config['params']['loss_function']
    METRICS = config['params']['metrics']
    OPTIMIZER = config['params']['optimizer']
    artifacts_dir = config['artifacts']['artifacts_dir']
    model_name = config['artifacts']['model_name']
    file_base_name = time.strftime(f"{model_name}_%Y_%m_%d_%H_%M_%S_")
    (X_train, y_train), (X_valid, y_valid), (X_test, y_test) = get_data(validation_data_size)

    model = create_model(LOSS_FUNCTION, OPTIMIZER, METRICS)

    EPOCHS = config['params']['epochs']
    VALIDATION_SET = (X_valid, y_valid)
    CALLBACKS=get_callbacks(config,X_train)
    history = model.fit(X_train, y_train, epochs=EPOCHS,
                        validation_data=VALIDATION_SET,
                        callbacks=CALLBACKS)


    model_dir = config['artifacts']['model_dir']
    model_dir_path=os.path.join(artifacts_dir,model_dir)
    os.makedirs(model_dir_path,exist_ok=True)
    saveModel(model_dir_path, model,file_base_name)

    plot_dir=config['artifacts']['plots_dir']
    plot_dir_path=os.path.join(artifacts_dir,plot_dir)
    os.makedirs(plot_dir_path,exist_ok=True)
    save_plot(history.history,plot_dir_path,file_base_name)


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--config', '-c', default='config.yaml')
    parsed_args = args.parse_args()
    training(config_path=parsed_args.config)
