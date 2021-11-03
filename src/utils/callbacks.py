import tensorflow as tf
import os
import numpy as np
import time

def get_timestamp(name):
    timestamp = time.asctime().replace(" ","_").replace(":","_")
    unique_name=f"{name}_at_{timestamp}"
    return unique_name

def get_callbacks(config,X_train):
    logs=config['logs']
    unique_dir_name=get_timestamp("tb_logs")
    TENSORBOARD_ROOT_LOG_DIR=os.path.join(logs['log_dir'],logs['tensorboard_logs'],unique_dir_name)

    os.makedirs(TENSORBOARD_ROOT_LOG_DIR,exist_ok=True)
    tensorboard_cb=tf.keras.callbacks.TensorBoard(TENSORBOARD_ROOT_LOG_DIR)
    file_writer=tf.summary.create_file_writer(TENSORBOARD_ROOT_LOG_DIR)
    with file_writer.as_default():
        images=np.reshape(X_train[10:30],(-1,28,28,1))
        tf.summary.image("20 handWritten example",images,max_outputs=25,step=0)


    params =config['params']
    early_stopping_cb=tf.keras.callbacks.EarlyStopping(patience=params['patience'],restore_best_weights=params['restore_best_weights'])

    
    artifacts_dir = config['artifacts']['artifacts_dir']
    checkpoint_dir = config['artifacts']['checkpoint_dir']
    ckpt_dir=os.path.join(artifacts_dir,checkpoint_dir)
    os.makedirs(ckpt_dir,exist_ok=True)
    ckpt_path=os.path.join(ckpt_dir,'model_ckpt.h5')
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(ckpt_path, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

    return [tensorboard_cb,early_stopping_cb,checkpoint_cb]

