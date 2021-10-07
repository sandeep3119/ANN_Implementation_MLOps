import tensorflow as tf
import os,time


def create_model(LOSS_FUNCTION,OPTIMIZER,METRICS):
    LAYERS = [tf.keras.layers.Flatten(input_shape=[28, 28], name="inputLayer"),
          tf.keras.layers.Dense(300, activation="relu", name="hiddenLayer1"),
          tf.keras.layers.Dense(100, activation="relu", name="hiddenLayer2"),
          tf.keras.layers.Dense(10, activation="softmax", name="outputLayer")]

    model = tf.keras.models.Sequential(LAYERS)  
    model.compile(loss=LOSS_FUNCTION,
              optimizer=OPTIMIZER,
              metrics=METRICS) 
    model.summary() 
    return model

def saveModel(model_dir,model):  
    fileName = time.strftime("Model_%Y_%m_%d_%H_%M_%S_.h5")  
    model_path = os.path.join(model_dir, fileName)
    model.save(model_path)
    print(f"your model is saved at the following location\n{model_path}")
