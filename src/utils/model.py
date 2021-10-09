import tensorflow as tf
import os,time
import matplotlib.pyplot as plt
import pandas as pd

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

def saveModel(model_dir,model,file_base_name):
    fileName = f"{file_base_name}.h5"
    model_path = os.path.join(model_dir, fileName)
    model.save(model_path)
    print(f"Your model is saved at the following location\n{model_path}")

def save_plot(df,plot_dir,file_base_name):
  fileName = f"{file_base_name}.jpg"
  plotPath = os.path.join(plot_dir, fileName) # model/filename
  print(pd.DataFrame(df))
  pd.DataFrame(df).plot(figsize=(8,5))
  plt.grid(True)
  plt.savefig(plotPath)
  print(f"Your Plot is saved at the following location\n{plotPath}")