import json
import time
import keras
from sklearn.model_selection import train_test_split
import numpy as np
import os
from data_generator import CrossSectionDataBase

def init_model(size):
    
    width, height = size
    
    folder_path = f"{width}x{height}"
    file_path = f"{folder_path}\model_start.h5"
    
    if not os.path.exists(file_path):
        model = keras.Sequential([
                keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(width, height, 1)),
                keras.layers.MaxPooling2D((2, 2)),
                keras.layers.Flatten(),
                keras.layers.Dense(128, activation='relu'),
                keras.layers.Dense(4)  # 4 output neurons for the four float numbers
        ])
        
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
        
        with open(f"{folder_path}\histories.json", 'w') as f:
            json.dump({},f)

        model.save(file_path)
  
def train_models_by_size(size,start,end,num_run):

    width, height = size

    for n in range(start,end,100):
        epoch = num_run//n
        print(f"\n\n\nI do {epoch} epoch on {n} input data, so I iterate trough {epoch*n} image\n")
        
        # load the starting model
        model = keras.models.load_model(f"{width}x{height}/model_start.h5")
        
        # load the train data set
        data_base = CrossSectionDataBase(size)
        data_base.load('train',folder=f"{width}x{height}/")
        
        # prepare data for train
        mean = np.mean(data_base.output_data[:n], axis=0)
        std = np.std(data_base.output_data[:n], axis=0)

        x_train, x_val, y_train, y_val = train_test_split(
            data_base.input_data[:n], 
            (data_base.output_data[:n] - mean)/std, 
            test_size=0.3, 
            random_state=42
            )


        start_time = time.time()
        history = model.fit(
            x_train,
            y_train,
            epochs=epoch,
            batch_size=32,
            validation_data=(x_val, y_val),
            verbose=2  
        )
        end_time = time.time()

        training_hitory = {
            'mean': list(mean),
            'std': list(std),
            'loss': history.history['loss'],
            'mae': history.history['mae'],
            'val_loss': history.history['val_loss'],
            'val_mae': history.history['val_mae'],
            'time': end_time-start_time
            }

        
        
        with open(f"{width}x{height}/histories.json", 'r') as f:
            histories = json.load(f)
        
        model_name = f'model_{str(n).zfill(5)}.h5'
        histories[model_name] = training_hitory
        model.save(f"{width}x{height}/models/"+ model_name)
        
        # extend the recent histories with the new one
        with open(f"{width}x{height}/histories.json", 'w') as f:
            json.dump(histories,f)
          
def test_on_self_benchmark(size,start,end):
    width, height = size
    
    # load the benchmark data set
    data_base = CrossSectionDataBase(size)
    data_base.load('benchmark',folder=f"{width}x{height}/")

    for n in range(start,end,100):
        model_name = f'model_{str(n).zfill(5)}.h5'
        print(f'{model_name} -> START')
        
        with open(f"{width}x{height}/histories.json",'r') as f:
            histories = json.load(f)
            
        model = keras.models.load_model(f"{width}x{height}/models/{model_name}")
       
        # read the mean and std to convert back the prediction
        mean = histories[model_name]['mean']
        std = histories[model_name]['std']
            
        x_test = data_base.input_data
        y_test = (data_base.output_data-mean)/std
        y_pred = model.predict(x_test)
        
        
        histories[model_name]['benchmark_mse'] = list(np.sum(   (y_test - y_pred)**2,axis=0)/data_base.num)
        histories[model_name]['benchmark_mae'] = list(np.sum(abs(y_test - y_pred)   ,axis=0)/data_base.num)
        
        with open(f"{width}x{height}/histories.json", 'w') as f:
            json.dump(histories,f)


def test_on_common_benchmark(size,start,end,benchmark_set:CrossSectionDataBase):
    width, height = size
    resized_set = benchmark_set.resize_I(size)

    for n in range(start,end,100):
        model_name = f'model_{str(n).zfill(5)}.h5'
        print(f'{model_name} -> START')
        
        with open(f"{width}x{height}/histories.json",'r') as f:
            histories = json.load(f)
            
        model = keras.models.load_model(f"{width}x{height}/models/{model_name}")
       
        # read the mean and std to convert back the prediction
        mean = histories[model_name]['mean']
        std = histories[model_name]['std']
            
        x_test = resized_set.input_data
        y_test = (resized_set.output_data-mean)/std
        y_pred = model.predict(x_test)
        
        
        histories[model_name]['benchmark_mse'] = list(np.sum(   (y_test - y_pred)**2,axis=0)/resized_set.num)
        histories[model_name]['benchmark_mae'] = list(np.sum(abs(y_test - y_pred)   ,axis=0)/resized_set.num)
        
        with open(f"{width}x{height}/histories.json", 'w') as f:
            json.dump(histories,f)
            
if __name__ == "__main__":

    benchmark_set = CrossSectionDataBase((600,600))
    benchmark_set.generate(5e3,['rectangle','ellipse','bezier','bezier'])
    print('Benchmark set ready!!!')

    sizes = [(50,50),(100,100),(150,150),(200,200),(250,250),(300,300)]

    for size in sizes:
        test_on_common_benchmark(size=size,start=500,end=4000,benchmark_set=benchmark_set)
        print(f'Size={size} ready!')