import json
import time
import keras
from sklearn.model_selection import train_test_split
import numpy as np
import os
from data_generator import CrossSectionDataBase

def train_model(epochs,size,data_set:CrossSectionDataBase):
    
    resized_data_set = data_set.resize_I(size)
        
    w,h = size
        
    model = keras.Sequential([
                keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(w, h, 1)),
                keras.layers.MaxPooling2D((2, 2)),
                keras.layers.Flatten(),
                keras.layers.Dense(128, activation='relu'),
                keras.layers.Dense(4)  # 4 output neurons for the four float numbers
    ])
        
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
        
    mean = np.mean(resized_data_set.output_data, axis=0)
    std = np.std(resized_data_set.output_data, axis=0)
        
    x_train, x_val, y_train, y_val = train_test_split(
            resized_data_set.input_data, 
            (resized_data_set.output_data - mean)/std, 
            test_size=0.3, 
            random_state=42
    )
    
    start_time = time.time()
    history = model.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=32,
        validation_data=(x_val, y_val),
        verbose=2  
    )
    
    end_time = time.time()

    result = {
        'mean': list(mean),
        'std': list(std),
        'train':{
            'epochs': float(epochs),
            'loss': history.history['loss'],
            'mae': history.history['mae'],
            'val_loss': history.history['val_loss'],
            'val_mae': history.history['val_mae'],
            'time': end_time-start_time
        }
    }

    if not os.path.exists('models'):
        os.mkdir("models")
        
    if os.path.exists('results.json'):
        with open("results.json", 'r') as f:
            results = json.load(f)
    else:
        results = {}

    results[f'{w}x{h}']=result

    model.save(f'models/{w}x{h}.h5')
        
    with open("results.json", 'w') as f:
        json.dump(results,f)


  
def test_model(size,data_set:CrossSectionDataBase):
    resized_data_set = data_set.resize_I(size)
        
    w,h = size
    
    with open("results.json",'r') as f:
        results = json.load(f)
    
    model = keras.models.load_model(f"models/{w}x{h}.h5")
       
    # read the mean and std to convert back the prediction
    mean = np.array(results[f"{w}x{h}"]['mean'])
    std =  np.array(results[f"{w}x{h}"]['std'])
            
    x_test = resized_data_set.input_data
    y_test = (resized_data_set.output_data-mean)/std
    y_pred = model.predict(x_test)
    
    results[f'{w}x{h}']['test'] = {
        'mse' : list(np.sum(   (y_test - y_pred)**2,axis=0)/resized_data_set.num),
        'mae' : list(np.sum(abs(y_test - y_pred)   ,axis=0)/resized_data_set.num),
        }
        
        
    with open("results.json", 'w') as f:
        json.dump(results,f)
    
    
    
            
if __name__ == "__main__":

    training_set = CrossSectionDataBase((300,300))
    training_set.generate_input(1e2,['rectangle','ellipse','bezier','bezier'])
    training_set.calculate_output(numint=False)
    training_set.save('train')
    print('Training set ready!!!')
    
    benchmark_set = CrossSectionDataBase((300,300))
    benchmark_set.generate_input(1e2,['rectangle','ellipse','bezier','bezier'])
    benchmark_set.calculate_output(numint=False)
    training_set.save('benchmark')
    print('Benchmark set ready!!!')

    
    
    sizes = [(50,50),(100,100),(150,150),(200,200),(250,250),(300,300)]
    sizes = [(50,50),(100,100)]
    
    for size in sizes:
        train_model(epochs=2,size=size,data_set=training_set)
        print(f'{size} train ready!')
        
        test_model(size=size,data_set=benchmark_set)
        print(f'{size} test ready!')