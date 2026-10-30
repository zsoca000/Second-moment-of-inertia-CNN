from data_generator import CrossSectionDataBase
import numpy as np
import time
import json
import keras

def eval_ref(data_set: CrossSectionDataBase,num=500):
    
    start = time.time()
    data_set.generate_input(num=num,shapes=['rectangle','ellipse','bezier','bezier'])
    data_set.calculate_output(numint=False)
    end = time.time()
    
    with open("results.json",'r') as f:
        log = json.load(f)
    
    if str(1000) not in log['numeric']:
        log['numeric'][str(1000)] = {
                'time': end - start,
                'I_x': list(data_set.output_data[:,0]),
                'I_y': list(data_set.output_data[:,1]),
                'I_xy': list(data_set.output_data[:,2]),
                'A': list(data_set.output_data[:,3])     
            }
    else:
        log['numeric'][str(1000)]['time']+=end - start
        log['numeric'][str(1000)]['I_x'] +=list(data_set.output_data[:,0])
        log['numeric'][str(1000)]['I_y'] +=list(data_set.output_data[:,1])
        log['numeric'][str(1000)]['I_xy']+=list(data_set.output_data[:,2])
        log['numeric'][str(1000)]['A']   +=list(data_set.output_data[:,3])

    with open("results.json",'w') as f:
        json.dump(log,f)
   
def eval_numeric(data_set: CrossSectionDataBase):
    
    for i in np.logspace(np.log10(50), np.log10(1000), num=20):
        if i == 1000:
            break
        
        w,h = int(i),int(i)
        
        with open("results.json",'r') as f:
            log = json.load(f)

        start = time.time()
        resized_set = data_set.resize_IO(new_size=(w,h),numint=True)
        end = time.time()
        
        if str(w) not in log['numeric']:
        
            log['numeric'][str(w)] = {
                'time': end - start,
                'I_x': list(resized_set.output_data[:,0]),
                'I_y': list(resized_set.output_data[:,1]),
                'I_xy': list(resized_set.output_data[:,2]),
                'A': list(resized_set.output_data[:,3])     
            }
        else:
            log['numeric'][str(w)]['time']+=end - start
            log['numeric'][str(w)]['I_x'] +=list(data_set.output_data[:,0])
            log['numeric'][str(w)]['I_y'] +=list(data_set.output_data[:,1])
            log['numeric'][str(w)]['I_xy']+=list(data_set.output_data[:,2])
            log['numeric'][str(w)]['A']   +=list(data_set.output_data[:,3])
            
        with open("results.json",'w') as f:
            json.dump(log,f)

        print(f'({w}x{h}) numeric solution ready!')
        
def eval_data_driven(data_set: CrossSectionDataBase,sizes):
    
    for w,h in zip(sizes,sizes):
        
        with open("results.json",'r') as f:
            log = json.load(f)
        
        resized_set = data_set.resize_I(new_size=(w,h))
        
        model = keras.models.load_model(f"models/{w}x{h}.h5")
        
        start = time.time()
        y_pred = model.predict(resized_set.input_data)
        end = time.time()
        
        with open("models/results.json",'r') as f:
            results = json.load(f)
            mean = np.array(results[f"{w}x{h}"]['mean'])
            std =  np.array(results[f"{w}x{h}"]['std'])

            output_data = y_pred*std + mean
        
        if str(w) not in log['data-driven']:
            log['data-driven'][str(w)] = {
                'time': end - start,
                'I_x': list(output_data[:,0]),
                'I_y': list(output_data[:,1]),
                'I_xy': list(output_data[:,2]),
                'A': list(output_data[:,3])     
            }
        else:
            log['data-driven'][str(w)]['time']+=end - start
            log['data-driven'][str(w)]['I_x'] +=list(output_data[:,0])
            log['data-driven'][str(w)]['I_y'] +=list(output_data[:,1])
            log['data-driven'][str(w)]['I_xy']+=list(output_data[:,2])
            log['data-driven'][str(w)]['A']   +=list(output_data[:,3])

        with open("results.json",'w') as f:
            json.dump(log,f)

        print(f'({w}x{h}) data-driven solution ready!')

if __name__ == "__main__":
    
    data_set = CrossSectionDataBase((1000,1000),L_pix = 8/10) # defines the L_pix to the trained moodels
    
    data_set.load('ref_2')
    eval_data_driven(data_set,sizes=[125,175])
    
    
    
    # eval_ref(data_set)
    

    

       
          
    
    
 


        
        
        
        
        
        
        
    
    