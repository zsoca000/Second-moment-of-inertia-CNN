import numpy as np
import json
import matplotlib.pyplot as plt
from data_generator import CrossSectionDataBase


class ModelEvaluator:

    def benchmark_eval(self,size):
        width, height = size
        
        with open(f'results\{width}x{height}.json','r') as f:
            histories = json.load(f)

        num = []
        benchmark_mse = []
        benchmark_mae = []

        for model_name in histories.keys(): # os.listdir(os.getcwd()+'\\'+folder_name+'\\models\\'):
            
            num.append(int(model_name.replace('.h5','').replace('model_',''))) # size of training data set
            
            benchmark_mse.append(histories[model_name]['benchmark_mse'])
            benchmark_mae.append(histories[model_name]['benchmark_mae'])

        benchmark_mse=np.array(benchmark_mse)
        benchmark_mae=np.array(benchmark_mae)

        _, ax = plt.subplots(2,1,figsize=(8,8),dpi=100,sharex=True)  

        ax[0].plot(num,benchmark_mse[:,0],color='#7D58FF',label='I_x',marker='o')
        ax[0].plot(num,benchmark_mse[:,1],color='#000E84',label='I_y',marker='o')
        ax[0].plot(num,benchmark_mse[:,2],color='#0C8400',label='I_xy',marker='o')
        ax[0].plot(num,benchmark_mse[:,3],color='#FF0000',label='A',marker='o')
        ax[0].set_ylabel('Benchmark MSE')
        ax[0].legend()

        ax[1].plot(num,benchmark_mae[:,0],color='#7D58FF',label='I_x',marker='o')
        ax[1].plot(num,benchmark_mae[:,1],color='#000E84',label='I_y',marker='o')
        ax[1].plot(num,benchmark_mae[:,2],color='#0C8400',label='I_xy',marker='o')
        ax[1].plot(num,benchmark_mae[:,3],color='#FF0000',label='A',marker='o')
        ax[1].set_ylabel('Benchmark MAE')
        ax[1].set_xlabel('# of training data')
        ax[1].legend()


    def train_eval(self,size): 
        width, height = size
        with open(f'results\{width}x{height}.json','r') as f:
            histories = json.load(f)
            
            
        _, ax = plt.subplots(4,2,figsize=(16,12),dpi=100,sharex=True)   

        for model_name in histories.keys(): # os.listdir(os.getcwd()+'\\'+folder_name+'\\models\\'):

            loss = histories[model_name]['loss']
            mae = histories[model_name]['mae']
            val_loss = histories[model_name]['val_loss']
            val_mae = histories[model_name]['val_mae']   
            num = int(model_name.replace('.h5','').replace('model_',''))
            epoch = len(histories[model_name]['loss'])
            
            ax[0,0].plot(np.arange(epoch),loss,linewidth=1, label=num)
            ax[1,0].plot(np.arange(epoch),mae,linewidth=1, label=num)
            ax[2,0].plot(np.arange(epoch),val_loss,linewidth=1, label=num)
            ax[3,0].plot(np.arange(epoch),val_mae,linewidth=1, label=num)
            
            ax[0,0].scatter(epoch-1,loss[epoch-1],marker='o')
            ax[1,0].scatter(epoch-1,mae[epoch-1],marker='o')
            ax[2,0].scatter(epoch-1,val_loss[epoch-1],marker='o')
            ax[3,0].scatter(epoch-1,val_mae[epoch-1],marker='o')
            
            ax[0,1].plot(np.arange(5,epoch),loss[5:],linewidth=1, label=num)
            ax[1,1].plot(np.arange(5,epoch),mae[5:],linewidth=1, label=num)
            ax[2,1].plot(np.arange(5,epoch),val_loss[5:],linewidth=1, label=num)
            ax[3,1].plot(np.arange(5,epoch),val_mae[5:],linewidth=1, label=num)
            
            ax[0,1].scatter(epoch-1,loss[epoch-1],marker='o')
            ax[1,1].scatter(epoch-1,mae[epoch-1],marker='o')
            ax[2,1].scatter(epoch-1,val_loss[epoch-1],marker='o')
            ax[3,1].scatter(epoch-1,val_mae[epoch-1],marker='o')

        ax[3,0].set_xlabel('# of epochs')

        ax[0,0].set_ylabel('loss')
        ax[1,0].set_ylabel('mae')
        ax[2,0].set_ylabel('val_loss')
        ax[3,0].set_ylabel('val_mae')

        ax[3,1].set_xlabel('# of epochs')

        ax[0,1].set_ylabel('loss')
        ax[1,1].set_ylabel('mae')
        ax[2,1].set_ylabel('val_loss')
        ax[3,1].set_ylabel('val_mae')
        

    def data_set_distribution(self,size,name='train'):    

        data_base = CrossSectionDataBase(size)
        data_base.load(name)

        _, ax = plt.subplots(5,5,figsize=(10,10),dpi=100,subplot_kw={'xticks': [], 'yticks': []})
        n = 500
        for i in range(5):
            for j in range(5):
                ax[i,j].imshow(np.sum(data_base.input_data[:n],axis=0),interpolation=None)
                ax[i,j].set_title(f'#{n}')
                n+=100
                
        plt.tight_layout()
        plt.show()
        
        
    def compare_benchmark_evals(self,size_list):
        
        _, ax = plt.subplots(2,1,figsize=(8,8),dpi=100,sharex=True)
        
        for size in size_list:
            
            width, height = size
            
            with open(f'results\{width}x{height}.json','r') as f:
                histories = json.load(f)

            num = []
            benchmark_mse = []
            benchmark_mae = []
            
            for model_name in histories.keys(): # os.listdir(os.getcwd()+'\\'+folder_name+'\\models\\'):
            
                num.append(int(model_name.replace('.h5','').replace('model_',''))) # size of training data set
                
                benchmark_mse.append(histories[model_name]['benchmark_mse'])
                benchmark_mae.append(histories[model_name]['benchmark_mae'])

            benchmark_mse=np.array(benchmark_mse)
            benchmark_mae=np.array(benchmark_mae)

            

            np.average(benchmark_mse,axis=1)
            
            ax[0].plot(num,np.average(benchmark_mse,axis=1),label=f'{width}x{height}',marker='o')
            ax[1].plot(num,np.average(benchmark_mae,axis=1),label=f'{width}x{height}',marker='o')
        
        ax[0].set_title("Benchmark evaluation")
        ax[0].set_ylabel('MSE')
        ax[0].legend(loc='upper right')
        ax[1].set_ylabel('MAE')
        ax[1].set_xlabel('# of training data')
        ax[1].legend(loc='upper right')
        
import os

if __name__ == "__main__":
    
    for size in [(100,100),(150,150),(200,200),(300,300)]:
        
        width, height = size
        folder_name = f'{width}x{height}'
        
        os.rename(f'result/{width}x{height}', f'result/{width}x{height}.json')