from data_generator import CrossSectionDataBase
import matplotlib.pyplot as plt
import numpy as np
import json


def distribution_variing(nums,different_shapes):
    _, ax = plt.subplots(4,4,figsize=(10,10),dpi=150,subplot_kw={'xticks': [], 'yticks': []})

    j=0
    for num in nums:
        i=0
        for shapes in different_shapes:
            data_base = CrossSectionDataBase((150,150))
            data_base.generate(num=num,shapes=shapes)
            ax[j,i].imshow(data_base.get_input_distribution(),interpolation=None)
            ax[j,0].set_ylabel(f'#{num}',fontsize=18)
            ax[0,i].set_title([shape[0] for shape in shapes],fontsize=18)
            i+=1
        j+=1

    plt.tight_layout()
    plt.show()
    
    
def save_num_conv_json(start,end,num):
    
    data_base = CrossSectionDataBase((end,end))
    
    data_base.generate(num=num,shapes=['rectangle','ellipse','bezier','bezier'],numint=True)


    # calc

    sizes = []
    numint_plot = []
    semidiscr_plot = []


    for i in np.logspace(np.log2(start), np.log2(end), base=2,num=30):
        new_size = (int(i),int(i))
        sizes.append(int(i))
        
        resized_data_base = data_base.resize_IO(new_size=new_size,numint=True)
        o_avg = resized_data_base.output_avg
        numint_plot.append(o_avg)
        
        resized_data_base = data_base.resize_IO(new_size=new_size,numint=False)
        o_avg = resized_data_base.output_avg
        semidiscr_plot.append(o_avg)

        print(f'{new_size} ready!')
        
    numint_plot = np.array(numint_plot) 
    semidiscr_plot = np.array(semidiscr_plot) 
    sizes = np.array(sizes)

    with open('results.json','r') as f:
        result = json.load(f)
        
    result[str(num)] = {
        'sizes': [float(size) for size in list(sizes)],
        'numint':{
            'I_x' : list(numint_plot[:,0]),
            'I_y' : list(numint_plot[:,1]),
            'I_xy': list(numint_plot[:,2]),
            'A'   : list(numint_plot[:,3]),
        },
        'semidiscr':{
            'I_x' : list(semidiscr_plot[:,0]),
            'I_y' : list(semidiscr_plot[:,1]),
            'I_xy': list(semidiscr_plot[:,2]),
            'A'   : list(semidiscr_plot[:,3]),
        }
    }

    with open('results.json','w') as f:
        json.dump(result,f)


def plot_two_num_conv(num1,num2):
    with open('results.json','r') as f:
        result = json.load(f)
    
    #plot
    fig, ax = plt.subplots(4,2,figsize=(14,12),dpi=100,sharex=True)
    i=0
    sizes = np.array(result[str(num1)]["sizes"])
    ax[0,0].set_title(f'#{num1}',fontsize=32)
    for value in ['I_x','I_y','I_xy','A']:

        numint_plot     = np.array(result[str(num1)]["numint"][value])
        semidiscr_plot  = np.array(result[str(num1)]["semidiscr"][value])
        
        ax[i,0].plot(sizes,numint_plot   ,'.',color='blue',label='semi. discr.')
        ax[i,0].plot(sizes,semidiscr_plot,'.',color='red',label='num. int.')
        ax[i,0].grid(True)
        ax[i,0].legend(loc = 'lower right')
        ax[i,0].set_ylabel(value,fontsize=16)
        ax[i,0].set_xticks([0,100,200,300,400,500,600,700,800,900,1000,1100])
        i+=1
        
    ax[3,0].set_xlabel('image size',fontsize=16)
    
    i=0
    sizes = np.array(result[str(num2)]["sizes"])
    ax[0,1].set_title(f'#{num2}',fontsize=32)
    for value in ['I_x','I_y','I_xy','A']:

        numint_plot     = np.array(result[str(num2)]["numint"][value])
        semidiscr_plot  = np.array(result[str(num2)]["semidiscr"][value])
        
        ax[i,1].plot(sizes,numint_plot   ,'.',color='blue',label='semi. discr.')
        ax[i,1].plot(sizes,semidiscr_plot,'.',color='red',label='num. int.')
        ax[i,1].grid(True)
        ax[i,1].legend(loc = 'lower right')
        # ax[i,1].set_ylabel(value,fontsize=16)
        ax[i,1].set_xticks([0,100,200,300,400,500,600,700,800,900,1000,1100])
        i+=1
        
    ax[3,1].set_xlabel('image size',fontsize=16)
    plt.tight_layout()
    plt.show()
    
    

def plot_num_conv(num):
    with open('results.json','r') as f:
        result = json.load(f)
    
    sizes = np.array(result[str(num)]["sizes"])
  
    #plot
    fig, ax = plt.subplots(4,1,figsize=(8,12),dpi=100,sharex=True)
    i=0
    for value in ['I_x','I_y','I_xy','A']:

        numint_plot     = np.array(result[str(num)]["numint"][value])
        semidiscr_plot  = np.array(result[str(num)]["semidiscr"][value])
        
        ax[i].plot(sizes,numint_plot   ,'.',color='blue',label='semi. discr.')
        ax[i].plot(sizes,semidiscr_plot,'.',color='red',label='num. int.')
        ax[i].grid(True)
        ax[i].legend(loc = 'lower right')
        ax[i].set_ylabel(value,fontsize=18)
        ax[i].set_xticks([0,100,200,300,400,500,600,700,800,900,1000,1100])
        i+=1
        
    ax[3].set_xlabel('pixel',fontsize=18)

    plt.tight_layout()
    plt.show()