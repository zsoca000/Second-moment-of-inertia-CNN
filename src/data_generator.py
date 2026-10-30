import numpy as np
from PIL import Image, ImageDraw
import h5py
import random
from scipy.special import binom
import os
from scipy.ndimage import zoom

bernstein = lambda n, k, t: binom(n,k)* t**k * (1.-t)**(n-k)

def bezier(points, num=200):
    N = len(points)
    t = np.linspace(0, 1, num=num)
    curve = np.zeros((num, 2))
    for i in range(N):
        curve += np.outer(bernstein(N - 1, i, t), points[i])
    return curve

class Segment():
    def __init__(self, p1, p2, angle1, angle2, **kw):
        self.p1 = p1; self.p2 = p2
        self.angle1 = angle1; self.angle2 = angle2
        self.numpoints = kw.get("numpoints", 100)
        r = kw.get("r", 0.3)
        d = np.sqrt(np.sum((self.p2-self.p1)**2))
        self.r = r*d
        self.p = np.zeros((4,2))
        self.p[0,:] = self.p1[:]
        self.p[3,:] = self.p2[:]
        self.calc_intermediate_points(self.r)

    def calc_intermediate_points(self,r):
        self.p[1,:] = self.p1 + np.array([self.r*np.cos(self.angle1),
                                    self.r*np.sin(self.angle1)])
        self.p[2,:] = self.p2 + np.array([self.r*np.cos(self.angle2+np.pi),
                                    self.r*np.sin(self.angle2+np.pi)])
        self.curve = bezier(self.p,self.numpoints)

def get_curve(points, **kw):
    segments = []
    for i in range(len(points)-1):
        seg = Segment(points[i,:2], points[i+1,:2], points[i,2],points[i+1,2],**kw)
        segments.append(seg)
    curve = np.concatenate([s.curve for s in segments])
    return segments, curve

def ccw_sort(p):
    d = p-np.mean(p,axis=0)
    s = np.arctan2(d[:,0], d[:,1])
    return p[np.argsort(s),:]

def get_bezier_curve(a, rad=0.2, edgy=0):
    """ given an array of points *a*, create a curve through
    those points. 
    *rad* is a number between 0 and 1 to steer the distance of
          control points.
    *edgy* is a parameter which controls how "edgy" the curve is,
           edgy=0 is smoothest."""
    p = np.arctan(edgy)/np.pi+.5
    a = ccw_sort(a)
    a = np.append(a, np.atleast_2d(a[0,:]), axis=0)
    d = np.diff(a, axis=0)
    ang = np.arctan2(d[:,1],d[:,0])
    f = lambda ang : (ang>=0)*ang + (ang<0)*(ang+2*np.pi)
    ang = f(ang)
    ang1 = ang
    ang2 = np.roll(ang,1)
    ang = p*ang1 + (1-p)*ang2 + (np.abs(ang2-ang1) > np.pi )*np.pi
    ang = np.append(ang, [ang[0]])
    a = np.append(a, np.atleast_2d(ang).T, axis=1)
    s, c = get_curve(a, r=rad, method="var")
    x,y = c.T
    return x,y, a

def get_random_points(n=5, scale=0.8, mindst=None, rec=0):
    """ create n random points in the unit square, which are *mindst*
    apart, then scale them."""
    mindst = mindst or .7/n
    a = np.random.rand(n,2)
    d = np.sqrt(np.sum(np.diff(ccw_sort(a), axis=0), axis=1)**2)
    if np.all(d >= mindst) or rec>=200:
        return a*scale
    else:
        return get_random_points(n=n, scale=scale, mindst=mindst, rec=rec+1)
    
def bezier(points, num=200):
    N = len(points)
    t = np.linspace(0, 1, num=num)
    curve = np.zeros((num, 2))
    for i in range(N):
        curve += np.outer(bernstein(N - 1, i, t), points[i])
    return curve

class CrossSectionDataBase:
    
    num : int
    size: tuple
    input_data : np.array
    output_data : np.array
    shapes: list
    name: str
    L_pix: int
    
    def __init__(self,size,L_pix=1,name=None):
        self.size = size
        self.L_pix = L_pix
        self.name = name
    
    def generate_input(self,num=None,shapes=None,img_list=None):
        w, h = self.size
        
        # Random generation
        if num is not None and shapes is not None:
            
            self.num = int(num)
            self.shapes = shapes
            self.input_data  = np.empty((self.num,int(w),int(h),1))
            self.output_data = np.empty((self.num,4))
            
            for i in range(self.num):
                img = self._generate_random_shape(
                    width=int(self.size[0]), 
                    height=int(self.size[1]), 
                    shapes=shapes
                )
                
                self.input_data[i] = self._img2arr(img)
                print(f'{i}. input is ready')
        
        # Manual data
        elif img_list is not None:
            
            self.num = len(img_list)
            self.shapes = None
            self.input_data  = np.empty((self.num,int(w),int(h),1))
            self.output_data = np.empty((self.num,4))
            
            for i,img in enumerate(img_list):
                self.input_data[i] = self._img2arr(img)
                print(f'{i}. input is ready')

    def calculate_output(self,numint=True):
        for i in range(self.num):
            if numint:  
                self.output_data[i] = self._inertia_values_numint(self.input_data[i])
            else:
                self.output_data[i] = self._inertia_values_semidiscr(self.input_data[i])
            
            print(f'{i}. output is ready')
    
    def generate(self,num,shapes,numint=True):
        self.generate_input(num=num,shapes=shapes)
        self.calculate_output(numint=numint)
    
    def save(self,name,folder=''):
        
        if not os.path.exists(folder + 'data'):
            os.makedirs(folder + 'data')

        file_path = folder + f"data/{name}.h5"
        
        if not os.path.exists(file_path):
            with h5py.File(file_path, 'w') as hf:
                hf.create_dataset("shapes",data=self.shapes)
                hf.create_dataset("input_data", data=self.input_data)
                hf.create_dataset("output_data", data=self.output_data)
        else:
            print(f'{file_path} existed before')
    
    def load(self,name,folder=''):
        
        self.add_name(name)
        
        file_path = folder + f"data/{name}.h5"
        
        with h5py.File(file_path, 'r') as hf:
            self.shapes = list(hf['shapes'])
            self.input_data = hf['input_data'][()]
            self.output_data = hf['output_data'][()]
            self.num = self.input_data.shape[0]
            self.size = (self.input_data.shape[1],self.input_data.shape[2])
            
    def add_name(self,name):
        self.name = name
    
    @property
    def input_distribution(self):
        return np.sum(self.input_data,axis=0)
    
    @property
    def output_avg(self):
        return np.average(self.output_data,axis=0)
    
    def resize_I(self,new_size):
        width, height = self.size
        new_width, new_height = new_size
        
        reduce_factor = min(new_width/width,new_height/height)
        
        ret = CrossSectionDataBase(new_size,L_pix=self.L_pix/reduce_factor)
        
        arrays = []
        
        for i in range(self.num):
            arr = np.round(zoom(self.input_data[i,:,:,0], zoom=reduce_factor, order=0)).astype(int)
            arrays.append(arr.reshape(arr.shape[0],arr.shape[1],1))
            
            print(f'\t - {i}/{self.num}. I resized! ')
            
        ret.input_data  = np.array(arrays)
        ret.output_data = self.output_data
        ret.shapes = self.shapes
        ret.name = self.name
        ret.num = self.num
        
        return ret
    
    def resize_IO(self,new_size,numint=True):
        width, height = self.size
        new_width, new_height = new_size
        
        reduce_factor = min(new_width/width,new_height/height)
        
        ret = CrossSectionDataBase(new_size,L_pix=self.L_pix/reduce_factor)

        inertias = []
        arrays = []
        
        for i in range(self.num):
            
            arr = np.round(zoom(self.input_data[i,:,:,0], zoom=reduce_factor, order=0)).astype(int)
            arr = arr.reshape(arr.shape[0],arr.shape[1],1)
            
            if numint:
                inertia = ret._inertia_values_numint(arr)
            else:
                inertia = ret._inertia_values_semidiscr(arr)
            arrays.append(arr)
            inertias.append(inertia)
            
            print(f'\t - {i}/{self.num}. IO resized! ')

        ret.output_data = np.array(inertias)
        ret.input_data = np.array(arrays)
        ret.shapes = self.shapes
        ret.name = self.name
        ret.num = self.num
        
        return ret
    
    def _generate_random_shape(self,width,height,shapes):
        image = Image.new("L", (width, height), 255)  # Create a white image
        draw = ImageDraw.Draw(image)

        # Generate random parameters for the shape
        shape_type = random.choice(shapes)
        color = 0  # Black color

        if shape_type == "rectangle":
            left = random.randint(0, width // 2)
            top = random.randint(0, height // 2)
            right = random.randint(width // 2, width)
            bottom = random.randint(height // 2, height)
            draw.rectangle([left, top, right, bottom], outline=color, fill=color)
        
        elif shape_type == "ellipse":
            left = random.randint(0, width // 2)
            top = random.randint(0, height // 2)
            right = random.randint(width // 2, width)
            bottom = random.randint(height // 2, height)
            draw.ellipse([left, top, right, bottom], outline=color, fill=color)
        
        elif shape_type == "bezier":
            rad = random.uniform(0, 0.5)
            edgy = random.uniform(0, 1)
            n = random.randint(3,10)
            scale = random.randint(0.2*width,0.8*width)

            points = get_random_points(n=n, scale=scale)
            x,y, _ = get_bezier_curve(points, rad=rad, edgy=edgy)
            x_c = random.randint(0,width - scale)
            y_c = random.randint(0,height - scale)

            verticies = list(zip(x + x_c,y + y_c))

            draw.polygon(verticies, outline="black", fill="black")

        return image 
    
    def _img2arr(slef,img):
        img = img.convert("L") # color -> black & white
        img = np.array(img) # black & white -> array
        img = np.where(img!=0,0,1) # 1-255 -> 0 and 0 -> 1
        arr = np.resize(img,(img.shape[0],img.shape[1],1)) # (w,h) -> (w,h,1)
        return arr    
    
    def _inertia_values_numint(self,arr):
        
        rows,cols,_ = arr.shape
        I_x,I_y,I_xy,A  = 0,0,0,0
        A_pix = self.L_pix*self.L_pix # area of a pixel
        
        for r in range(rows):
            for c in range(cols):
                x = (c - cols/2 + 1/2) * self.L_pix
                y = (r - rows/2 + 1/2) * self.L_pix
                if arr[r,c] != 0:  # we only add, if cs[r,c] = 1
                    I_x  += y*y*A_pix
                    I_y  += x*x*A_pix
                    I_xy += x*y*A_pix
                    A    +=     A_pix
    
        return np.array([I_x,I_y,I_xy,A])
    
    def _inertia_values_semidiscr(self,arr):
        rows,cols,_ = arr.shape
        I_x,I_y,I_xy,A  = 0,0,0,0

        for r in range(rows):
            for c in range(cols):
                x1 = (c - cols/2) * self.L_pix
                y1 = (r - rows/2) * self.L_pix
                x2 = x1 + self.L_pix
                y2 = y1 + self.L_pix
                
                if arr[r,c,0] != 0:  # we only add, if cs[r,c] = 1
                    I_x  += self.L_pix*self.L_pix/3 * (y1*y1+y1*y2+y2*y2)
                    I_y  += self.L_pix*self.L_pix/3 * (x1*x1+x1*x2+x2*x2)
                    I_xy += self.L_pix*self.L_pix/4 * (x2+x1)*(y2+y1)
                    A    += self.L_pix*self.L_pix
        
        return np.array([I_x,I_y,I_xy,A])
    

if __name__ == "__main__":
    
    data_base = CrossSectionDataBase((200,200))
    
    data_base.generate_input(1,['rectangle'])
    
    
    