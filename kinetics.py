import nd2reader as nd2
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import regionprops
from scipy.ndimage import label
from skimage.filters import gaussian, threshold_otsu, threshold_local, threshold_multiotsu
from skimage.morphology import *
import time
#%% Set rcParams
import matplotlib as mpl
mpl.rcParams['image.interpolation']=None
#%%Load dataset and test img
dataset=nd2.ND2Reader('Dose Kinetics.nd2')
"""
Axis 'x' size: 2048
Axis 'y' size: 2048
Axis 'c' size: 2
Axis 't' size: 49
Axis 'v' size: 60
"""
dataset.bundle_axes='yx'
dataset.default_coords['v']=0
dataset.default_coords['c']=1
dataset.iter_axes='t'

dataset.default_coords['v']=45
positive=dataset[48]
plt.imshow(positive)


dataset.default_coords['v']=55
negative=dataset[48][:512,:512]
plt.imshow(negative)

fig1, (ax2,ax1)=plt.subplots(1,2)
ax1.imshow(positive, cmap='gray')
ax1.axis('off')
ax1.set_title('Positive', size=20)
ax2.imshow(negative, cmap='gray')
ax2.axis('off')
ax2.set_title('Negative', size=20)
plt.tight_layout()

#%% Brightest picture
results=[]
for loc in range(60):
   dataset.default_coords['v']=loc
   results.append((loc,np.percentile(dataset[48], 99),np.count_nonzero(dataset[48]>17000)))
   print(loc)

test=sorted(results, key=lambda tup: tup[1])

num=[tup[0] for tup in results]
x=[tup[1] for tup in results]
y=[tup[2] for tup in results]

plt.scatter(x, y)


#%%test agg detection
ctrl_img=gaussian(positive)
thresh=threshold_otsu(white_tophat(ctrl_img, selem=disk(10)))
plt.imshow(white_tophat(ctrl_img, selem=disk(10))>thresh)
plt.show()
plt.imshow(ctrl_img)
plt.show()

neg_img=negative
plt.imshow(white_tophat(neg_img, selem=disk(10))>thresh)
plt.show()
plt.imshow(neg_img)
plt.show()

disk_size=11
thresh=threshold_otsu(white_tophat(ctrl_img, selem=disk(10)))
for img in dataset:
   img=gaussian(img)[:256,:256]
   fig, (ax1,ax2)=plt.subplots(1,2)
   ax1.imshow(img)
   ax1.axis('off')
   ax2.imshow(white_tophat(img, selem=disk(disk_size))>thresh)
   ax2.axis('off')
   plt.show()
   
for v in [44, 55]:
   dataset.default_coords['v']=v
   start=time.time()
   disk_size=11
   thresh=threshold_otsu(white_tophat(ctrl_img[:512,:512], selem=disk(10)))
   obj_count=[]
   for i, img in enumerate(dataset):
      obj_count.append([i,label(white_tophat(gaussian(img[:512,:512]), selem=disk(disk_size))>thresh)[1]])
      
   print('finished in ' + str(time.time()-start))
   
   x,y=zip(*obj_count)
   x=list(x)
   y=list(y)
   import seaborn as sns
   sns.regplot(x,y,lowess=True)

#%% Data extraction and object identification
res=[]
for v in range(dataset.sizes['v']):
   dataset.default_coords['v']=v
   start=time.time()
   disk_size=11
   thresh=threshold_otsu(white_tophat(ctrl_img, selem=disk(10)))
   objs=[]
   for i, img in enumerate(dataset):
      objects=label(white_tophat(gaussian(img), selem=disk(disk_size))>thresh)[0]
      obj_props=regionprops(objects, img)
      if len(obj_props)>0:
         obj_len=list(range(len(obj_props)))
         for t, obj in enumerate(reversed(obj_props)):
            if obj.area < 10 or obj.area> 100:
               obj_props.pop(obj_len[-(t+1)])
      if len(obj_props)>0:
         sizes,ints=zip(*[(x.area, x.mean_intensity) for x in obj_props])
         mean_size=np.mean(sizes)
         mean_int=np.mean(ints)
         num=len(sizes)
         objs.append((num, mean_size, mean_int))
      else: 
         objs.append((0, 0, 0))
      print('Finished t '+str(i+1))
   res.append(objs)
   print('Finished well '+str(v+1))
   
#%% Dataframe construction and export
import pandas as pd
well_ID=pd.read_excel('Dose Kinetics/lettered_layout.xlsx', index=None, header=None)
well_ID=np.array(well_ID).ravel()
treatments=pd.read_excel('Dose Kinetics/plateLayout.xlsx', index=None, header=None)
treatments=np.array(treatments).ravel()
sample_order=pd.read_excel('Dose Kinetics/acquisition_order.xlsx', index=None, header=None)
sample_order=np.array(sample_order).ravel()

df=pd.DataFrame([pd.Series(sample_order),pd.Series(well_ID),pd.Series(treatments)]).T
df=df.sort_values([0])
df.columns=['imaging_order','well_ID','treatment']
df = df[df.imaging_order != 0]
df=df.reset_index()
df=df.drop(columns='index')

for i, name in enumerate(['agg_count','mean_size','mean_int']):
   plate=[]
   for well in res:
      plate.append([timepoint[i] for timepoint in well])
   
   df_plate=pd.DataFrame(plate)
   df2=pd.concat([df,df_plate], axis=1)
   df2=df2.drop(columns='imaging_order')
   df2.to_excel("../"+name+".xlsx", index=None)

#%% plotting
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

df=pd.read_excel("../agg_count.xlsx")
df=df.set_index('well_ID')
mean_count=df.groupby('treatment').mean().T
df=pd.read_excel("../mean_size.xlsx")
df=df.set_index('well_ID')
mean_size=df.groupby('treatment').mean().T
df=pd.read_excel("../mean_int.xlsx")
df=df.set_index('well_ID')
mean_int=df.groupby('treatment').mean().T


controls=['Neg_Ctrl','DMSO','Positive']
high_conc=controls+[name for name in mean_count.columns if '10' in name]
low_conc=controls+[name for name in mean_count.columns if '1u' in name]

df=mean_count[high_conc]
df.plot()

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'Helvetica'

# set the style of the axes and the text color
plt.rcParams['axes.edgecolor']='#333F4B'
plt.rcParams['axes.linewidth']=0.8
plt.rcParams['xtick.color']='#333F4B'
plt.rcParams['ytick.color']='#333F4B'
plt.rcParams['text.color']='#333F4B'
#%% plotting kinetics
fig, ax1=plt.subplots(figsize=(7,5))
df=mean_count[high_conc]
for sample in df:
   sns.regplot(np.arange(len(df[sample])), df[sample], order=2, scatter_kws={'s':0},ax=ax1)
plt.legend(['Negative Control', 
 'DMSO',
 'Non-treateted',
 'Enzastaurin',
 'GF109203X',
 'SB203580',
 'SB202190',
 'VX-745'], bbox_to_anchor=(1.0, 0.75))
ax1.spines['top'].set_color('none')
ax1.spines['right'].set_color('none')

ax1.set_xlabel('hours',fontsize=15)
ax1.set_ylabel('Aggregates/Well',fontsize=15)
plt.tight_layout()
plt.savefig('10uM kinetic.png', dpi=300)
