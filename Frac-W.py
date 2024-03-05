import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mlt
#from mpl_toolkits.mplot3d import Axes3D 
import mpl_toolkits.mplot3d.art3d as art3d
#from welly import Well
#import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse
pd.options.mode.chained_assignment = None


csv_dev = "D:\Miguel\Houston\Python\MD TVD - Bear Claw 04H.csv"
dfd = pd.read_csv(csv_dev) #, usecols=['x', 'y', 'z', 'MD', 'TVD']

csv_com = "D:\Miguel\Houston\Python\Bear Claw Completion 4.csv"
dfc = pd.read_csv(csv_com) #, usecols=['Well', 'Stage #', 'Bottom Perfz', 'Top Perf', 'XF' , 'HT']


#print(df)
%matplotlib qt

 #Step 2: Create a 3D line plot
# fig = plt.figure(figsize=(8, 6))
# ax = fig.add_subplot(111, projection='3d')

# Extract x, y, z coordinates
# x_coords = dfd['X']
# y_coords = dfd['Y']
# z_coords = dfd['TVD']

# Plot the 3D line
# ax.plot(x_coords, y_coords, z_coords, label='3D Line', color='b')

# Customize plot (add labels, title, etc.)
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('TVD')
# ax.set_title('3D Line Plot')
# ax.legend()
# ax.invert_zaxis()
# Show the plot
# plt.show()

def pathpatch_2d_to_3d(pathpatch, z = 0, normal = 'z'):
    """
    Transforms a 2D Patch to a 3D patch using the given normal vector.

    The patch is projected into they XY plane, rotated about the origin
    and finally translated by z.
    """
    if type(normal) is str: #Translate strings to normal vectors
        index = "xyz".index(normal)
        normal = np.roll((1,0,0), index)

    path = pathpatch.get_path() #Get the path and the associated transform
    trans = pathpatch.get_patch_transform()

    path = trans.transform_path(path) #Apply the transform

    pathpatch.__class__ = art3d.PathPatch3D #Change the class
    pathpatch._code3d = path.codes #Copy the codes
    pathpatch._facecolor3d = pathpatch.get_facecolor #Get the face color

    verts = path.vertices #Get the vertices in 2D

    M = rotation_matrix(normal,(0, 0, 1)) #Get the rotation matrix

    pathpatch._segment3d = np.array([np.dot(M, (x, y, 0)) + (0, 0, z) for x, y in verts])



def rotation_matrix(v1,v2):
    """
    Calculates the rotation matrix that changes v1 into v2.
    """
    v1/=np.linalg.norm(v1)
    v2/=np.linalg.norm(v2)

    cos_angle=np.dot(v1,v2)
    d=np.cross(v1,v2)
    sin_angle=np.linalg.norm(d)

    if sin_angle == 0:
        M = np.identity(3) if cos_angle>0. else -np.identity(3)
    else:
        d/=sin_angle

        eye = np.eye(3)
        ddt = np.outer(d, d)
        # skew = np.array([[    0,  d[2],  -d[1]],
        #               [-d[2],     0,  d[0]],
        #               [d[1], -d[0],    0]], dtype=np.float64)
        skew = np.array([[    0,  d[2],  -d[1]],
                      [0,     0,  d[0]],
                      [0, -d[0],    0]], dtype=np.float64)
        M = ddt + cos_angle * (eye - ddt) + sin_angle * skew

    return M


def pathpatch_translate(pathpatch, delta):
    """
    Translates the 3D pathpatch by the amount delta.
    """
    pathpatch._segment3d += delta


def plot_func(df,color):
    x = dfd['X']
    y = dfd['Y']
    z = dfd['TVD']

    ax.plot(x, y, z, c=color, lw=4)

fig = plt.figure(figsize=(15,8))
ax = fig.add_subplot(111, projection='3d')

cmap = mlt.cm.jet
mappable = plt.cm.ScalarMappable(cmap=plt.cm.jet)

#Assume 250 and 2500 to me the limits of Xf
mappable.set_clim(vmin=2*dfc['XF'].min(),vmax=2*dfc['XF'].max())

#df_stage=df_stage_all.loc[df_stage_all['WELL']==m]
#df_survey =data_survey.loc[data_survey['WELL']==m]

  # Find X, Y Locations for the stages

#df_stage['X_NEW']=np.interp(df_stage['Mid_Depth'].values,df_survey['MD'].values, df_survey['X'].values)
dfc['X']=np.interp(dfc['Top Perforation'].values,dfd['MD'].values, dfd['X'].values)

#df_stage['Y_NEW']=np.interp(df_stage['Mid_Depth'].values,df_survey['MD'].values, df_survey['Y'].values)  
dfc['Y']=np.interp(dfc['Top Perforation'].values,dfd['MD'].values, dfd['Y'].values)


dfc['TVD']=np.interp(dfc['Top Perforation'].values,dfd['MD'].values, dfd['TVD'].values)

print(dfd)
print(dfc.columns.tolist())

# Create lists for stage parameters
 
tvd_list = dfc['TVD'].tolist()
x_list = dfc['X'].tolist()
y_list = dfc['Y'].tolist()
xf_list = dfc['XF'].tolist()
ht_list = dfc['HT'].tolist()

minflux = dfc['XF'].min()
maxflux = dfc['XF'].max()


for i,j,y,k,l in zip(tvd_list,x_list,y_list,xf_list,ht_list):
    
      normflux = (k - minflux) / (maxflux - minflux)
      fluxcolor = (1. - normflux, 1. - normflux, normflux) 
      p= Ellipse(xy=(0,0), width=k, height=l,edgecolor=cmap(normflux),  fc=cmap(normflux), angle=0, lw=2, alpha=0.7)

      #Project the Ellipse from 2D to 3D
      ax.add_patch(p)
  
      # Change orientation of frac to the desired Shmin direction by changing the parameter (-0.2) in the normal. 0 = perpendicular to wellbore.

      pathpatch_2d_to_3d(p, z=0, normal=(-.2, 1, 0))
      pathpatch_translate(p, (j, y, i))


#Plot well trajectories
plot_func(dfd,'b') # Well1
#plot_func(data2,'g') # Well2

ax.invert_zaxis()
ax.set_xlim(2177000, 2181500)  # Set x-axis limits
ax.set_ylim(492000, 506000)  # Set y-axis limits
ax.set_zlim(18000, 000 )  # Set z-axis limits

#Set Colorbar
clb = fig.colorbar(mappable)
clb.ax.set_title('2*Xf')
plt.tight_layout()
plt.show()
