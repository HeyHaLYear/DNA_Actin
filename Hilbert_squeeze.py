# Kuang Liu 20170930 fractal globule
#!/usr/bin/python

import random
import sys, os, glob
import copy as cp
import numpy as np
import scipy as sp
import datetime
from scipy.interpolate import interp1d
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as ptch
from matplotlib.patches import Ellipse
import matplotlib.lines as lne
import matplotlib.colors as mcol
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.cm as cm
from collections import defaultdict
import os.path

EMPTY=-1
N=2
radius=float(0.5/N)*0.95
lattice_const=float(1.0/N)
Max_L=128.0
elastic_constant=1.0
Maximum_repulsion=2.0
mass=1.0
boundary_constant=2.0
timestep=0.01
speed_boundary=0.02

List_L = [2,4,8,16,32,64,128]


# initialization: gridding
gridsize=2*radius


def gridding(coordinates,L,high_bound,low_bound,right_bound,left_bound):
	for k in range(N*L*L-(N-1)):
		index_x=int((coordinates[k,0]-left_bound)/gridsize)
		index_y=int((coordinates[k,1]-low_bound)/gridsize)
		#print index_x,index_y
		for i in range(3):
			if GRID[index_x,index_y,i]==EMPTY:
				GRID[index_x,index_y,i]=k
				break
	return GRID

def random_force(systemsize,mu,sigma):
	random_F=np.zeros((int(systemsize),2))
	for i in range(N*L*L-(N-1)):
		random_F[i,0]=random.gauss(mu,sigma)
		random_F[i,1]=random.gauss(mu,sigma)
	return random_F

def repulsive_force(coordinates,GRID,L,boundary_switch,high_boundary,low_boundary,right_boundary,left_boundary): # include boundary
	repulsive_F=np.zeros((len(coordinates[:,0]),2))
	tempx=len(GRID[:,0,0])-1
	tempy=len(GRID[0,:,0])-1
	for i in range(tempx):
		for j in range(tempy):
			for m in GRID[i,j,:]:
				if m!=EMPTY:
					for n in GRID[i,j,:]:
						if n!=m and n!=EMPTY:
							r_mn=np.sqrt((coordinates[m,0]-coordinates[n,0])**2+(coordinates[m,1]-coordinates[n,1])**2)
							if r_mn<gridsize:
								F_mn=(Maximum_repulsion*(1-r_mn/gridsize)*(coordinates[m,0]-coordinates[n,0])/r_mn,Maximum_repulsion*(1-r_mn/gridsize)*(coordinates[m,1]-coordinates[n,1])/r_mn)
								repulsive_F[m,:]+=F_mn
					for n in GRID[i+1,j,:]:
						if n!= EMPTY:
							r_mn=np.sqrt((coordinates[m,0]-coordinates[n,0])**2+(coordinates[m,1]-coordinates[n,1])**2)
							if r_mn<gridsize:
								F_mn=(Maximum_repulsion*(1-r_mn/gridsize)*(coordinates[m,0]-coordinates[n,0])/r_mn,Maximum_repulsion*(1-r_mn/gridsize)*(coordinates[m,1]-coordinates[n,1])/r_mn)
								repulsive_F[m,:]+=F_mn
								repulsive_F[n,:]-=F_mn
							if i==tempx-1 and boundary_switch==True:
								if (right_boundary-coordinates[n,0])<radius:
									repulsive_F[n,0]-=boundary_constant*(radius+right_boundary-coordinates[n,0])
					for n in GRID[i,j+1,:]:
						if n!= EMPTY:
							r_mn=np.sqrt((coordinates[m,0]-coordinates[n,0])**2+(coordinates[m,1]-coordinates[n,1])**2)
							if r_mn<gridsize:
								F_mn=(Maximum_repulsion*(1-r_mn/gridsize)*(coordinates[m,0]-coordinates[n,0])/r_mn,Maximum_repulsion*(1-r_mn/gridsize)*(coordinates[m,1]-coordinates[n,1])/r_mn)
								repulsive_F[m,:]+=F_mn
								repulsive_F[n,:]-=F_mn
							if j==tempy-1 and boundary_switch==True:
								if (high_boundary-coordinates[n,1])<radius:
									repulsive_F[n,1]-=boundary_constant*(radius+high_boundary-coordinates[n,1])
					for n in GRID[i+1,j+1,:]:
						if n!= EMPTY:
							r_mn=np.sqrt((coordinates[m,0]-coordinates[n,0])**2+(coordinates[m,1]-coordinates[n,1])**2)
							if r_mn<gridsize:
								F_mn=(Maximum_repulsion*(1-r_mn/gridsize)*(coordinates[m,0]-coordinates[n,0])/r_mn,Maximum_repulsion*(1-r_mn/gridsize)*(coordinates[m,1]-coordinates[n,1])/r_mn)
								repulsive_F[m,:]+=F_mn
								repulsive_F[n,:]-=F_mn
							if i==tempx-1 and j==tempy-1 and boundary_switch==True:
								if (high_boundary-coordinates[n,1])<radius:
									repulsive_F[n,1]-=boundary_constant*(radius+high_boundary-coordinates[n,1])
								if (right_boundary-coordinates[n,0])<radius:
									repulsive_F[n,0]-=boundary_constant*(radius+right_boundary-coordinates[n,0])
					if boundary_switch==True:
						if j==0:
							if (coordinates[m,1]-low_boundary)<radius:
								repulsive_F[m,1]+=boundary_constant*(radius-coordinates[m,1]+low_boundary)
						if i==0:
							if (coordinates[m,0]-left_boundary)<radius:
								repulsive_F[m,0]+=boundary_constant*(radius-coordinates[m,0]+left_boundary)
						if j==tempy-1:
							if (high_boundary-coordinates[m,1])<radius:
								repulsive_F[m,1]-=boundary_constant*(radius+high_boundary-coordinates[m,1])
						if i==tempx-1:
							if (right_boundary-coordinates[m,0])<radius:
								repulsive_F[m,0]-=boundary_constant*(radius+right_boundary-coordinates[m,0])				

	return repulsive_F

def elastic_force(coordinates):
	elastic_F=np.zeros((len(coordinates[:,0]),2))
	for i in range(len(coordinates[:,0])-1):
		r_ii1=np.sqrt((coordinates[i,0]-coordinates[i+1,0])**2+(coordinates[i,1]-coordinates[i+1,1])**2)
		F_ii1=((coordinates[i+1,0]-coordinates[i,0])/r_ii1*(r_ii1-lattice_const)*elastic_constant,(coordinates[i+1,1]-coordinates[i,1])/r_ii1*(r_ii1-lattice_const)*elastic_constant)
		elastic_F[i,:]+=F_ii1
		elastic_F[i+1,:]-=F_ii1
	return elastic_F


def Plot_current_curve(L,coordinates,step,newpath,high_boundary,low_boundary,right_boundary,left_boundary):
	fig=plt.figure(figsize=(10,10))
	systemsize=N*L*L-(N-1)	
	axval= fig.add_subplot(1,1,1)
	axval.set_xlim(left_boundary,right_boundary)
	axval.set_ylim(low_boundary,high_boundary)
	for k in range(systemsize-1):
		angline=lne.Line2D([coordinates[k,0],coordinates[k+1,0]],[coordinates[k,1],coordinates[k+1,1]],color='r',linewidth=1)
		axval.add_line(angline)
		cir1=ptch.Circle((coordinates[k,0],coordinates[k,1]),radius=radius,ec=(0.5, 0.5, 0.5),fc=(0.5, 0.5, 0.5), linewidth=1)
		axval.add_patch(cir1)
	cir1=ptch.Circle((coordinates[-1,0],coordinates[-1,1]),radius=radius,ec=(0.5, 0.5, 0.5),fc=(0.5, 0.5, 0.5), linewidth=1)
	axval.add_patch(cir1)
	if not os.path.exists(newpath):
		os.makedirs(newpath)
	figname=newpath+'/'+str(step).zfill(4)+'.png'
	fig.savefig(figname,dpi=60)
	plt.close()




coordinates=np.loadtxt("new_coordinates_by2D2020180304.dat")
systemsize=len(coordinates[:,0])
L=32
high_boundary=max(coordinates[:,1])+radius
low_boundary=min(coordinates[:,1])-radius
middle_v=(high_boundary+low_boundary)/2.0*np.ones(systemsize)
right_boundary=max(coordinates[:,0])+radius
left_boundary=min(coordinates[:,0])-radius
middle_p=(right_boundary+left_boundary)/2.0*np.ones(systemsize)

time=0.0
Total_time=int(L/speed_boundary)
Total_Acceleration=np.zeros((len(coordinates[:,0]),2))
Total_Velocity=np.zeros((len(coordinates[:,0]),2))
newpath='squeeze_L_'+str(int(L))+'_N_'+str(N)+'_'+str(Total_time)+'_'+str(timestep)
if not os.path.exists(newpath):
    os.makedirs(newpath)
for step in range(int(Total_time/timestep)):
	high_bound=max(coordinates[:,1])+radius
	low_bound=min(coordinates[:,1])-radius
	right_bound=max(coordinates[:,0])+radius
	left_bound=min(coordinates[:,0])-radius
	if right_boundary-left_boundary>(L+radius):
		left_boundary+=(0.5*speed_boundary*timestep)
		right_boundary-=(0.5*speed_boundary*timestep)
		coordinates[:,0]=coordinates[:,0]-(coordinates[:,0]-middle_p)*(speed_boundary*timestep)/(right_boundary-left_boundary)
	else:
		if high_boundary-low_boundary>18:
			low_boundary+=(0.5*speed_boundary*timestep)
			high_boundary-=(0.5*speed_boundary*timestep)
			coordinates[:,1]=coordinates[:,1]-(coordinates[:,1]-middle_v)*(speed_boundary*timestep)/(high_boundary-low_boundary)
		# else:
		# 	break
	# if right_boundary>right_bound+0.5*radius:
	# 	right_boundary=right_bound+0.5*radius
	# if left_boundary<left_bound-0.5*radius:
	# 	left_boundary=left_bound-0.5*radius
	# if right_boundary-left_boundary<(2*L):
	# 	speed_boundary=0.002
	GRID=EMPTY*np.ones((int((right_bound-left_bound)/gridsize)+2,int((high_bound-low_bound)/gridsize)+2,3))
	GRID=gridding(coordinates,L,high_bound,low_bound,right_bound,left_bound)
	#print GRID, '\n'
	Total_F=elastic_force(coordinates)+repulsive_force(coordinates,GRID,L,True,high_boundary,low_boundary,right_boundary,left_boundary)-0.2*Total_Velocity#+random_force(systemsize,0.0,0.1)
	#print Total_F, '\n'
	Total_Acceleration=Total_F/mass
	#print Total_Acceleration, '\n'
	Total_Velocity+=Total_Acceleration*timestep
	#print Total_Velocity, '\n\n\n\n'
	coordinates+=Total_Velocity*timestep
	#print coordinates, '\n\n\n'
	if step%1000==0:
		Plot_current_curve(L,coordinates,int(step/1000),newpath,high_boundary,low_boundary,right_boundary,left_boundary)
		filename=newpath+'/squeeze_new_coordinates_'+str(int(step/1000))+'.dat'
		np.savetxt(filename,coordinates)
	time+=timestep

# print GRID[0:2*L,0:2*L,:]


# 	fig=plt.figure(figsize=(10,10))
# 				axval= fig.add_subplot(1, 1, 1)
# 				for k in range(systemsize):
# 					cir1=ptch.Circle((position[k,0],position[k,1]),radius=0.1,ec=(0.5, 0.5, 0.5),fc=(0.5, 0.5, 0.5), linewidth=1)
# 					axval.add_patch(cir1)
# 					for neighbor in Neighbor_List[k,:]:
# 						if neighbor>=0:
# 							x0=position[k,0]
# 							x1=position[neighbor,0]
# 							y0=position[k,1]
# 							y1=position[neighbor,1]
# 							x1=x1-L*rest_length*np.round((x1-x0)/float(L))
# 							angline=lne.Line2D([x0,x1],[y0,y1],color=(0.5, 0.5, 0.5),linewidth=1)
# 							axval.add_line(angline)




# axval.set_xlim(-1,L+1)
# 					axval.set_ylim(-1,0.5*np.sqrt(3)*L+1)
					#fig.legend(loc='best')
