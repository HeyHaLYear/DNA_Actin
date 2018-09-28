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
timestep=0.02

List_L = []
# initialization: First iteration
dictionary= {}
for x in range(1,7):
	List_L.append(int(2**x))
	dictionary['coordinates_%02d' % x] = EMPTY*np.ones((4**x,2))

dictionary['coordinates_01'][0,:]=(0.5,0.5)
dictionary['coordinates_01'][1,:]=(0.5,1.5)
dictionary['coordinates_01'][2,:]=(1.5,1.5)
dictionary['coordinates_01'][3,:]=(1.5,0.5)

# initialization: gridding
gridsize=2*radius


def gridding(coordinates,L,high_boundary,low_boundary):
	for k in range(N*L*L-(N-1)):
		index_x=int((coordinates[k,0]-low_boundary)/gridsize)
		index_y=int((coordinates[k,1]-low_boundary)/gridsize)
		#print index_x,index_y
		for i in range(3):
			if GRID[index_x,index_y,i]==EMPTY:
				GRID[index_x,index_y,i]=k
				break
	return GRID

def random_force(L,mu,sigma):
	random_F=np.zeros((int(N*L*L-(N-1)),2))
	for i in range(N*L*L-(N-1)):
		random_F[i,0]=random.gauss(mu,sigma)
		random_F[i,1]=random.gauss(mu,sigma)
	return random_F

def repulsive_force(coordinates,GRID,L,boundary_switch,high_boundary,low_boundary): # include boundary
	repulsive_F=np.zeros((len(coordinates[:,0]),2))
	temp=len(GRID[:,0,0])-1
	for i in range(temp):
		for j in range(temp):
			for m in GRID[i,j,:]:
				if m!=EMPTY:
					for n in GRID[i,j,:]:
						if n!=m and n!=EMPTY:
							r_mn=np.sqrt((coordinates[m,0]-coordinates[n,0])**2+(coordinates[m,1]-coordinates[n,1])**2)
							if r_mn<gridsize:
								F_mn=(Maximum_repulsion*(gridsize-r_mn)*(coordinates[m,0]-coordinates[n,0])/r_mn,Maximum_repulsion*(gridsize-r_mn)*(coordinates[m,1]-coordinates[n,1])/r_mn)
								repulsive_F[m,:]+=F_mn
					for n in GRID[i+1,j,:]:
						if n!= EMPTY:
							r_mn=np.sqrt((coordinates[m,0]-coordinates[n,0])**2+(coordinates[m,1]-coordinates[n,1])**2)
							if r_mn<gridsize:
								F_mn=(Maximum_repulsion*(gridsize-r_mn)*(coordinates[m,0]-coordinates[n,0])/r_mn,Maximum_repulsion*(gridsize-r_mn)*(coordinates[m,1]-coordinates[n,1])/r_mn)
								repulsive_F[m,:]+=F_mn
								repulsive_F[n,:]-=F_mn
							if i==temp-1 and boundary_switch==True:
								if (high_boundary-coordinates[n,0])<radius:
									repulsive_F[n,0]-=boundary_constant*(radius+high_boundary-coordinates[n,0])
					for n in GRID[i,j+1,:]:
						if n!= EMPTY:
							r_mn=np.sqrt((coordinates[m,0]-coordinates[n,0])**2+(coordinates[m,1]-coordinates[n,1])**2)
							if r_mn<gridsize:
								F_mn=(Maximum_repulsion*(gridsize-r_mn)*(coordinates[m,0]-coordinates[n,0])/r_mn,Maximum_repulsion*(gridsize-r_mn)*(coordinates[m,1]-coordinates[n,1])/r_mn)
								repulsive_F[m,:]+=F_mn
								repulsive_F[n,:]-=F_mn
							if j==temp-1 and boundary_switch==True:
								if (high_boundary-coordinates[n,1])<radius:
									repulsive_F[n,1]-=boundary_constant*(radius+high_boundary-coordinates[n,1])
					for n in GRID[i+1,j+1,:]:
						if n!= EMPTY:
							r_mn=np.sqrt((coordinates[m,0]-coordinates[n,0])**2+(coordinates[m,1]-coordinates[n,1])**2)
							if r_mn<gridsize:
								F_mn=(Maximum_repulsion*(gridsize-r_mn)*(coordinates[m,0]-coordinates[n,0])/r_mn,Maximum_repulsion*(gridsize-r_mn)*(coordinates[m,1]-coordinates[n,1])/r_mn)
								repulsive_F[m,:]+=F_mn
								repulsive_F[n,:]-=F_mn
							if i==temp-1 and j==temp-1 and boundary_switch==True:
								if (high_boundary-coordinates[n,1])<radius:
									repulsive_F[n,1]-=boundary_constant*(radius+high_boundary-coordinates[n,1])
								if (high_boundary-coordinates[n,0])<radius:
									repulsive_F[n,0]-=boundary_constant*(radius+high_boundary-coordinates[n,0])
					if boundary_switch==True:
						if j==0:
							if (coordinates[m,1]-low_boundary)<radius:
								repulsive_F[m,1]+=boundary_constant*(radius-coordinates[m,1]+low_boundary)
						if i==0:
							if (coordinates[m,0]-low_boundary)<radius:
								repulsive_F[m,0]+=boundary_constant*(radius-coordinates[m,0]+low_boundary)
						if j==temp-1:
							if (high_boundary-coordinates[m,1])<radius:
								repulsive_F[m,1]-=boundary_constant*(radius+high_boundary-coordinates[m,1])
						if i==temp-1:
							if (high_boundary-coordinates[m,0])<radius:
								repulsive_F[m,0]-=boundary_constant*(radius+high_boundary-coordinates[m,0])				

	return repulsive_F

def elastic_force(coordinates):
	elastic_F=np.zeros((len(coordinates[:,0]),2))
	for i in range(len(coordinates[:,0])-1):
		r_ii1=np.sqrt((coordinates[i,0]-coordinates[i+1,0])**2+(coordinates[i,1]-coordinates[i+1,1])**2)
		F_ii1=((coordinates[i+1,0]-coordinates[i,0])/r_ii1*(r_ii1-lattice_const)*elastic_constant,(coordinates[i+1,1]-coordinates[i,1])/r_ii1*(r_ii1-lattice_const)*elastic_constant)
		elastic_F[i,:]+=F_ii1
		elastic_F[i+1,:]-=F_ii1
	return elastic_F


def boundary_elastic_force(coordinates):
	boundary_elastic_F=np.zeros((len(coordinates[:,0]),2))

def create_curve():
	# other generations:
	for x in range(1,7):
		M_keep=EMPTY*np.ones((4,2))
		M_13=EMPTY*np.ones((4,2))
		M_02=EMPTY*np.ones((4,2))
		filename='coordinates_L_'+str(List_L[x-1])+'_'+str(N)+'.dat'
		temp=np.zeros((int(N*len(dictionary['coordinates_%02d' % x][:,0])-(N-1)),2))
		for i in range(len(dictionary['coordinates_%02d' % x][:,0])-1):
			for j in range(int(N)):
				temp[N*i+j,:]=(1-float(j)/N)*dictionary['coordinates_%02d' % x][i,:]+float(j)/N*dictionary['coordinates_%02d' % x][i+1,:]
		temp[-1,:]=dictionary['coordinates_%02d' % x][-1,:]
		np.savetxt(filename,temp)

		for k in range(int(len(dictionary['coordinates_%02d' % x][:,0])/4)):	
			if x==6:
				break
			min_x=np.amin(dictionary['coordinates_%02d' % x][4*k:4*k+4,0])
			min_y=np.amin(dictionary['coordinates_%02d' % x][4*k:4*k+4,1])
			M_keep=dictionary['coordinates_%02d' % x][4*k:4*k+4,]-((min_x,min_y),(min_x,min_y),(min_x,min_y),(min_x,min_y))
			M_02[0,:]=M_keep[2,:]
			M_02[1,:]=M_keep[1,:]
			M_02[2,:]=M_keep[0,:]
			M_02[3,:]=M_keep[3,:]
			M_13[0,:]=M_keep[0,:]
			M_13[1,:]=M_keep[3,:]
			M_13[2,:]=M_keep[2,:]
			M_13[3,:]=M_keep[1,:]
			x_old=2*dictionary['coordinates_%02d' % x][4*k,0]-0.5
			y_old=2*dictionary['coordinates_%02d' % x][4*k,1]-0.5
			dictionary['coordinates_%02d' % (x+1)][16*k:16*k+4,:]=M_13+((x_old,y_old),(x_old,y_old),(x_old,y_old),(x_old,y_old))
			x_old=2*dictionary['coordinates_%02d' % x][4*k+1,0]-0.5
			y_old=2*dictionary['coordinates_%02d' % x][4*k+1,1]-0.5
			dictionary['coordinates_%02d' % (x+1)][16*k+4:16*k+8,:]=M_keep+((x_old,y_old),(x_old,y_old),(x_old,y_old),(x_old,y_old))
			x_old=2*dictionary['coordinates_%02d' % x][4*k+2,0]-0.5
			y_old=2*dictionary['coordinates_%02d' % x][4*k+2,1]-0.5
			dictionary['coordinates_%02d' % (x+1)][16*k+8:16*k+12,:]=M_keep+((x_old,y_old),(x_old,y_old),(x_old,y_old),(x_old,y_old))
			x_old=2*dictionary['coordinates_%02d' % x][4*k+3,0]-0.5
			y_old=2*dictionary['coordinates_%02d' % x][4*k+3,1]-0.5
			dictionary['coordinates_%02d' % (x+1)][16*k+12:16*k+16,:]=M_02+((x_old,y_old),(x_old,y_old),(x_old,y_old),(x_old,y_old))
		
	# print dictionary['coordinates_03']

def Plot_Hilbert_Curve():
	fig=plt.figure(figsize=(30,20))
	for L in enumerate(List_L):
		systemsize=N*L[1]*L[1]-(N-1)
		filename='coordinates_L_'+str(L[1])+'_'+str(N)+'.dat'
		if os.path.isfile(filename)==False:
			create_curve()
		coordinates=np.loadtxt(filename)
		# print coordinates
		
		axval= fig.add_subplot(2, 3, int(L[0]+1))
		axval.set_xlim(0.0,L[1])
		axval.set_ylim(0.0,L[1])
		for k in range(systemsize-1):
			angline=lne.Line2D([coordinates[k,0],coordinates[k+1,0]],[coordinates[k,1],coordinates[k+1,1]],color='r',linewidth=1)
			axval.add_line(angline)
	figname='Hilbert_curve.png'
	fig.savefig(figname,dpi=400)

def Plot_current_curve(L,coordinates,step,newpath,high_boundary,low_boundary):
	fig=plt.figure(figsize=(10,10))
	systemsize=N*L*L-(N-1)	
	axval= fig.add_subplot(1,1,1)
	axval.set_xlim(low_boundary,high_boundary)
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

# def Create_diluted_network():



# def Coupling_layer():


L=List_L[4]
low_boundary=-0.1*float(L)
high_boundary=float(L)*1.1
systemsize=N*L*L-(N-1)
filename='coordinates_L_'+str(L)+'_'+str(N)+'.dat'
if os.path.isfile(filename)==False:
	create_curve()
figname='Hilbert_curve.png'
if os.path.isfile(figname)==False:
	Plot_Hilbert_Curve()
coordinates=np.loadtxt(filename)
# print coordinates
time=0.0
Total_time=2000
Total_Acceleration=np.zeros((len(coordinates[:,0]),2))
Total_Velocity=np.zeros((len(coordinates[:,0]),2))
newpath='L_'+str(int(L))+'_N_'+str(N)+'_'+str(Total_time)+'_'+str(timestep)
for step in range(int(Total_time/timestep)):
	GRID=EMPTY*np.ones((int((high_boundary-low_boundary)/gridsize)+2,int((high_boundary-low_boundary)/gridsize)+2,3))
	GRID=gridding(coordinates,L,high_boundary,low_boundary)
	#print GRID, '\n'
	Total_F=elastic_force(coordinates)+repulsive_force(coordinates,GRID,L,True,high_boundary,low_boundary)+random_force(L,0.0,0.15)-0.2*Total_Velocity
	#print Total_F, '\n'
	Total_Acceleration=Total_F/mass
	#print Total_Acceleration, '\n'
	Total_Velocity+=Total_Acceleration*timestep
	#print Total_Velocity, '\n\n\n\n'
	coordinates+=Total_Velocity*timestep
	#print coordinates, '\n\n\n'
	if step%10000==0:
		Plot_current_curve(L,coordinates,int(step/100),newpath,high_boundary,low_boundary)
	time+=timestep
filename=newpath+'/'+'new_coordinates.dat'
np.savetxt(filename,coordinates)
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
