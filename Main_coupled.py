# Kuang Liu 20170704 rupture of diluted triangular network
#!/usr/bin/python

import random
import sys, os, glob
# Note: changed to simple pickle on move to python 3. Compatibility problems with loading old pickle files expected!
import copy as cp
import numpy as np
import scipy as sp
# import scipy.io as sio
from scipy.interpolate import interp1d
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as ptch
import matplotlib.lines as lne
import matplotlib.colors as mcol
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.cm as cm
from matplotlib import animation
import datetime

#Set parameters:
Delta_t=0.002
EMPTY=-1

k_actin=1.0 # spring constant in actin region
a_actin=3.0	# lattice constant in actin region
m_actin=1.0	# mass of sites in actin region
damp_actin=0.2 # damping coefficient in actin region
fracture_lamda_actin=5.0
p=0.0 # fraction of actin bonds that are occupied
R_0=0.95*a_actin # radius of contractile center in actin region
F_0=0.5 # Maximum force of contractile center in actin region


# features of single layer
k_layer=3.333
a_layer=0.3
m_layer=1.0
damp_layer=0.4
fracture_lamda_layer=10.0
R_0_layer=[0.2,0.2] #with actin and DNA respectively
F_0_layer=[0.3,0.3]
radius=0.15

# features of DNA. Width:
k_DNA=5.0
a_DNA=0.5
m_DNA=1.0
radius_DNA=float(0.5*a_DNA)*0.995
gridsize=2.0*radius_DNA
damp_DNA=0.4
fracture_lamda_DNA=10.0
Max_repulsion_DNA=2.0 # repulsion between DNA particles
R_0_DNA=0.6666*a_actin
F_0_DNA=0.05
k_bottom_boundary=10.0 # bottom boundary repulsion in DNA region
k_attraction=5.0

Length=32 # system length
time_step=0.01 # simulation time step
total_time=200

def Spring_Interaction(x1,y1,x2,y2,rest_length,k,fracture_lamda):
	x1=x1-Length*rest_length*np.round((x1-x2)/float(Length))
	Distance = np.sqrt((x1-x2)**2+(y1-y2)**2)
	force=np.zeros(2)
	BROKEN=False
	if Distance>(1.0+fracture_lamda)*rest_length:
		BROKEN=True
		# for neighbor in Neighbor_List[j,:]:
		# 	if neighbor==i:
		# 		neighbor=BROKEN
	else:						
		force[0]=k*(Distance-rest_length)*(x2-x1)/Distance
		force[1]=k*(Distance-rest_length)*(y2-y1)/Distance
	# if i==m and step>=0:
	# 	print x1,x2,y1,y2
	# 	print Distance, rest_length
	# 	print spring_constant
	# 	print force
	return BROKEN,force

def Contractile_Active_units(x,y,Active_Unit_x,Active_Unit_y):
	force=np.zeros(2)
	for i in range(len(Active_Unit_x)):
		Distance=np.sqrt((x-Active_Unit_x[i])**2+(y-Active_Unit_y[i])**2)
		if Distance<R_0:
			force[0]+=(Active_Unit_x[i]-x)/R_0*F_0
			force[1]+=(Active_Unit_y[i]-y)/R_0*F_0
		else:
			if Distance<2*R_0:
				force[0]+=(Active_Unit_x[i]-x)/Distance*(2-Distance/R_0)*F_0
				force[1]+=(Active_Unit_y[i]-y)/Distance*(2-Distance/R_0)*F_0
	return force

def Contractile_Active_units_DNA(x,y,Active_Unit_x,Active_Unit_y):
	force=np.zeros(2)
	for i in range(len(Active_Unit_x)):
		Distance=np.sqrt((x-Active_Unit_x[i])**2+(y-Active_Unit_y[i])**2)
		if Distance<R_0_DNA:
			force[0]-=(Active_Unit_x[i]-x)/R_0_DNA*F_0_DNA
			force[1]-=(Active_Unit_y[i]-y)/R_0_DNA*F_0_DNA
		else:
			if Distance<2*R_0_DNA:
				force[0]-=(Active_Unit_x[i]-x)/Distance*(2-Distance/R_0_DNA)*F_0_DNA
				force[1]-=(Active_Unit_y[i]-y)/Distance*(2-Distance/R_0_DNA)*F_0_DNA
	return force

def Contractile_Layer(x,y,position_layer,TYPE):
	force=np.zeros(2)
	minimal_distance=2*R_0_layer[TYPE]
	nearest=0
	for i in range(len(position_layer[:,0])):
		Distance=np.sqrt((x-position_layer[i,0])**2+(y-position_layer[i,1])**2)
		if Distance<minimal_distance:
			if Distance<R_0_layer[TYPE]:
				force[0]=(position_layer[i,0]-x)/R_0_layer[TYPE]*F_0_layer[TYPE]
				force[1]=(position_layer[i,1]-y)/R_0_layer[TYPE]*F_0_layer[TYPE]
			else:
				if Distance<2*R_0:
					force[0]=(position_layer[i,0]-x)/Distance*(2-Distance/R_0_layer[TYPE])*F_0_layer[TYPE]
					force[1]=(position_layer[i,1]-y)/Distance*(2-Distance/R_0_layer[TYPE])*F_0_layer[TYPE]

			minimal_distance=Distance
			nearest=i
	return force, nearest


def Create_Actin_Network():
	num_row = int(Length/a_actin)
	num_column = int(Length/a_actin)+1
	systemsize=num_column*num_row
	Num_neighbors = np.zeros(systemsize)
	position = np.zeros((systemsize,2))
	position_previous = np.zeros((systemsize,2))
	Neighbor_List = EMPTY * np.ones((systemsize,6))


	for row in range(num_row):
		for column in range(num_column):
			label = row*num_column+column
			position[label,0] = float(column*1.0*a_actin+int(row%2)*0.5*a_actin)
			position[label,1] = float(row*1.0*a_actin*0.5*np.sqrt(3))
			
			
			if row<num_row-1 and column<num_column-1:
				label_1=(row+1)*num_column+column
				label_2=label+1
				#label_2=(row+1)*num_column+((column-1+2*(row%2)+num_column)%num_column)

				occupied=random.random()
				if occupied<p:
					Neighbor_List[label,Num_neighbors[label]]=label_1
					Num_neighbors[label]+=1
					Neighbor_List[label_1,Num_neighbors[label_1]]=label
					Num_neighbors[label_1]+=1

				occupied=random.random()
				if occupied<p:
					Neighbor_List[label,Num_neighbors[label]]=label_2
					Num_neighbors[label]+=1
					Neighbor_List[label_2,Num_neighbors[label_2]]=label
					Num_neighbors[label_2]+=1

				if row%2==1:
					label_3=(row+1)*num_column+column+1
					label_4=(row-1)*num_column+column+1
					occupied=random.random()
					if occupied<p:
						Neighbor_List[label,Num_neighbors[label]]=label_3
						Num_neighbors[label]+=1
						Neighbor_List[label_3,Num_neighbors[label_3]]=label
						Num_neighbors[label_3]+=1

					occupied=random.random()
					if occupied<p:
						Neighbor_List[label,Num_neighbors[label]]=label_4
						Num_neighbors[label]+=1
						Neighbor_List[label_4,Num_neighbors[label_4]]=label
						Num_neighbors[label_4]+=1

	return position, Neighbor_List

def Create_Spring_layer():
	position=np.zeros((int(Length/a_layer)+1,2))
	for i in range(int(Length/a_layer)+1):
		position[i,0]=i*a_layer

	return position

def Create_DNA():
	if not os.path.exists("new_coordinates.dat"):
		print "No created hilbert curve"
		return EMPTY
	else:
		position=np.loadtxt("new_coordinates.dat")
		offset=np.amax(position[:,1])
		for i in range(len(position[:,1])):
			position[i,1]-=offset
		bottom_boundary=np.amin(position[:,1])
		return position, bottom_boundary

def Put_Active_units(Length,Active_Unit_x,Active_Unit_y):
	x=random.random()*Length
	y=random.random()*Length*0.5*np.sqrt(3)
	Active_Unit_x.append(x)
	Active_Unit_y.append(y)

def Put_Active_units_DNA(Length,Active_Unit_x,Active_Unit_y):
	x=random.random()*Length
	y=random.random()*(-16.0)
	Active_Unit_x.append(x)
	Active_Unit_y.append(y)


def Plot(position_actin,Neighbor_actin,position_layer,position_DNA,Small_Attraction_DNA,Active_Unit_x,Active_Unit_y,parameter):
	fig=plt.figure(figsize=(12,12))
	axval= fig.add_subplot(1, 1, 1)


	# Make a user-defined colormap.
	cm1 = mcol.LinearSegmentedColormap.from_list("MyCmapName",["r","b"])
	# Make a normalizer that will map the time values from
	# [start_time,end_time+1] -> [0,1].
	cnorm = mcol.Normalize(vmin=-0.1,vmax=0.3)
	# Turn these into an object that can be used to map time values to colors and
	# can be passed to plt.colorbar().
	cpick = cm.ScalarMappable(norm=cnorm,cmap=cm1)
	cpick.set_array([])
	#color=cpick.to_rgba(#some number between vmin and vmax)

	for k in range(len(position_actin)-(int(Length/a_actin)+1)):
		# cir1=ptch.Circle((position[k,0],position[k,1]),radius=radius,ec=(0.5, 0.5, 0.5),fc=(0.5, 0.5, 0.5), linewidth=2)
		# axval.add_patch(cir1)
		for neighbor in Neighbor_actin[k,:]:
			if neighbor>=0 and abs(position_actin[k,0]-position_actin[neighbor,0])<(Length/2.0):
				Distance=np.sqrt((position_actin[k,0]-position_actin[neighbor,0])**2+(position_actin[k,1]-position_actin[neighbor,1])**2)
				angline=lne.Line2D([position_actin[k,0],position_actin[neighbor,0]],[position_actin[k,1],position_actin[neighbor,1]],color=cpick.to_rgba(float(Distance-a_actin)/a_actin),linewidth=2*a_actin)
				axval.add_line(angline)

	for k in range(len(position_layer)-1):
		cir1=ptch.Circle((position_layer[k,0],position_layer[k,1]),radius=radius,ec='k',fc='k', linewidth=0.7)
		axval.add_patch(cir1)
		angline=lne.Line2D([position_layer[k,0],position_layer[neighbor,0]],[position_layer[k,1],position_layer[neighbor,1]],color='k',linewidth=0.7)
	cir1=ptch.Circle((position_layer[-1,0],position_layer[-1,1]),radius=radius,ec='k',fc='k', linewidth=0.7)
	axval.add_patch(cir1)

	for k in range(len(position_DNA)-1):
		angline=lne.Line2D([position_DNA[k,0],position_DNA[k+1,0]],[position_DNA[k,1],position_DNA[k+1,1]],color='r',linewidth=1)
		axval.add_line(angline)
		cir1=ptch.Circle((position_DNA[k,0],position_DNA[k,1]),radius=radius_DNA,ec=(0.5,0.5,0.5),fc=(0.5,0.5,0.5), linewidth=1)
		axval.add_patch(cir1)
	cir1=ptch.Circle((position_DNA[-1,0],position_DNA[-1,1]),radius=radius_DNA,ec=(0.5,0.5,0.5),fc=(0.5,0.5,0.5), linewidth=1)
	axval.add_patch(cir1)

	for k in range(len(Small_Attraction_DNA[:,0])):
		p1=Small_Attraction_DNA[k,0]
		p2=Small_Attraction_DNA[k,1]
		angline=lne.Line2D([position_DNA[p1,0],position_DNA[p2,0]],[position_DNA[p1,1],position_DNA[p2,1]],color='b',linewidth=1)
		axval.add_line(angline)



	for k in range(len(Active_Unit_x)):
		cir1=ptch.Circle((Active_Unit_x[k],Active_Unit_y[k]),radius=R_0,ec='y',fc='y', linewidth=0.7)
		axval.add_patch(cir1)

	for k in range(len(Active_Unit_x_DNA)):
		cir1=ptch.Circle((Active_Unit_x_DNA[k],Active_Unit_y_DNA[k]),radius=R_0_DNA,ec='y',fc='y', linewidth=0.7)
		axval.add_patch(cir1)


	axval.set_xlim(-1.5,33.5)
	axval.set_ylim(-20,25)
	#fig.legend(loc='best')
	plt.colorbar(cpick,label="elastic force in actin network")
	figname='animation_All_coupled/'+str(parameter).zfill(4)+'.png'
	fig.savefig(figname,dpi=100)
	plt.clf()
	plt.close()

def gridding(position_DNA):
	left=-2.0
	right=35.0
	bottom=-21.0
	top=15.0
	num_x=int((right-left)/gridsize)
	num_y=int((top-bottom)/gridsize)
	GRID=EMPTY*np.ones((num_x,num_y,4))
	Index=np.zeros((len(position_DNA[:,0]),2))
	for k in range(len(position_DNA[:,0])):
		index_x=int((position_DNA[k,0]-left)/gridsize)
		index_y=int((position_DNA[k,1]-bottom)/gridsize)
		#print index_x,index_y
		Index[k,0]=index_x
		Index[k,1]=index_y
		for i in range(4):
			if GRID[index_x,index_y,i]==EMPTY:
				GRID[index_x,index_y,i]=k
				break
	return GRID, Index

def Repulsion_DNA(GRID,Index,position_DNA):
	Repulsion_Force_DNA=np.zeros((len(position_DNA[:,0]),2))
	for k in range(len(position_DNA[:,0])):
		x1=position_DNA[k,0]
		y1=position_DNA[k,1]
		for i in range(3):
			for j in range(3):
				for m in GRID[Index[k,0]+i-1,Index[k,1]+j-1,:]:
					if m!=EMPTY and m!=k:
						x2=position_DNA[m,0]
						y2=position_DNA[m,1]
						Distance=np.sqrt((x1-x2)**2+(y1-y2)**2)
						if Distance<gridsize:
							F=(Max_repulsion_DNA*(1.0-Distance/gridsize)*(x1-x2)/Distance,Max_repulsion_DNA*(1.0-Distance/gridsize)*(y1-y2)/Distance)
							Repulsion_Force_DNA[k,:]+=F
	return Repulsion_Force_DNA



def Update(position_actin,velocity_actin,position_layer,velocity_layer,position_DNA,velocity_DNA,Small_Attraction_DNA,Neighbor_List,Active_Unit_x,Active_Unit_y,bottom_boundary):
	num_row = int(Length/a_actin)
	num_column = int(Length/a_actin)+1
	Force_actin=np.zeros((num_row*num_column,2))
	Force_layer=np.zeros((len(position_layer[:,0]),2))
	Force_DNA=np.zeros((len(position_DNA[:,0]),2))
	for i in range(num_row*num_column-num_column):
		x1=position_actin[i,0]
		y1=position_actin[i,1]
		if x1>0 and x1<int(Length/a_actin)*a_actin+0.5:
			force=Contractile_Active_units(x1,y1,Active_Unit_x,Active_Unit_y)
			Force_actin[num_column+i,:]+=force
			for k in range(len(Neighbor_List[i,:])):
				if Neighbor_List[i,k]!=EMPTY:
					x2=position_actin[Neighbor_List[i,k],0]
					y2=position_actin[Neighbor_List[i,k],1]
					Broken,force=Spring_Interaction(x1,y1,x2,y2,a_actin,k_actin,fracture_lamda_actin)
					if Broken:
						Neighbor_List[i,k]=EMPTY
						print 'BROKEN!'
					else:
						Force_actin[i,:]+=force
	for i in range(num_column):
		x1=position_actin[i,0]
		y1=position_actin[i,1]
		force,nearest=Contractile_Layer(x1,y1,position_layer,0)
		Force_actin[i,:]+=force
		Force_layer[nearest,:]-=force
	Force_actin[0,:]=(0,0)
	Force_actin[num_column-1,:]=(0,0)

	Acceleration_actin=1.0/m_actin*(Force_actin-damp_actin*velocity_actin)
	velocity_actin+=(Acceleration_actin*time_step)
	position_actin+=(velocity_actin*time_step+0.5*time_step**2*Acceleration_actin)

	# update single-layer spring(envelope)
	for i in range(len(position_layer[:,0])-1):
		x1=position_layer[i,0]
		y1=position_layer[i,1]
		x2=position_layer[i+1,0]
		y2=position_layer[i+1,1]
		Broken,force=Spring_Interaction(x1,y1,x2,y2,a_layer,k_layer,fracture_lamda_layer)
		Force_layer[i,:]+=force
		Force_layer[i+1,:]-=force
		
	Force_layer[0,:]=(0,0)
	Force_layer[i,:]=(0,0)

	


	# update DNA part
	GRID,Index=gridding(position_DNA)
	temp=Repulsion_DNA(GRID,Index,position_DNA)
	Force_DNA+=temp
	for i in range(len(position_DNA[:,0])-1):
		x1=position_DNA[i,0]
		y1=position_DNA[i,1]
		force=Contractile_Active_units_DNA(x1,y1,Active_Unit_x_DNA,Active_Unit_y_DNA)
		Force_DNA[i,:]+=force

		x2=position_DNA[i+1,0]
		y2=position_DNA[i+1,1]
		Broken,force=Spring_Interaction(x1,y1,x2,y2,a_DNA,k_DNA,fracture_lamda_DNA)
		Force_DNA[i,:]+=force
		Force_DNA[i+1,:]-=force
		del force

		if y1>-1.5:
			force,nearest=Contractile_Layer(x1,y1,position_layer,1)
			Force_DNA[i,:]+=force
			Force_layer[nearest,:]-=force
		if y1<bottom_boundary:
			Force_DNA[i,:]+=[0.0,k_bottom_boundary*(bottom_boundary-y1)]
	for i in range(len(Small_Attraction_DNA[:,0])):
		x1=position_DNA[Small_Attraction_DNA[i,0],0]
		y1=position_DNA[Small_Attraction_DNA[i,0],1]
		x2=position_DNA[Small_Attraction_DNA[i,1],0]
		y2=position_DNA[Small_Attraction_DNA[i,1],1]
		Broken,force=Spring_Interaction(x1,y1,x2,y2,Small_Attraction_DNA[i,2],k_DNA,fracture_lamda_DNA)
		Force_DNA[Small_Attraction_DNA[i,0],:]+=force
		Force_DNA[Small_Attraction_DNA[i,1],:]-=force


	Acceleration_layer=1.0/m_layer*(Force_layer-damp_layer*velocity_layer)
	velocity_layer+=(Acceleration_layer*time_step)
	position_layer+=(velocity_layer*time_step+0.5*time_step**2*Acceleration_layer)

	Acceleration_DNA=1.0/m_DNA*(Force_DNA-damp_DNA*velocity_DNA)
	velocity_DNA+=(Acceleration_DNA*time_step)
	position_DNA+=(velocity_DNA*time_step+0.5*time_step**2*Acceleration_DNA)



	return position_actin, Neighbor_List, position_layer, position_DNA





# Main function
position_actin, Neighbor_actin=Create_Actin_Network()
velocity_actin=np.zeros((len(position_actin[:,0]),2))
position_layer=Create_Spring_layer()
velocity_layer=np.zeros((len(position_layer[:,0]),2))
position_DNA, bottom_boundary=Create_DNA()
velocity_DNA=np.zeros((len(position_DNA[:,0]),2))

# put in active units by hand or by calling 'Put...' function
# num_active_units=1
Active_Unit_x=[0.5*Length]
Active_Unit_y=[0.25*np.sqrt(3)*Length]

# Active_Unit_x_DNA=[0.333*Length,0.666*Length,0.333*Length,0.666*Length]
# Active_Unit_y_DNA=[-4.0,-4.0,-12.0,-12.0]

Active_Unit_x_DNA=[0.5*Length]
Active_Unit_y_DNA=[-8.0]
# Active_Unit_x_DNA=[]
# Active_Unit_y_DNA=[]
# for i in range(4):
# 	Put_Active_units_DNA(Length,Active_Unit_x_DNA,Active_Unit_y_DNA)

Small_Attraction_DNA=np.zeros((100,3))
GRID, Index=gridding(position_DNA)
i=0
while i<len(Small_Attraction_DNA[:,0]):
	particle_temp=int(random.random()*(len(Index[:,0])-1))
	Small_Attraction_DNA[i,0]=particle_temp
	neighbor_temp=[]
	for k in GRID[Index[particle_temp,0],Index[particle_temp,1],:]:
		if k!=EMPTY and abs(k-particle_temp)>1:
			neighbor_temp.append(k)
	for k in GRID[Index[particle_temp,0]+1,Index[particle_temp,1],:]:
		if k!=EMPTY and abs(k-particle_temp)>1:
			neighbor_temp.append(k)
	for k in GRID[Index[particle_temp,0]-1,Index[particle_temp,1],:]:
		if k!=EMPTY and abs(k-particle_temp)>1:
			neighbor_temp.append(k)
	for k in GRID[Index[particle_temp,0],Index[particle_temp,1]+1,:]:
		if k!=EMPTY and abs(k-particle_temp)>1:
			neighbor_temp.append(k)
	for k in GRID[Index[particle_temp,0],Index[particle_temp,1]-1,:]:
		if k!=EMPTY and abs(k-particle_temp)>1:
			neighbor_temp.append(k)
	for k in GRID[Index[particle_temp,0]+1,Index[particle_temp,1]+1,:]:
		if k!=EMPTY and abs(k-particle_temp)>1:
			neighbor_temp.append(k)
	for k in GRID[Index[particle_temp,0]+1,Index[particle_temp,1]-1,:]:
		if k!=EMPTY and abs(k-particle_temp)>1:
			neighbor_temp.append(k)
	for k in GRID[Index[particle_temp,0]-1,Index[particle_temp,1]+1,:]:
		if k!=EMPTY and abs(k-particle_temp)>1:
			neighbor_temp.append(k)
	for k in GRID[Index[particle_temp,0]-1,Index[particle_temp,1]-1,:]:
		if k!=EMPTY and abs(k-particle_temp)>1:
			neighbor_temp.append(k)
	if len(neighbor_temp)>0:
		pair_temp=random.choice(neighbor_temp)
		#pair_temp=neighbor_temp[int(random.random()*len(neighbor_temp))]
		Small_Attraction_DNA[i,1]=pair_temp
		delta_x=position_DNA[particle_temp,0]-position_DNA[pair_temp,0]
		delta_y=position_DNA[particle_temp,1]-position_DNA[pair_temp,1]
		Small_Attraction_DNA[i,2]=np.sqrt(delta_x**2+delta_y**2)
		i+=1

print Small_Attraction_DNA


# print position_actin
for i in range(int(total_time/time_step)):
	position_actin,Neighbor_actin,position_layer,position_DNA=Update(position_actin, \
		velocity_actin,position_layer,velocity_layer,position_DNA,velocity_DNA,Small_Attraction_DNA,Neighbor_actin,Active_Unit_x,Active_Unit_y,bottom_boundary)
	#print position_actin
	# Update_layer()
	# Update_DNA()
	if i%int(total_time)==0:
		#Plot(position_actin,Neighbor_actin,position_layer,position_DNA,Active_Unit_x,Active_Unit_y,int(i/int(total_time)))
		# filename='animation_All_coupled/information_L_32_p_0D5_lambda_5_R0_0D95_F0_0D5/position'+str(int(i/int(total_time)))+'.txt'
		# np.savetxt(filename,position_actin)
		# filename='animation_All_coupled/information_L_32_p_0D5_lambda_5_R0_0D95_F0_0D5/neighbor'+str(int(i/int(total_time)))+'.txt'
		# np.savetxt(filename,Neighbor_actin)
		Plot(position_actin,Neighbor_actin,position_layer,position_DNA,Small_Attraction_DNA,Active_Unit_x,Active_Unit_y,int(i/int(total_time)))
	# if i%10==0:
	# 	#Plot(position_actin,Neighbor_actin,position_layer,position_DNA,Active_Unit_x,Active_Unit_y,int(i/int(total_time)))
	# 	filename='animation_All_coupled/information_L_32_p_0D5_lambda_5_R0_0D95_F0_0D5/position'+str(int(i/10))+'.txt'
	# 	np.savetxt(filename,position_actin)
	# 	filename='animation_All_coupled/information_L_32_p_0D5_lambda_5_R0_0D95_F0_0D5/neighbor'+str(int(i/10))+'.txt'
	# 	np.savetxt(filename,Neighbor_actin)
	# 	Plot(position_actin,Neighbor_actin,position_layer,position_DNA,Active_Unit_x,Active_Unit_y,int(i/10))




# filename='position99.txt'
# position_actin=np.loadtxt(filename)
# filename='neighbor99.txt'
# Neighbor_actin=np.loadtxt(filename)
# Plot(position_actin,Neighbor_actin,position_layer,position_DNA,Active_Unit_x,Active_Unit_y,100)


# for i in range(int(total_time/time_step)):
# 	# position_actin,Neighbor_actin=Update_actin(position_actin,velocity_actin,Neighbor_actin,Active_Unit_x,Active_Unit_y)
# 	#print position_actin
# 	# Update_layer()
# 	# Update_DNA()
# 	if i%int(total_time)==0:
# 		filename='animation/information_L_32_p_0D5_lambda_5_R0_0D95_F0_0D5/position'+str(int(i/int(total_time)))+'.txt'
# 		position_actin=np.loadtxt(filename)
# 		filename='animation/information_L_32_p_0D5_lambda_5_R0_0D95_F0_0D5/neighbor'+str(int(i/int(total_time)))+'.txt'
# 		Neighbor_actin=np.loadtxt(filename)
# 		Plot(position_actin,Neighbor_actin,position_layer,position_DNA,Active_Unit_x,Active_Unit_y,int(i/int(total_time)))
