import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# load the data file
N_times = 50 #Poner acá el tiempo (int)
plt.ion()
for i in range(1,N_times + 1):
	data_file = pd.read_csv('/home/victor/Cosas_Importantes/Charlas/Taller_HPC/examples/lines_'+str(i)+'.csv')

	position = data_file['Position[m]']
	phis = data_file.loc[:,'phi1':'phi2']

	plt.plot(phis['phi1'],position,'r')
	plt.plot(phis['phi2'],position,'g')
	plt.ylim(top=1.0)
	plt.ylim(bottom=0.0)
	plt.legend(['phi1','phi2'])
	plt.xlabel('Position (m)')
	plt.ylabel('Concentration')
	plt.show()
	plt.pause(0.1)
	plt.clf()
