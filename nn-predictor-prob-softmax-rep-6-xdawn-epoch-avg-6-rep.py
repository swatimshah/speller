import numpy
import math 
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from imblearn.over_sampling import SMOTE
from numpy import savetxt
from numpy import loadtxt
from numpy import asarray
from sklearn.preprocessing import StandardScaler
from numpy.random import seed
import pickle
from keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from keras import backend as K 
import gc
import time
import datetime
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
import copy
from sklearn.ensemble import RandomForestClassifier
import touch
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn import preprocessing

class MyOVBox(OVBox):
	def __init__(self):
		OVBox.__init__(self)
		self.signalHeader = None
		self.stim = None
		self.stimulationHeader = None
		self.data = numpy.zeros((1, 1, 152))
		self.flashes = numpy.zeros((1, 152))
		self.m_stim = numpy.zeros((1,7))
		self.m_stimulation = numpy.zeros((1,1))
		self.count0 = 0
		self.count1 = 0	
		self.count2 = 0
		self.flashCount = 0
		self.stimCount = 0
		self.dimensionSizes = [1, 2]
		self.dimensionLabels = ["channels", "predictions"]
		self.m_flashstim = numpy.zeros((1,7))	
		touch.touch("d:\\model.h5")
		self.model = load_model('d:\\model.h5')

		#------------------------------------------------		

		seed(1)

	def initialize(self):
		#outputHeader = OVStreamedMatrixHeader(self.getCurrentTime(), self.getCurrentTime(), self.dimensionSizes, self.dimensionLabels)
		#self.output[0].append(outputHeader)					
		print("initialize")


	def uninitialize(self):
		#self.f.close()
		print("uninitialize")


	def process(self):


		for chunkIndex0 in range( len(self.input[0]) ):
			chunk0 = self.input[0].pop()
			if(type(chunk0) == OVStimulationSet):
				for stimIdx in range(len(chunk0)):
					self.stim=chunk0.pop();
					if(self.stim.identifier >= 33025 and self.stim.identifier <= 33036):
						if(self.stim.identifier == 33025.):
							self.m_flashstim = numpy.insert(self.m_flashstim, self.count2, numpy.array([[1, 2, 3, 4, 5, 6, self.stim.identifier]]), axis=0)	
						if(self.stim.identifier == 33026.):
							self.m_flashstim = numpy.insert(self.m_flashstim, self.count2, numpy.array([[7, 8, 9, 10, 11, 12, self.stim.identifier]]), axis=0)	
						if(self.stim.identifier == 33027.):
							self.m_flashstim = numpy.insert(self.m_flashstim, self.count2, numpy.array([[13, 14, 15, 16, 17, 18, self.stim.identifier]]), axis=0)	
						if(self.stim.identifier == 33028.):
							self.m_flashstim = numpy.insert(self.m_flashstim, self.count2, numpy.array([[19, 20, 21, 22, 23, 24, self.stim.identifier]]), axis=0)	
						if(self.stim.identifier == 33029.):
							self.m_flashstim = numpy.insert(self.m_flashstim, self.count2, numpy.array([[25, 26, 27, 28, 29, 30, self.stim.identifier]]), axis=0)	
						if(self.stim.identifier == 33030.):
							self.m_flashstim = numpy.insert(self.m_flashstim, self.count2, numpy.array([[31, 32, 33, 34, 35, 36, self.stim.identifier]]), axis=0)	

						if(self.stim.identifier == 33031.):
							self.m_flashstim = numpy.insert(self.m_flashstim, self.count2, numpy.array([[1, 7, 13, 19, 25, 31, self.stim.identifier]]), axis=0)	
						if(self.stim.identifier == 33032.):
							self.m_flashstim = numpy.insert(self.m_flashstim, self.count2, numpy.array([[2, 8, 14, 20, 26, 32, self.stim.identifier]]), axis=0)	
						if(self.stim.identifier == 33033.):
							self.m_flashstim = numpy.insert(self.m_flashstim, self.count2, numpy.array([[3, 9, 15, 21, 27, 33, self.stim.identifier]]), axis=0)	
						if(self.stim.identifier == 33034.):
							self.m_flashstim = numpy.insert(self.m_flashstim, self.count2, numpy.array([[4, 10, 16, 22, 28, 34, self.stim.identifier]]), axis=0)	
						if(self.stim.identifier == 33035.):
							self.m_flashstim = numpy.insert(self.m_flashstim, self.count2, numpy.array([[5, 11, 17, 23, 29, 35, self.stim.identifier]]), axis=0)	
						if(self.stim.identifier == 33036.):
							self.m_flashstim = numpy.insert(self.m_flashstim, self.count2, numpy.array([[6, 12, 18, 24, 30, 36, self.stim.identifier]]), axis=0)	

						m_flashstim_rows = len(self.m_flashstim[:])
						print(m_flashstim_rows)

						if(m_flashstim_rows == 75):
							self.m_stim = numpy.append(self.m_stim, self.m_flashstim[0:72,:], axis=0)
							self.m_stim = numpy.delete(self.m_stim,numpy.where(~self.m_stim.any(axis=1))[0], axis=0)
							print("m_stim length = ", len(self.m_stim[:]))
							self.m_flashstim = numpy.zeros((1,7))	
							m_flashstim_rows = 0															

						#self.m_stim = numpy.insert(self.m_stim, self.count0, self.m_flashstim, axis=0)
						#self.m_stimulation = numpy.insert(self.m_stimulation, self.count0, self.stim.identifier, axis=0) 	
					#else:
						#print(self.stim.identifier)	




		for chunkIndex1 in range( len(self.input[1]) ):
			chunk1 = self.input[1].pop()
			if(type(chunk1) == OVSignalHeader):
				self.signalHeader = chunk1
			elif(type(chunk1) == OVSignalBuffer):	
				self.data = numpy.array(chunk1).reshape(1, 152) 
				self.flashes = numpy.insert(self.flashes, self.count1, self.data, axis=0) 
				self.flashCount = self.flashCount + 1
				print("flashARCount = ", self.flashCount)			
			else:
				chunk1


		if(len(self.m_stim[:]) == 72 and self.flashCount == 72):
			self.flashes = numpy.delete(self.flashes,numpy.where(~self.flashes.any(axis=1))[0], axis=0)
			self.m_stim = self.m_stim = self.m_stim[0:72,:]
			self.flashes = numpy.append(self.flashes, self.m_stim, axis=1)

			print("started predicting...\n")			

			input = preprocessing.minmax_scale(self.flashes[:, 0:152], axis=1)
			fileName = time.strftime("%Y%m%d-%H%M%S")	
			savetxt('d:\\' + fileName + '.csv', input, delimiter=',')

			input = input.reshape(len(input), 4, 38)
			input = input.transpose(0, 2, 1)

			prediction_result_temp = self.model.predict_classes(input)
			print(prediction_result_temp)
						
			prediction_result = prediction_result_temp.astype(int)

			m_combination = numpy.append(prediction_result.reshape(72, 1), self.flashes[:, -1].reshape(72, 1), axis=1)

			print(m_combination)

			print("\n")

			m_combination_sorted = m_combination[numpy.argsort(m_combination[:, 1])]
	
			print(m_combination_sorted)

			list0 = [0,1,2,3,4,5] 
			list1 = [6,7,8,9,10,11]
			list2 = [12,13,14,15,16,17]
			list3 = [18,19,20,21,22,23]
			list4 = [24,25,26,27,28,29]
			list5 = [30,31,32,33,34,35]
			list6 = [36,37,38,39,40,41] 
			list7 = [42,43,44,45,46,47]
			list8 = [48,49,50,51,52,53]
			list9 = [54,55,56,57,58,59]
			list10 = [60,61,62,63,64,65]
			list11 = [66,67,68,69,70,71]

			b = numpy.zeros((12,2))

			b[0,:] = numpy.sum(m_combination_sorted[list0,:],axis=0)
			b[1,:] = numpy.sum(m_combination_sorted[list1,:],axis=0)
			b[2,:] = numpy.sum(m_combination_sorted[list2,:],axis=0)
			b[3,:] = numpy.sum(m_combination_sorted[list3,:],axis=0)
			b[4,:] = numpy.sum(m_combination_sorted[list4,:],axis=0)
			b[5,:] = numpy.sum(m_combination_sorted[list5,:],axis=0)
			b[6,:] = numpy.sum(m_combination_sorted[list6,:],axis=0)
			b[7,:] = numpy.sum(m_combination_sorted[list7,:],axis=0)
			b[8,:] = numpy.sum(m_combination_sorted[list8,:],axis=0)
			b[9,:] = numpy.sum(m_combination_sorted[list9,:],axis=0)
			b[10,:] = numpy.sum(m_combination_sorted[list10,:],axis=0)
			b[11,:] = numpy.sum(m_combination_sorted[list11,:],axis=0)
				
			sorted_b = b[b[:,0].argsort()[::-1]]
			
			col_b = numpy.true_divide(sorted_b[:,1], 6).reshape(12, 1)

			print(sorted_b.shape)	
			print(col_b.shape)

			m_final_table = numpy.append(sorted_b, col_b, axis=1)
			print(m_final_table)

			stimSetRow = OVStimulationSet(self.getCurrentTime(), self.getCurrentTime())
			stimSetCol = OVStimulationSet(self.getCurrentTime(), self.getCurrentTime())
				
			selected_row = 0
			selected_col = 0
			i = 0
			j = 0

			for i in range (12):
				if (m_final_table[i, 2] >= 33025 and m_final_table[i, 2] <= 33030):
					selected_row = m_final_table[i, 2]
					break;	

			for j in range (12):					
				if (m_final_table[j, 2] >= 33031 and m_final_table[j, 2] <= 33036):
					selected_col = m_final_table[j, 2]
					break;	

			stimSetRow.append(OVStimulation(math.ceil(selected_row), self.getCurrentTime(), 0.))
			stimSetCol.append(OVStimulation(math.ceil(selected_col), self.getCurrentTime(), 0.))

			self.output[0].append(stimSetRow)
			self.output[1].append(stimSetCol)


			self.data = numpy.zeros((1, 1, 152))
			self.flashes = numpy.zeros((1, 152))
			self.m_stim = numpy.zeros((1,7))
			self.m_stimulation = numpy.zeros((1,1))
			self.count0 = 0
			self.count1 = 0	
			self.count2 = 0
			self.flashCount = 0
			self.stimCount = 0	

box = MyOVBox()						