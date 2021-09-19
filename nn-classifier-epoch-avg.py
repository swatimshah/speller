import numpy
from keras.models import Sequential, save_model, load_model
from keras.layers import Dense, Dropout, Flatten
from keras.losses import sparse_categorical_crossentropy
from keras.optimizers import SGD
from imblearn.over_sampling import SMOTE
from numpy import savetxt
#import pickle
from sklearn import preprocessing

class MyOVBox(OVBox):
	def __init__(self):
		OVBox.__init__(self)
		self.signalHeader = None
		self.stim = None
		self.stimulationHeader = None
		self.count = 0
		self.count2 = 0
		self.Target1 = 0
		self.Target2 = 0
		self.Target = 0
		self.TargetRow = 0
		self.TargetCol = 0
		self.avgData = numpy.zeros((1, 153))	
		self.m_stim = numpy.zeros((1,7))
		self.m_flashstim = numpy.zeros((1,7))	
		self.keepTrack = 0
		self.charCount = 1

	def initialize(self):
		print("initialize")
		self.output[0].append(OVStimulationHeader(self.getCurrentTime(), self.getCurrentTime()))
		print("initialize complete")

	def uninitialize(self):
		self.m_stim = numpy.delete(self.m_stim,numpy.where(~self.m_stim.any(axis=1))[0], axis=0)
		print(self.m_stim.shape)	
		self.avgData = numpy.delete(self.avgData,numpy.where(~self.avgData.any(axis=1))[0], axis=0)
		separateTargetIdentity = self.avgData[:, 152]
		print(separateTargetIdentity.shape)
		print(self.avgData.shape)

		self.combinedData = numpy.append(self.avgData[:, 0:152], self.m_stim, axis=1)

		self.integratedData = numpy.append(self.combinedData, separateTargetIdentity.reshape(len(separateTargetIdentity), 1), axis=1)
		print(self.integratedData.shape)

		# save to csv file
		savetxt('d:\\table_of_flashes.csv', self.integratedData, delimiter=',')		

	def process(self):

		for chunkIndex0 in range( len(self.input[0]) ):
			chunk0 = self.input[0].pop()
			if(type(chunk0) == OVSignalHeader):
				self.signalHeader = chunk0
				self.data = numpy.zeros((1,self.signalHeader.dimensionSizes[1]))	
			elif(type(chunk0) == OVSignalBuffer):	
				print(numpy.array(chunk0).shape)				
				self.data = numpy.array(chunk0).reshape(1, 152) 
				self.data = numpy.append(self.data[-1, :], 1)	
				self.data.reshape(1, 153)			
				self.avgData = numpy.insert(self.avgData, self.count, self.data, axis=0) 	
			else:
				chunk0


		for chunkIndex1 in range( len(self.input[1]) ):
			chunk1 = self.input[1].pop()
			if(type(chunk1) == OVSignalHeader):
				self.signalHeader = chunk1
				self.data1 = numpy.empty((1,self.signalHeader.dimensionSizes[1]))
			elif(type(chunk1) == OVSignalBuffer):				
				self.data1 = numpy.array(chunk1).reshape(1, 152)
				self.data1 = numpy.append(self.data1[-1, :], 0)
				self.data1.reshape(1, 153)						
				self.avgData = numpy.insert(self.avgData, self.count, self.data1, axis=0) 	
			else:
				chunk1



		for chunkIndex in range( len(self.input[2]) ):
			chunk = self.input[2].pop()
			print("\n")	
			if(type(chunk) == OVStimulationSet):
				for stimIdx in range(len(chunk)):
					self.stim=chunk.pop();
					print(self.stim.identifier)
	
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

						if(m_flashstim_rows == 2):
							self.Target1 = self.m_flashstim[0][6]
							print("Target - 1", self.Target1)
							if(self.Target1 >= 33025. and self.Target1 <= 33030.):	
								self.TargetRow = self.Target1 					
							if(self.Target1 >= 33031. and self.Target1 <= 33036.):	
								self.TargetCol = self.Target1 					

						if(m_flashstim_rows == 3):
							self.Target2 = self.m_flashstim[0][6]
							print("Target - 2", self.Target2)
							if(self.Target2 >= 33025. and self.Target2 <= 33030.):	
								self.TargetRow = self.Target2 					
							if(self.Target2 >= 33031. and self.Target2 <= 33036.):	
								self.TargetCol = self.Target2						
							if(self.TargetRow == 33025. and self.TargetCol == 33031.):
								self.Target = 1	
							if(self.TargetRow == 33025. and self.TargetCol == 33032.):
								self.Target = 2	
							if(self.TargetRow == 33025. and self.TargetCol == 33033.):
								self.Target = 3	
							if(self.TargetRow == 33025. and self.TargetCol == 33034.):
								self.Target = 4	
							if(self.TargetRow == 33025. and self.TargetCol == 33035.):
								self.Target = 5	
							if(self.TargetRow == 33025. and self.TargetCol == 33036.):
								self.Target = 6	
							if(self.TargetRow == 33026. and self.TargetCol == 33031.):
								self.Target = 7	
							if(self.TargetRow == 33026. and self.TargetCol == 33032.):
								self.Target = 8	
							if(self.TargetRow == 33026. and self.TargetCol == 33033.):
								self.Target = 9	

							if(self.TargetRow == 33026. and self.TargetCol == 33034.):
								self.Target = 10	
							if(self.TargetRow == 33026. and self.TargetCol == 33035.):
								self.Target = 11	
							if(self.TargetRow == 33026. and self.TargetCol == 33036.):
								self.Target = 12	
							if(self.TargetRow == 33027. and self.TargetCol == 33031.):
								self.Target = 13	
							if(self.TargetRow == 33027. and self.TargetCol == 33032.):
								self.Target = 14	
							if(self.TargetRow == 33027. and self.TargetCol == 33033.):
								self.Target = 15	
							if(self.TargetRow == 33027. and self.TargetCol == 33034.):
								self.Target = 16	
							if(self.TargetRow == 33027. and self.TargetCol == 33035.):
								self.Target = 17	
							if(self.TargetRow == 33027. and self.TargetCol == 33036.):
								self.Target = 18	

							if(self.TargetRow == 33028. and self.TargetCol == 33031.):
								self.Target = 19	
							if(self.TargetRow == 33028. and self.TargetCol == 33032.):
								self.Target = 20	
							if(self.TargetRow == 33028. and self.TargetCol == 33033.):
								self.Target = 21	
							if(self.TargetRow == 33028. and self.TargetCol == 33034.):
								self.Target = 22
							if(self.TargetRow == 33028. and self.TargetCol == 33035.):
								self.Target = 23	
							if(self.TargetRow == 33028. and self.TargetCol == 33036.):
								self.Target = 24	
							if(self.TargetRow == 33029. and self.TargetCol == 33031.):
								self.Target = 25	
							if(self.TargetRow == 33029. and self.TargetCol == 33032.):
								self.Target = 26	
							if(self.TargetRow == 33029. and self.TargetCol == 33033.):
								self.Target = 27	

							if(self.TargetRow == 33029. and self.TargetCol == 33034.):
								self.Target = 28	
							if(self.TargetRow == 33029. and self.TargetCol == 33035.):
								self.Target = 29	
							if(self.TargetRow == 33029. and self.TargetCol == 33036.):
								self.Target = 30	
							if(self.TargetRow == 33030. and self.TargetCol == 33031.):
								self.Target = 31	
							if(self.TargetRow == 33030. and self.TargetCol == 33032.):
								self.Target = 32	
							if(self.TargetRow == 33030. and self.TargetCol == 33033.):
								self.Target = 33	
							if(self.TargetRow == 33030. and self.TargetCol == 33034.):
								self.Target = 34	
							if(self.TargetRow == 33030. and self.TargetCol == 33035.):
								self.Target = 35	
							if(self.TargetRow == 33030. and self.TargetCol == 33036.):
								self.Target = 36	

						if(m_flashstim_rows == 75):
							self.m_stim = numpy.append(self.m_stim, self.m_flashstim[0:72,:], axis=0)
							self.m_flashstim = numpy.zeros((1,7))	
							#self.TargetRow = 0
							#self.TargetCol = 0
							#self.Target = 0	
							m_flashstim_rows = 0																		
				
					elif(self.stim.identifier == 32770):
						stimSet = OVStimulationSet(self.getCurrentTime(), self.getCurrentTime())
						stimSet.append(OVStimulation(33287, self.getCurrentTime(), 0.))
						self.output[0].append(stimSet)
					else:
						chunk
						#print(self.stim.identifier)





		end = self.getCurrentTime()
		self.output[0].append(OVStimulationEnd(end, end))				
															
box = MyOVBox()