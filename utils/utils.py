import matplotlib.pyplot as plt

def plot_loss(history):
	# Loss Curves
	try:
		plt.figure(figsize=[8,6])
		plt.plot(history['loss'],'r',linewidth=3.0)
		plt.plot(history['val_loss'],'b',linewidth=3.0)
		plt.legend(['Training loss', 'Validation Loss'],fontsize=18)
		plt.xlabel('Epochs ',fontsize=16)
		plt.ylabel('Loss',fontsize=16)
		plt.title('Loss Curves',fontsize=16)
		plt.show()
	except:
		print("expected keras History.history object. Got " + repr(type(history)))
	return

def openfig():
	plt.figure(figsize=[8,6])
	plt.xlabel('hello')
	plt.show()
	return
