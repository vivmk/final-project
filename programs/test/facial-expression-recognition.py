from keras.preprocessing import image

import numpy as np
import matplotlib.pyplot as plt

#face expression recognizer initialization
from keras.models import model_from_json
model = model_from_json(open("c:/final-project-aug/json/facial_expression_model_structure.json", "r").read())
model.load_weights('c:/final-project-aug/weights/facial_expression_model_weights.h5') #load weights

#------------------------------
#function for drawing bar chart for emotion preditions
def emotion_analysis(emotions):
	objects = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
	y_pos = np.arange(len(objects))
	
	plt.bar(y_pos, emotions, align='center', alpha=0.5)
	plt.xticks(y_pos, objects)
	plt.ylabel('percentage')
	plt.title('emotion')
	
	plt.show()
#------------------------------

monitor_testset_results = False

if monitor_testset_results == True:
	#make predictions for test set
	predictions = model.predict(x_test)

	index = 0
	for i in predictions:
		if index < 30 and index >= 20:
			#print(i) #predicted scores
			#print(y_test[index]) #actual scores
			
			testing_img = np.array(x_test[index], 'float32')
			testing_img = testing_img.reshape([48, 48]);
			
			plt.gray()
			plt.imshow(testing_img)
			plt.show()
			
			print(i)
			
			emotion_analysis(i)
			print("----------------------------------------------")
			index = index + 1

#------------------------------
#make prediction for custom image out of test set

img = image.load_img("c:/final-project-aug/dataset/jackman.png", grayscale=True, target_size=(48, 48))

x = image.img_to_array(img)
x = np.expand_dims(x, axis = 0)

x /= 255

custom = model.predict(x)
emotion_analysis(custom[0])

x = np.array(x, 'float32')
x = x.reshape([48, 48]);

plt.show()
#------------------------------