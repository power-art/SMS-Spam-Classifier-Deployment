
from flask import Flask, render_template, request
import pickle

#Load Multinomial naive Bayes model and CountVectorizer object
#from disk

filename = 'SMS-sPAM-model.pkl'
classifier = pickle.load(open(filename,'rb'))
cv = pickle.load(open('cv-transform.pkl','rb'))


app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
	if request.method == 'POST':
		message = request.form['message']
		data = [message]
		vect = cv.transform(data).toarray()
		myprediction = classifier.predict(vect)
		return render_template('result.html', prediction=myprediction)


if __name__ == '__main__':
		app.run(debug=True)		