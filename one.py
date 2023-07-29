import numpy as np
from flask import Flask, request, render_template
import pickle

# app = Flask(__name__)
app = Flask(__name__, template_folder='template')
model = pickle.load(open('model.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():

    features = [float(x) for x in request.form.values()]
    final_features = [np.array(features)]
    prediction = model.predict(final_features)

    return render_template('index.html', prediction_text='Species = {}'.format(prediction[0]))


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=3000)



# from flask import Flask,render_template
# app=Flask(__name__,template_folder='template')
# @app.route("/")
# def Hello():
#     return render_template('index.html')


# if __name__ == "__main__":
#     app.run(debug=True,host='0.0.0.0',port=3000)

