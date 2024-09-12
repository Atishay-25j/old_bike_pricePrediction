from flask import Flask, request
from flask_cors import CORS, cross_origin
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)
CORS(app)


bike=pickle.load(open('linear_regression_model.pkl','rb'))
app=Flask(__name__)

@app.route('/predictB', methods=['POST'])
@cross_origin()
def predict():
    
        data = request.get_json()
        app.logger.info(data)
        kms_driven=int(data.get('Driven'))   
        owner=int(data.get('Owner'))
        age=int(data.get('Age'))
        power=int(data.get('Power'))
        brand=data.get('Brand_name')
        Brand = 0   
        
        if (brand=='Royal Enfield'):  
            Brand=1
        elif(brand=='KTM'):
            Brand=2
        elif(brand=='Bajaj'):
            Brand=3
        elif(brand=='Harley'):
            Brand=4
        elif(brand=='Yamaha'):
            Brand=5
        elif(brand=='Honda'):
            Brand=6
        elif(brand=='Suzuki'):
            Brand=7
        elif(brand=='TVS'):
            Brand=8
        elif(brand=='Kawasaki'):
            Brand=9
        elif(brand=='Hyosung'):
            Brand=10
        elif(brand=='Benelli'):
            Brand=11
        elif(brand=='Mahindra'):
            Brand=12
        elif(brand=='Triumph'):
            Brand=13
        elif(brand=='Ducati'):
            Brand=14
        elif(brand=='BMW'):
            Brand=15

        new_data = {
    'kms_driven': [kms_driven],
    'owner': [owner],
    'age': [age],
    'power': [power],
    'brand': [Brand]
}
        app.logger.info(new_data)

        new_data_df = pd.DataFrame(new_data)
        predictions = bike.predict(new_data_df)
        print(predictions[0])

        return str(np.round(predictions[0], 2))

if __name__ == "__main__":
    app.run(debug=True)



        