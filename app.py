from flask import Flask, render_template, request
#import jsonify
import requests
import pickle
import numpy as np
import sklearn
import numpy as np

app = Flask(__name__)
model = pickle.load(open('RandomForestClassifier_Model.pkl', 'rb'))

segment_encoder = pickle.load(open('segment_encoder.pkl', 'rb'))
market_encoder = pickle.load(open('market_encoder.pkl', 'rb'))
region_encoder = pickle.load(open('region_encoder.pkl', 'rb'))
category_encoder = pickle.load(open('category_encoder.pkl', 'rb'))
order_priority_encoder = pickle.load(open('order_priority_encoder.pkl', 'rb'))

@app.route('/',methods=['GET'])

def Home():
    return render_template('index.html')

@app.route("/classify", methods=['POST'])

def classify():

    if request.method == 'POST':
        Segment_ = request.form['segment_']
        Segment_ = segment_encoder.transform([Segment_])

        Market_ = request.form['market_']
        Market_ = market_encoder.transform([Market_])

        Region_ = request.form['region_']
        Region_ = region_encoder.transform([Region_])

        Category_ = request.form['category_']
        Category_ = category_encoder.transform([Category_])

        OrderPriority_ = request.form['order_priority_']
        OrderPriority_ = order_priority_encoder.transform([OrderPriority_])


        Costprice = round(float(request.form['Cost_Price']),3)
        Costprice = np.log(Costprice)

        Quantity = int(request.form['Quantity'])

        DiscountPercent = round(float(request.form['Discount_Percent']),2)

        ShippingCost = round(float(request.form['Shipping_Cost']),2)
        ShippingCost = np.log(ShippingCost)

        DaysToShip = int(request.form['Days_to_Ship'])


        prediction = model.predict([[Segment_,Market_,Region_,Category_,OrderPriority_,Costprice,Quantity,
                                    DiscountPercent,ShippingCost,DaysToShip]])

        

        if prediction==1:
            return render_template('index.html',prediction_text="Highly Contributing")
        elif prediction==2:
            return render_template('index.html',prediction_text="Above Average Contributing")
        elif prediction==3:
            return render_template('index.html',prediction_text="Below Average Contributing")
        else:
            return render_template('index.html',prediction_text="Least Contributing")
    else:
        return render_template('index.html')

if __name__=="__main__":
    app.run(debug=False)