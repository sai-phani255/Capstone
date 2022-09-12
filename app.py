from flask import Flask, render_template, request
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
        Segment = request.form['segment_']
        Segment_ = segment_encoder.transform([Segment])

        Market = request.form['market_']
        Market_ = market_encoder.transform([Market])

        Region = request.form['region_']
        Region_ = region_encoder.transform([Region])

        Category = request.form['category_']
        Category_ = category_encoder.transform([Category])

        OrderPriority = request.form['order_priority_']
        OrderPriority_ = order_priority_encoder.transform([OrderPriority])


        Costprice = round(float(request.form['Cost_Price']),3)
        Costprice_ = np.log(Costprice)

        Quantity = int(request.form['Quantity'])

        DiscountPercent = round(float(request.form['Discount_Percent']),2)

        ShippingCost = round(float(request.form['Shipping_Cost']),2)
        ShippingCost_ = np.log(ShippingCost)

        DaysToShip = int(request.form['Days_to_Ship'])


        prediction = model.predict([[Segment_,Market_,Region_,Category_,OrderPriority_,Costprice_,Quantity,
                                    DiscountPercent,ShippingCost_,DaysToShip]])
        

        if prediction==1:
            return render_template('result.html',prediction_text="Highly Contributing",
            costprice=Costprice,quantity=Quantity,disc_percent=DiscountPercent,shippingcost=ShippingCost,
            daystoship=DaysToShip,segment=Segment,market=Market,region=Region,category=Category,
            orderpriority=OrderPriority)

        elif prediction==2:
            return render_template('result.html',prediction_text="Above Average Contributing",
            costprice=Costprice,quantity=Quantity,disc_percent=DiscountPercent,shippingcost=ShippingCost,
            daystoship=DaysToShip,segment=Segment,market=Market,region=Region,category=Category,
            orderpriority=OrderPriority)
        elif prediction==3:
            return render_template('result.html',prediction_text="Below Average Contributing",
            costprice=Costprice,quantity=Quantity,disc_percent=DiscountPercent,shippingcost=ShippingCost,
            daystoship=DaysToShip,segment=Segment,market=Market,region=Region,category=Category,
            orderpriority=OrderPriority)
        else:
            return render_template('result.html',prediction_text="Least Contributing",
            costprice=Costprice,quantity=Quantity,disc_percent=DiscountPercent,shippingcost=ShippingCost,
            daystoship=DaysToShip,segment=Segment,market=Market,region=Region,category=Category,
            orderpriority=OrderPriority)
    #else:
    #    return render_template('index.html')

if __name__=="__main__":
    app.run(debug=False)
