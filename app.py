from flask import Flask, render_template, request, url_for
import train_linear_regession as model

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/forecast',methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        try:
            open_price = float(request.form['open'])
            high_price = float(request.form['high'])
            low_price = float(request.form['low'])
            volume = float(request.form['volume'])
            prediction = model.predict_price(open_price, high_price, low_price, volume)
        except ValueError:
            prediction = "Vui lòng nhập số hợp lệ!"
    return render_template('forecast.html', prediction=prediction)


if __name__ == '__main__':
    app.run(debug=True)