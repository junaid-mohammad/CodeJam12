from flask import Flask, render_template, request

app = Flask(__name__)


@app.route('/')
@app.route('/home')
def home():
    return render_template("index.html")


@app.route('/launch')
def launch():
    celsius = request.args.get("celsius", "")
    return fahrenheit_from()
    # return render_template("launch.html", celsius="celsius")


@app.route('/launch', methods=['POST', 'GET'])
def fahrenheit_from():
    """Convert Celsius to Fahrenheit degrees."""
    try:
        fahrenheit = float(42) * 9 / 5 + 32
        fahrenheit = round(fahrenheit, 3)  # Round to three decimal places
        return str(fahrenheit)
    except ValueError:
        return "invalid input"


# @app.route("/run", methods=['POST', "GET"])
# def run():
#     output = request.form.to_dict()
#     launch = output["launch"]
#
#     return render_template("index.html", launch="launch")


if __name__ == "__main__":
    app.run(debug=True, port=5001)
