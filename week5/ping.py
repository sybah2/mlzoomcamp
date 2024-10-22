from flask import Flask # type: ignore

app = Flask('ping')

@app.route('/ping', methods=['GET'])

def ping():
    return "PUNG"



if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)
