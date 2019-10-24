from flask import Flask
app = Flask(__name__)
from flask import request
from flask import send_file
import crop

@app.route('/')
def home():
    IMO = request.args.get("imo")
    outfile = crop.crop(IMO)
    if outfile:
        return send_file(outfile, mimetype="image/jpg")
    else:
        return None
    
if __name__=="__main__":
    app.run(host='0.0.0.0', debug=True)
