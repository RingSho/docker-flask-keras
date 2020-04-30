from flask import Flask, jsonify
import tensorflow as tf
from mnist2.mnist2 import MnistModel
 
 
app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
graph = tf.get_default_graph()


save_dir = "mnist2/"
weight_path = "mnist2/model.h5"
image_path = "5_test.png"
mnist_model = MnistModel()

@app.route('/')
def index():
    global graph
    with graph.as_default():
        dic = mnist_model.train(save_dir).json()
        return jsonify(dic)

@app.route('/predict')
def predict():
    global graph
    with graph.as_default():
        result = mnist_model.predict(weight_path, image_path)
        print(result)
        return jsonify(result)

@app.route('/hello')
def hello():
    return jsonify({
        "message": "This is mnist2!"
    })
 
if __name__ == '__main__':
    app.run()