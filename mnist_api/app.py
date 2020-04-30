from flask import Flask, jsonify
import tensorflow as tf
from mnist.mnist import MnistModel
 
 
app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
graph = tf.get_default_graph()


save_dir = "mnist/"
weight_path = "mnist/model.h5"
image_path = "images/input/5_test.png"
log_path = "logs/"
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
        with open(log_path + "log.txt", mode = "a") as f:
            print(f.write(str(result) + "\n"))
        return jsonify(result)

@app.route('/hello')
def hello():
    return jsonify({
        "message": "This is mnist!"
    })
 
if __name__ == '__main__':
    app.run()