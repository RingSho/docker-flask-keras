from flask import Flask, jsonify
import tensorflow as tf
from mnist2.mnist2 import MnistModel
from keras import backend as K

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4
sess = tf.Session(config=config)
K.set_session(sess)
graph = tf.get_default_graph()
 
app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False



save_dir = "mnist2/"
weight_path = "mnist2/model.h5"
image_path = "images/input/00000.png"
log_path = "logs/"
mnist_model = MnistModel()

@app.route('/')
def index():
    global graph
    with graph.as_default():
        dic = mnist_model.train(save_dir)
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
        "message": "This is mnist2!"
    })

if __name__ == '__main__':
    app.run()