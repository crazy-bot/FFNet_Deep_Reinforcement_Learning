import  os
import cv2
import scipy.io as io
import  numpy as np
from alexnet import AlexNet
import tensorflow as tf

if __name__ == '__main__':


    x = tf.placeholder(tf.float32, [1, 227, 227, 3])
    keep_prob = tf.placeholder(tf.float32)

    # create model with default config ( == no skip_layer and 1000 units in the last layer)
    model = AlexNet(x, keep_prob, 1000, [])

    with tf.Session() as sess:
        # need dict to save as mat file
        # Initialize all variables
        sess.run(tf.global_variables_initializer())

        # Load the pretrained weights into the model
        model.load_initial_weights(sess)
        dict = {}
        feat = []
        files = os.listdir('/data/Guha/Tour20/frames/MP7.mp4')
        imagenet_mean = np.array([104., 117., 124.], dtype=np.float32)
        for idx in range(0, len(files) - 1):
            framePath = os.path.join('/data/Guha/Tour20/frames/MP7.mp4', 'frame' + str(idx) + '.jpg')
            frame = cv2.imread(framePath)
            # pil = Image.open(framePath)
            if frame is None:
                continue
            frame = cv2.resize(frame.astype(np.float32), (227, 227))
            frame -= imagenet_mean
            frame = frame.reshape(1,227, 227,3)
            output = sess.run(model.fc7, feed_dict={x: frame,keep_prob: 1})
            output = np.asarray(output).reshape(4096)
            feat.append(output)

        dict['features'] = feat

        io.savemat('input2/MP7_alex_fc7_feat', dict)
