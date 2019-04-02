import tensorflow as tf
import sys
from model import ResNetModel
sys.path.insert(0, '../utils')
from preprocessor import BatchPreprocessor
import cv2
import uuid
import numpy as np

batch_size = 128
num_classes = 2
resnet_depth = 50
img_shape = [50, 50, 3]
mean_color=[132.2766, 139.6506, 146.9702]

def inference(model_path, test_list):
    # Placeholders
    x = tf.placeholder(tf.float32, [batch_size, img_shape[0], img_shape[1], img_shape[2]])
    is_training = tf.placeholder('bool', [])

    # Model
    model = ResNetModel(is_training, depth=resnet_depth, num_classes=num_classes)
    prediction = model.inference(x)
    
    val_preprocessor = BatchPreprocessor(dataset_file_path=test_list, num_classes=num_classes, output_size=img_shape[0:2], shuffle=True)
    val_batches_per_epoch = np.floor(len(val_preprocessor.labels) / batch_size).astype(np.int16)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        # Directly restore
        tf.train.Saver().restore(sess, model_path)

        for i in range(val_batches_per_epoch):
            batch_x, batch_y = val_preprocessor.next_batch(batch_size)
            points = sess.run(prediction, feed_dict={x: batch_x, is_training: False})
            print(i)
            rectify(batch_x, batch_y, points)

def rectify(batch_x, batch_y, points):
    for i in range(len(batch_x)):
        prefix = str(uuid.uuid1())
        batch_x[i] += np.array(mean_color)
        cv2.circle(batch_x[i], (int(points[i][0]*img_shape[0]), int(points[i][1]*img_shape[1])), 1, (0,255,0), 2)
        cv2.circle(batch_x[i], (int(batch_y[i][0]*img_shape[0]), int(batch_y[i][1]*img_shape[1])), 1, (0,0,255), 2)
        cv2.imwrite('./' + prefix + '.jpg', batch_x[i])


if __name__ == '__main__':
    inference(sys.argv[1], sys.argv[2])

