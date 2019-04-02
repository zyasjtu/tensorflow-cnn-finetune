import tensorflow as tf
import sys
from model import ResNetModel
sys.path.insert(0, '../utils')
from preprocessor import BatchPreprocessor
import cv2
import uuid
import numpy as np

batch_size = 128
num_classes = 8
resnet_depth = 50
img_shape = [224, 224, 3]
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

        train2 = open('./train2.txt', 'w')
        valid2 = open('./val2.txt', 'w')
        for i in range(val_batches_per_epoch):
            batch_x, batch_y = val_preprocessor.next_batch(batch_size)
            points = sess.run(prediction, feed_dict={x: batch_x, is_training: False})
            print(i)
            rectify(batch_x, batch_y, points, train2 if i < val_batches_per_epoch/4.0*3 else valid2)

def rectify(batch_x, batch_y, points, f):
    for i in range(len(batch_x)):
        offset_w = 25
        offset_h = 25
        prefix = str(uuid.uuid1())
        batch_x[i] += np.array(mean_color)
        for j in range(4):
            cv2.circle(batch_x[i], (int(points[i][j*2]*img_shape[0]), int(points[i][j*2+1]*img_shape[1])), 1, (0,255,0), 2)
            cv2.circle(batch_x[i], (int(batch_y[i][j*2]*img_shape[0]), int(batch_y[i][j*2+1]*img_shape[1])), 1, (0,0,255), 2)
        cv2.imwrite('./' + prefix + '-0' + '.jpg', batch_x[i])
        for j in range(4):
            if (abs(points[i][j*2]-batch_y[i][j*2])>25):
                continue
            if (abs(points[i][j*2+1]-batch_y[i][j*2+1])>25):
                continue
            x_from = int(max(points[i][j*2] * img_shape[0] - offset_w, 0))
            x_to = int(min(points[i][j*2] * img_shape[0] + offset_w, img_shape[0]))
            y_from = int(max(points[i][j*2+1] * img_shape[1] - offset_h, 0))
            y_to = int(min(points[i][j*2+1] * img_shape[1] + offset_h, img_shape[1]))
            small_img = batch_x[i][y_from:y_to,x_from:x_to]
            batch_y[i][j*2] = batch_y[i][j*2] * img_shape[0] - x_from
            batch_y[i][j*2+1] = batch_y[i][j*2+1] * img_shape[1] - y_from
            padding_img = np.zeros([50,50,3], dtype=np.uint8)
            padding_img[0:small_img.shape[0], 0:small_img.shape[1]] = small_img
            cv2.circle(padding_img, (int(batch_y[i][j*2]), int(batch_y[i][j*2+1])), 1, (255,0,0), 2)
            #cv2.imwrite('./' + prefix + '-' + str(j+1) + '.jpg', padding_img)
            #f.writelines('/volume/source/tensorflow-cnn-finetune-master/data/' + prefix + '-' + str(j+1) + '.jpg,' + str(int(batch_y[i][j*2])) + ',' + str(int(batch_y[i][j*2+1])) + '\n')


if __name__ == '__main__':
    inference(sys.argv[1], sys.argv[2])

