import tensorflow as tf
import numpy as np
import sys
import cv2
from model import ResNetModel


def resize_with_padding(img, output_width, output_height):
	div = 1.0 * output_width / output_height
	input_height, input_width, _ = img.shape

	if (input_width >= div * input_height):
		padding = int((input_width/div-input_height) / 2)
		img = cv2.copyMakeBorder(img, padding, int(input_width/div-input_height-padding), 0, 0, cv2.BORDER_CONSTANT, value=[0,0,0])
	else:
		padding = int((div*input_height-input_width) / 2)
		img = cv2.copyMakeBorder(img, 0, 0, padding, int(input_height*div-input_width-padding), cv2.BORDER_CONSTANT, value=[0,0,0])
	return cv2.resize(img, (output_width, output_height))


def inference(img, model_path, num_classes, resnet_depth=50, mean_color=[132.2766, 139.6506, 146.9702]):
    # Placeholders
    x = tf.placeholder(tf.float32, [1, img.shape[0], img.shape[1], img.shape[2]])
    is_training = tf.placeholder('bool', [])

    # Model
    model = ResNetModel(is_training, depth=resnet_depth, num_classes=num_classes)
    prediction = model.inference(x)

    # Subtract mean color
    img = img.astype(np.float32)
    img -= np.array(mean_color)
    batch_x = np.ndarray([1, img.shape[0], img.shape[1], img.shape[2]])
    batch_x[0] = img

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        # 
        sess.run(tf.global_variables_initializer())
        # Directly restore
        tf.train.Saver().restore(sess, model_path)
        points = sess.run(prediction, feed_dict={x: batch_x, is_training: False})
        rectify(batch_x, points, mean_color=[132.2766, 139.6506, 146.9702])


def rectify(batch_x, points, mean_color=[132.2766, 139.6506, 146.9702]):
    for i in range(len(batch_x)):
        predict = batch_x[i] + np.array(mean_color)
        result = predict.copy()
        pts = points[i].reshape(4, 2) * [img.shape[0], img.shape[1]]
        for j in range(4):
            cv2.circle(predict, (int(pts[j][0]+0.5), int(pts[j][1]+0.5)), 1, (0,255,0), 2)
        cv2.imwrite('./predict.jpg', predict)
        
        origin = np.float32([pts[0], pts[1], pts[2], pts[3]])
        perspected = np.float32([ [0,0], [175,0], [0,110], [175,110] ])
        matrix = cv2.getPerspectiveTransform(origin, perspected)
	result = cv2.warpPerspective(result, matrix, (175, 110))
        cv2.imwrite('./result.jpg', result)


if __name__ == '__main__':
	img = resize_with_padding(cv2.imread(sys.argv[1]), 224, 224)
	inference(img, sys.argv[2], 8)
