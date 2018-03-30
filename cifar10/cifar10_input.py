# coding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

# Process images of this size. Note that this differs from the original CIFAR
# image size of 32 x 32. If one alters this number, then the entire model
# architecture will change and any model would need to be retrained.
IMAGE_SIZE = 24

# Global constants describing the CIFAR-10 data set.
NUM_CLASSES = 10
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000

def read_cifar10(filename_queue):

    class CIFAR10Record(object):
        pass

    result = CIFAR10Record()

    label_bytes = 1
    result.height = 32
    result.width = 32
    result.depth = 3

    image_bytes = result.height * result.depth 
    record_bytes = label_bytes + image_bytes 
    
    # 各レコードがバイトの固定数であるbinary fileを読む
    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
    result.key, value = reader.read(filename_queue)

    # convert from a string to a vector of uint8 that is record_bytes long.
    record_bytes = tf.decode_raw(value,tf.uint8)

    # the first bytes represent the label, which we convert from uint8 -> int32
    result.label = tf.cast(tf.strided_slice(record_bytes, [0], [label_bytes]), tf.int32)

    # the remaining bytes after the label represent the image, which we reshape 
    # from [depth*height*width] to [depth, height, width]
    depth_major = tf.reshape(
        tf.strided_slice(record_bytes,[label_bytes], [label_bytes + image_bytes]),
        [result.depth, result.height, result.width]
    )

    # convert from [depth, height, width] to [height, width, depth]
    result.uint8image = tf.transpose(depth_major, [1, 2, 0])


    return result 


def _generate_image_and_label_batch(image, label, min_queue_examples,
                                    batch_size, shuffle):
    """
    construct a queued batch of images and labels

    Args:
        image: 3-D Tensor of [height, width, 3] of type.float32.
        label: 1-D Tensor of type.int32
        min_queue_examples: int32, minimum number of samples to retain
        in the queue that provides of batches of examples.
        batch_size: Number of images per batch.
        shuffle: boolean indicating whether to use a shuffling queue.
    Returns:
        images: Images. 4D tensor of [batch_size, height, width, 3] size.
        labels: Labels. 1D tensor of [batch_size] size.
    """

    num_preprocess_threds = 16
    if shuffle:
        images, label_batch = tf.train.shuffle_batch(
            [image, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threds,
            capacity=min_queue_examples + 3*batch_size,
            min_after_dequeue=min_queue_examples
        )
    else:
        images, label_batch = tf.train.batch(
            [image, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size
        )

    # display the training images in the visualizer.
    tf.summary.image('images', images)

    return images, tf.reshape(label_batch, [batch_size])

def distorted_input(data_dir, batch_size):
    """
    construct distorted input for cifar training using the reader op. 

    Args:
        data_dir: Path to the CIFAR-10 data directory.
        batch_size: Number of images per batch.
    Returns:
        images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
        labels: Labels. 1D tensor of [batch_size] size.
    """
    filenames = [os.path.join(data_dir, 'data_batch_%d.bin' % i)
                for i in range(1, 6)]
    for f in filenames:
        if not tf.gfile.Exists(f):
            raise ValueError('Failed to find file: ', f)
    
    # create a queue that produces the filenames to read 
    filename_queue = tf.train.string_input_producer(filenames)

    with tf.name_scope("data_augmentation"):
        # read examples from files in the filename queue
        read_input = read_cifar10(filename_queue)
        reshape_image = tf.cast(read_input.uint8image, tf.float32)

        height = IMAGE_SIZE
        width = IMAGE_SIZE

        # image processing for training the network. Note the many random 
        # distortions applied to the image. 

        # randomly crop a [height, width] section of the image. 
        distorted_image = tf.random_crop(reshaped_image, [height, width, 3])

        # randomly flip the image horizontally. 
        distorted_image = tf.image.random_flip_left_right(distorted_image)

        # because these operations are not commutative, consider randomizing 
        # the order their operation. 

        # NOTE: since per_image_standardization zeros the mean and makes
        # the stddev unit, this likely has no effect see tensorflow#1458.

        distorted_image = tf.image.random_brightness(distorted_image,max_delta=63)
        distorted_image = tf.image.random_contrast(distorted_image, lower=0.2, upper=1.8)

        # subtract off the mean and divide by the variance of the pixels. 
        float_image = tf.image.per_image_standardization(distorted_image)

        # set the shapes of tensors 
        float_image.set_shape([height, width, 3])
        read_input.label.set_shape([1])

        # ensure that the random shuffling has good mixing properties. 
        min_fraction_of_examples_in_queue = 0.4 
        min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN*
                                min_fraction_of_examples_in_queue)
        min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
                             min_fraction_of_examples_in_queue)
        print ('Filling queue with %d CIFAR images before starting to train. '
            'This will take a few minutes.' % min_queue_examples)

    return _generate_image_and_label_batch(float_image, read_input.label,
                                min_queue_examples, batch_size,shuffle=True)


def inputs(eval_data, data_dir, batch_size):
    """
    construct input for cifar evaluation using the reader op

    Args:
        eval_data: bool, indicating if one should use the train or eval data set.
        data_dir: Path to the CIFAR-10 data directory.
        batch_size: Number of images per batch.
    Returns:
        images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
        labels: Labels. 1D tensor of [batch_size] size.

    """

    if not eval_data:
        filenames = [os.path.join(data_dir, 'data_batch_%d.bin' % i)
                 for i in range(1, 6)]
        num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
    else:
        filenames = [os.path.join(data_dir, 'test_batch.bin')]
        num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

    for f in filenames:
        if not tf.gfile.Exists(f):
            raise ValueError('Failed to find file: ' + f)

    with tf.name_scope("input"):
        filename_queue = tf.train.string_input_producer(filenames)

        # Read examples from files in the filename queue.
        read_input = read_cifar10(filename_queue)
        reshaped_image = tf.cast(read_input.uint8image, tf.float32)

        height = IMAGE_SIZE
        width = IMAGE_SIZE

        # Image processing for evaluation.
        # Crop the central [height, width] of the image.
        resized_image = tf.image.resize_image_with_crop_or_pad(reshaped_image,
                                        height, width)
        # Subtract off the mean and divide by the variance of the pixels.
        float_image = tf.image.per_image_standardization(resized_image)

        # Set the shapes of tensors.
        float_image.set_shape([height, width, 3])
        read_input.label.set_shape([1])

        # Ensure that the random shuffling has good mixing properties.
        min_fraction_of_examples_in_queue = 0.4
        min_queue_examples = int(num_examples_per_epoch *
                                min_fraction_of_examples_in_queue)

    # Generate a batch of images and labels by building up a queue of examples.
    return _generate_image_and_label_batch(float_image, read_input.label,
                                            min_queue_examples, batch_size,
                                            shuffle=False)
