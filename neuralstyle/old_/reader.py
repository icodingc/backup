from os import listdir
from os.path import isfile, join
import tensorflow as tf

mean_pixel = [123.68, 116.779, 103.939] # ImageNet average from VGG ..

def preprocess(image, size, max_length):
    shape = tf.shape(image)
    size_t = tf.constant(size, tf.float64)
    height = tf.cast(shape[0], tf.float64)
    width = tf.cast(shape[1], tf.float64)
    print width 
    cond_op = tf.less(width, height) if max_length else tf.less(height, width)

    new_height, new_width = tf.cond(
        cond_op,
        lambda: (size_t, (width * size_t) / height),
        lambda: ((height * size_t) / width, size_t))
    resized_image = tf.image.resize_images(image, tf.to_int32(new_height), tf.to_int32(new_width))

#    normalised_image = resized_image - mean_pixel
    normalised_image = resized_image
    return normalised_image

# max_length: Wether size dictates longest or shortest side. Default longest
def get_image(path, size, max_length=True):
    png = path.lower().endswith('png')
    img_bytes = tf.read_file(path)
    image = tf.image.decode_png(img_bytes, channels=3) if png else tf.image.decode_jpeg(img_bytes, channels=3)
    return preprocess(image, size, max_length)

def image(n, size, image_path, style_path, epochs=2, crop=True):
    #img , style | img,style |.....
    filenames=[]
    for f in listdir(image_path):
	    filenames.append(join(image_path,f))
	    filenames.append(join(style_path,f))

    png = filenames[0].lower().endswith('png') # If first file is a png, assume they all are

    filename_queue = tf.train.string_input_producer(filenames, num_epochs=epochs, shuffle=False)
    reader = tf.WholeFileReader()
    _, img_bytes = reader.read(filename_queue)
    image = tf.image.decode_png(img_bytes, channels=3) if png else tf.image.decode_jpeg(img_bytes, channels=3)
    processed_image = preprocess(image, size, False)
    if not crop:
        return tf.train.batch([processed_image], n, dynamic_pad=True)

    cropped_image = tf.slice(processed_image, [0,0,0], [size, size, 3])
    cropped_image.set_shape((size, size, 3))

    images = tf.train.batch([cropped_image], n)
    return images
if __name__ == "__main__":
        image_path = '/home/ldap/zhangxuesen/Data/coco'
        style_path = '/home/ldap/zhangxuesen/Data/try_coco'
        images = image(1,256,image_path,style_path)
