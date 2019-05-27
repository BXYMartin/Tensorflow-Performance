from tensorflow.contrib.slim.nets import alexnet
from tensorflow.contrib.slim.nets import inception
from tensorflow.contrib.slim.nets import overfeat
from tensorflow.contrib.slim.nets import resnet_v1
from tensorflow.contrib.slim.nets import resnet_v2
from tensorflow.contrib.slim.nets import vgg
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.ops import variables
from time import time
models = [
        "vgg.vgg_16",
        "vgg.vgg_19",
        "resnet_v1.resnet_v1_50",
        "resnet_v1.resnet_v1_101",
        "resnet_v1.resnet_v1_152",
        "resnet_v1.resnet_v1_200",
        "resnet_v2.resnet_v2_50",
        "resnet_v2.resnet_v2_101",
        "resnet_v2.resnet_v2_152",
        "resnet_v2.resnet_v2_200",
        "alexnet.alexnet_v2",
        "inception.inception_v1",
        "inception.inception_v2",
        "inception.inception_v3",

        ]

out = ""
ROUND = 3
session_config = tf.ConfigProto(
        )

for name in models:
    run_name = eval(name)
    #init = tf.global_variables_initializer()
    #init = variables.global_variables_initializer()
    with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
        for batch in [1, 2, 4, 8, 16]:
            with tf.Session(config=session_config) as sess:
                run_inputs = tf.zeros([batch, 224, 224, 3])
                logits, _ = run_name(run_inputs, is_training=False)
                sess.run(variables.global_variables_initializer())
                output = sess.run(logits)
                start = time()
                for i in range(ROUND):
                    output = sess.run(logits)
                end = (time() - start)/ROUND
                out = out + (name + "|" + str(batch) + "|" + str(end*1000)) + "\n";
                with open("result_1.txt", "a+") as res:
                    res.write((name + "|" + str(batch) + "|" + str(end*1000)) + "\n")

print(out)
