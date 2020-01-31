mnist(keras).py

   $ python mnist(keras).py -e 20 -b 256 -l 0.005

       default  epochs         10
                batch_size     128
                learning_rate  0.001

mnist(keras_cnn).py

   $ python mnist(keras_cnn).py -e 40 -b 128 -l 0.005

       default  epochs         50
                batch_size     128
                learning_rate  0.01

mnist(no_keras_no_tensorflow).py

   $ python mnist(no_keras_no_tensorflow).py -e 5 -l 0.01 -i 5

       default  epochs         3
                learning_rate  0.1
                input_volume   3   (max=<600)
