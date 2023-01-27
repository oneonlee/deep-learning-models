from tensorflow.keras import datasets, layers, models, activations, losses, optimizers, metrics

def create_yolo():
    model = models.Sequential()
    
    # Block1
    model.add(layers.Convolution2D(64, (7, 7), strides=(2, 2), input_shape=(448, 448, 3), padding='same'))
    model.add(layers.LeakyReLU(alpha=0.1))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

    # Block2
    model.add(layers.Convolution2D(192, (3, 3), padding='same'))
    model.add(layers.LeakyReLU(alpha=0.1))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))    
    
    # Block3
    model.add(layers.Convolution2D(128, (1, 1), padding='same'))
    model.add(layers.LeakyReLU(alpha=0.1))
    model.add(layers.Convolution2D(256, (3, 3), padding='same'))
    model.add(layers.LeakyReLU(alpha=0.1))
    model.add(layers.Convolution2D(256, (1, 1), padding='same'))
    model.add(layers.LeakyReLU(alpha=0.1))
    model.add(layers.Convolution2D(512, (3, 3), padding='same'))
    model.add(layers.LeakyReLU(alpha=0.1))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

    # Block4
    model.add(layers.Convolution2D(256, (1, 1), padding='same'))
    model.add(layers.LeakyReLU(alpha=0.1))
    model.add(layers.Convolution2D(512, (3, 3), padding='same'))
    model.add(layers.LeakyReLU(alpha=0.1))
    model.add(layers.Convolution2D(256, (1, 1), padding='same'))
    model.add(layers.LeakyReLU(alpha=0.1))
    model.add(layers.Convolution2D(512, (3, 3), padding='same'))
    model.add(layers.LeakyReLU(alpha=0.1))
    model.add(layers.Convolution2D(256, (1, 1), padding='same'))
    model.add(layers.LeakyReLU(alpha=0.1))
    model.add(layers.Convolution2D(512, (3, 3), padding='same'))
    model.add(layers.LeakyReLU(alpha=0.1))
    model.add(layers.Convolution2D(256, (1, 1), padding='same'))
    model.add(layers.LeakyReLU(alpha=0.1))
    model.add(layers.Convolution2D(512, (3, 3), padding='same'))
    model.add(layers.LeakyReLU(alpha=0.1))
    model.add(layers.Convolution2D(512, (1, 1), padding='same'))
    model.add(layers.LeakyReLU(alpha=0.1))
    model.add(layers.Convolution2D(1024, (3, 3), padding='same'))
    model.add(layers.LeakyReLU(alpha=0.1))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
    
    # Block5
    model.add(layers.Convolution2D(512, (1, 1), padding='same'))
    model.add(layers.LeakyReLU(alpha=0.1))
    model.add(layers.Convolution2D(1024, (3, 3), padding='same'))
    model.add(layers.LeakyReLU(alpha=0.1))
    model.add(layers.Convolution2D(512, (1, 1), padding='same'))
    model.add(layers.LeakyReLU(alpha=0.1))
    model.add(layers.Convolution2D(1024, (3, 3), padding='same'))
    model.add(layers.LeakyReLU(alpha=0.1))
    model.add(layers.Convolution2D(1024, (3, 3), padding='same'))
    model.add(layers.LeakyReLU(alpha=0.1))
    model.add(layers.Convolution2D(1024, (3, 3), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU(alpha=0.1))

    # Block6
    model.add(layers.Convolution2D(1024, (3, 3), padding='same'))
    model.add(layers.LeakyReLU(alpha=0.1))
    model.add(layers.Convolution2D(1024, (3, 3), padding='same'))
    model.add(layers.LeakyReLU(alpha=0.1))
    
    # Last Block
    model.add(layers.Flatten())
    model.add(layers.Dense(4096))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(7 * 7 * 30))
    model.add(layers.Reshape(target_shape=(7, 7, 30)))
    
    return model