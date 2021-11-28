import tensorflow as tf
import numpy as np

def discriminator(input_dim,filter,kernel,stride,useBatchNormalization,dropout):
    disc_input = tf.keras.layers.Input(input_dim,name='disc_input')
    x=disc_input
    for i in range(len(filter)):
        x = tf.keras.layers.Conv2D(filter[i],kernel[i],stride[i],name = "conv_block_"+str(i),padding='same')(x)

        if useBatchNormalization and i>0:
            x = tf.keras.layers.BatchNormalization()(x)

        x = tf.keras.layers.Activation('relu')(x)
        if dropout!=0:
            x =tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Flatten()(x)
    disc_output = tf.keras.layers.Dense(1,activation='sigmoid',name = 'disc_output')(x)
    
    return tf.keras.Model(disc_input,disc_output,name='discriminator')

def generator(z_dim,initial_size,filters,kernel_size,stride,useBatchNormalization,dropout,mom):
    gen_input = tf.keras.layers.Input((z_dim,),name='gen_input')
    x = gen_input
    x = tf.keras.layers.Dense(np.prod(initial_size))(x)
    if useBatchNormalization:
        x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Reshape(initial_size)(x)
    if dropout:
        x = tf.keras.layers.Dropout(dropout)(x)
    for i in range(len(filters)):
        x = tf.keras.layers.UpSampling2D()(x)
        x = tf.keras.layers.Conv2D(filters[i],kernel_size[i],stride[i],padding='same',name = "conv_block_"+str(i))(x)
        if i<len(filters)-1:
            if useBatchNormalization:
                x = tf.keras.layers.BatchNormalization(momentum=mom)(x)
            x= tf.keras.layers.Activation('relu')(x)
        else:
            x = tf.keras.layers.Activation('tanh')(x)
    generator_output = x
    return tf.keras.Model(gen_input,generator_output)

if __name__ == '__main__':
    disc = discriminator((28,28,3),[64,64,128,128],[5,5,5,5],[2,2,2,1],False,25)
    disc.summary()
    gen = generator(2,(7,7,64),[128,64, 64,1],[5,5,5,5],[1,1,2,2],False,0.5,mom=0.2)
    gen.summary()