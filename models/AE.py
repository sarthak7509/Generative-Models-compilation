import tensorflow as tf
import numpy as np
import json
import os
import pickle

class AUTOENCODER:
    def __init__(self,input_dim, enc_con_filter, enc_con_kernel_size, encoder_stride_size, dec_con_filter, dec_con_kernel_size, dec_stride, z_dim, useBatchNormalization=False, useDropout=False) :
        super().__init__()
        self.name = "AUTOENCODER"
        self.input_dim = input_dim
        self.enc_con_filter = enc_con_filter
        self.enc_con_kernel_size = enc_con_kernel_size
        self.encoder_stride_size = encoder_stride_size
        self.dec_con_filter = dec_con_filter
        self.dec_con_kernel_size = dec_con_kernel_size
        self.dec_stride = dec_stride
        self.z_dim = z_dim
        self.useBatchNormalization = useBatchNormalization
        self.useDropout = useDropout

        self.number_encoder_conv = len(self.enc_con_filter)
        self.number_decoder_conv = len(self.dec_con_filter)

        self._build()

    def _build(self):
        #Start building encoder input and connect to encoder model.
        encoder_input = tf.keras.layers.Input(self.input_dim, name="encoder_layer")
        x = encoder_input
        for i in range(self.number_encoder_conv):
            conv_layer = tf.keras.layers.Conv2D(
                self.enc_con_filter[i], self.encoder_stride_size[i], self.encoder_stride_size[i], padding='same', name='conv_layer_'+str(i))
            x = conv_layer(x)
            x = tf.keras.layers.LeakyReLU()(x)
            if self.useBatchNormalization:
                x = tf.keras.layers.BatchNormalization()(x)
            if self.useDropout:
                x = tf.keras.layers.Dropout(0.25)(x)
        shape_before_flattning = tf.keras.backend.int_shape(x)[1:]
        print(shape_before_flattning)

        x = tf.keras.layers.Flatten()(x)
        encoder_output = tf.keras.layers.Dense(self.z_dim,name='encoder_output')(x)

        self.encoder = tf.keras.Model(encoder_input,encoder_output)
        # self.encoder.summary()

        #decoder starts
        decoder_input = tf.keras.layers.Input(shape = (self.z_dim,),name= 'Decoder_input')
        x = tf.keras.layers.Dense(np.prod(shape_before_flattning))(decoder_input)
        x = tf.keras.layers.Reshape(shape_before_flattning)(x)

        for i in range(self.number_decoder_conv):
            conv_trans = tf.keras.layers.Conv2DTranspose(
                self.dec_con_filter[i],
                self.dec_con_kernel_size[i],
                self.dec_stride[i],
                padding='same',
                name= 'conv_transpose_layer_'+str(i)
            )
            x = conv_trans(x)

            if i < self.number_decoder_conv - 1:
                x = tf.keras.layers.LeakyReLU()(x)

                if self.useBatchNormalization:
                    x = tf.keras.layers.BatchNormalization()(x)

                if self.useDropout:
                    x = tf.keras.layers.Dropout(rate=0.25)(x)
            else:
                x = tf.keras.layers.Activation('sigmoid')(x)
        decoder_output = x

        self.decoder = tf.keras.Model(decoder_input,decoder_output)

        # final Auto encoder
        model_input = encoder_input
        model_output = self.decoder(encoder_output)
        self.model = tf.keras.Model(model_input,model_output)
        # self.model.summary()
    def compile(self,learning_rate):
        self.learning_rate = learning_rate
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        def r_loss(y_true,y_pred):
            return tf.reduce_mean(tf.square(y_true-y_pred),axis=[1,2,3])
        self.model.compile(optimizer=optimizer,loss=r_loss)
    def save(self,folder):
        if not os.path.exists(folder):
            os.makedirs(folder)
            os.makedirs(os.path.join(folder, 'viz'))
            os.makedirs(os.path.join(folder, 'weights'))
            os.makedirs(os.path.join(folder, 'images'))
        with open(os.path.join(folder, 'params.pkl'), 'wb') as f:
            pickle.dump([
                self.input_dim
                , self.enc_con_filter
                , self.enc_con_kernel_size
                , self.encoder_stride_size
                , self.dec_con_filter
                , self.dec_con_kernel_size
                , self.dec_stride
                , self.z_dim
                , self.useBatchNormalization
                , self.useDropout
            ], f)

        self.plot_model(folder)
    def load_weights(self,file_path):
        self.model.load_weights(file_path)
    def train(self,x_train, batch_size, epochs, run_folder, print_every_n_batches = 100, initial_epoch = 0, lr_decay = 1):
        checkpoint2 = tf.keras.callbacks.ModelCheckpoint(os.path.join(run_folder, 'weights/weights.h5'), save_weights_only=True, verbose=1)
        callbacks_list = [checkpoint2]
        self.model.fit(
            x_train
            , x_train
            , batch_size=batch_size
            , shuffle=True
            , epochs=epochs
            , initial_epoch=initial_epoch
            , callbacks=callbacks_list
        )

    def plot_model(self, run_folder):
        tf.keras.utils.plot_model(self.model, to_file=os.path.join(run_folder, 'viz/model.png'), show_shapes=True, show_layer_names=True)
        tf.keras.utils.plot_model(self.encoder, to_file=os.path.join(run_folder, 'viz/encoder.png'), show_shapes=True, show_layer_names=True)
        tf.keras.utils.plot_model(self.decoder, to_file=os.path.join(run_folder, 'viz/decoder.png'), show_shapes=True, show_layer_names=True)
if __name__ =='__main__':

    AE = AUTOENCODER(
        input_dim=(28,28,1),
        enc_con_filter=[32,64,64, 64],
        enc_con_kernel_size=[3,3,3,3],
        encoder_stride_size= [1,2,2,1],
        dec_con_filter= [64,64,32,1],
        dec_con_kernel_size= [3,3,3,3],
        dec_stride= [1,2,2,1],
        z_dim=2
    )