import tensorflow as tf
import numpy as np
import json
import os
import pickle

class VARIATIONALAE:
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
        # Start building encoder input and connect to encoder model.
        encoder_input = tf.keras.layers.Input(self.input_dim, name="encoder_layer")
        x = encoder_input
        for i in range(self.number_encoder_conv):
            conv_layer = tf.keras.layers.Conv2D(
                self.enc_con_filter[i], self.encoder_stride_size[i], self.encoder_stride_size[i], padding='same',
                name='conv_layer_' + str(i))
            x = conv_layer(x)
            x = tf.keras.layers.LeakyReLU()(x)
            if self.useBatchNormalization:
                x = tf.keras.layers.BatchNormalization()(x)
            if self.useDropout:
                x = tf.keras.layers.Dropout(0.25)(x)
        shape_before_flattning = tf.keras.backend.int_shape(x)[1:]
        print(shape_before_flattning)

        x = tf.keras.layers.Flatten()(x)
        self.mu = tf.keras.layers.Dense(self.z_dim,name="mu")(x)
        self.log_var = tf.keras.layers.Dense(self.z_dim,name='log_var')(x)
        encoder_mu_log_var = tf.keras.Model(encoder_input,[self.mu,self.log_var])
        def sampling(args):
            mu,log_var = args
            epsilon = tf.random.normal(shape=(tf.shape(mu)),mean=0,stddev=1)
            sigma = tf.exp(log_var/2)
            return mu+sigma*epsilon
        encoder_output = tf.keras.layers.Lambda(sampling,name='encoder_output')([self.mu,self.log_var])
        self.encoder = tf.keras.Model(encoder_input,encoder_output)
        self.encoder.summary()

        # decoder starts
        decoder_input = tf.keras.layers.Input(shape=(self.z_dim,), name='Decoder_input')
        x = tf.keras.layers.Dense(np.prod(shape_before_flattning))(decoder_input)
        x = tf.keras.layers.Reshape(shape_before_flattning)(x)

        for i in range(self.number_decoder_conv):
            conv_trans = tf.keras.layers.Conv2DTranspose(
                self.dec_con_filter[i],
                self.dec_con_kernel_size[i],
                self.dec_stride[i],
                padding='same',
                name='conv_transpose_layer_' + str(i)
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

        self.decoder = tf.keras.Model(decoder_input, decoder_output)
        # final Auto encoder
        model_input = encoder_input
        model_output = self.decoder(encoder_output)
        self.model = tf.keras.Model(model_input, model_output)
        # self.model.summary()

    def compile(self,learning_rate,r_loss_factor):
        self.learning_rate = learning_rate
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        def vae_r_loss(y_true,y_pred):
            return tf.reduce_mean(tf.square(y_true-y_pred),axis=[1,2,3]) * r_loss_factor
        def vae_kl_loss(y_true,y_pred):
            kl_loss = -0.5 * tf.reduce_sum(1+self.log_var-tf.square(self.mu)-tf.exp(self.log_var),axis=1)
            return kl_loss
        def vae_loss(y_true,y_pred):
            kl_loss = vae_kl_loss(y_true,y_pred)
            r_loss=vae_r_loss(y_true,y_pred)
            return r_loss+kl_loss
        self.model.compile(optimizer=optimizer,loss=vae_loss,metrics = [vae_r_loss, vae_kl_loss])
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
        checkpoint_filepath = os.path.join(run_folder, "weights/weights-{epoch:03d}-{loss:.2f}.h5")
        checkpoint1 = tf.keras.callbacks.ModelCheckpoint(checkpoint_filepath, save_weights_only=True, verbose=1)
        checkpoint2 = tf.keras.callbacks.ModelCheckpoint(os.path.join(run_folder, 'weights/weights.h5'), save_weights_only=True, verbose=1)
        callbacks_list = [checkpoint1,checkpoint2]
        self.model.fit(
            x_train
            , x_train
            , batch_size=batch_size
            , shuffle=True
            , epochs=epochs
            , initial_epoch=initial_epoch
        )

    def plot_model(self, run_folder):
        tf.keras.utils.plot_model(self.model, to_file=os.path.join(run_folder, 'viz/model.png'), show_shapes=True, show_layer_names=True)
        tf.keras.utils.plot_model(self.encoder, to_file=os.path.join(run_folder, 'viz/encoder.png'), show_shapes=True, show_layer_names=True)
        tf.keras.utils.plot_model(self.decoder, to_file=os.path.join(run_folder, 'viz/decoder.png'), show_shapes=True, show_layer_names=True)
if __name__ =='__main__':

    AE = VARIATIONALAE(
        input_dim=(28,28,1),
        enc_con_filter=[32,64,64, 64],
        enc_con_kernel_size=[3,3,3,3],
        encoder_stride_size= [1,2,2,1],
        dec_con_filter= [64,64,32,1],
        dec_con_kernel_size= [3,3,3,3],
        dec_stride= [1,2,2,1],
        z_dim=2
    )