import tensorflow as tf
import tensorflow_addons as tfa

def Unetgenerator(input_dim,gen_filter,stride,num_filter=4,channels=3):
    def downsample(layer_input,filters,f_size=4):
        d = tf.keras.layers.Conv2D(filters,kernel_size=f_size, padding='same',strides=2)(layer_input)
        d = tfa.layers.InstanceNormalization(axis=-1,center=False,scale=False)(d)
        d = tf.keras.layers.Activation('relu')(d)
        return d
    def upsample(layer_input,skip_input,filters,f_size=4,dropout_rate=0):
        u = tf.keras.layers.UpSampling2D(size=2)(layer_input)
        u = tf.keras.layers.Conv2D(filters,kernel_size=f_size,padding='same',strides=1)(u)
        u = tfa.layers.InstanceNormalization(axis=-1,center=False,scale=False)(u)
        u = tf.keras.layers.Activation('relu')(u)
        if dropout_rate:
            u = tf.keras.layers.Dropout()(u)
        u = tf.keras.layers.Concatenate()([u,skip_input])
        return u
    gen_input = tf.keras.layers.Input(input_dim,name='gen_input')
    d1 = downsample(gen_input,gen_filter)
    d2 = downsample(d1, gen_filter*2)
    d3 = downsample(d2, gen_filter*4)
    d4 = downsample(d3, gen_filter*8)

    u1 = upsample(d4,d3,gen_filter*4)
    u2 = upsample(u1,d2,gen_filter*2)
    u3 = upsample(u2,d1,gen_filter)

    u4 = tf.keras.layers.UpSampling2D(size=2)(u3)
    output = tf.keras.layers.Conv2D(channels,kernel_size=4,strides=1,padding='same',activation='tanh',)(u4)
    return tf.keras.Model(gen_input,output)

def discriminator(input_dim,disc_filter,stride=2):
    def convblock(layer_input,filter,stride,norm=True):
        y = tf.keras.layers.Conv2D(filter,kernel_size=4,strides=stride,padding='same')(layer_input)
        if norm:
            y = tfa.layers.InstanceNormalization(axis=-1,center=False,scale=False)(y)
        y = tf.keras.layers.LeakyReLU(0.2)(y)
        return y
    img = tf.keras.layers.Input(input_dim)
    y = convblock(img,disc_filter,stride,norm=False)
    y = convblock(y,disc_filter*2,stride)
    y = convblock(y,disc_filter*4,stride)
    y = convblock(y,disc_filter*8,stride)

    output = tf.keras.layers.Conv2D(1,kernel_size=4,strides=1,padding='same')(y)

    return tf.keras.Model(img,output)


class CycleGan(tf.keras.Model):
    def __init__(self, generator_AB, generator_BA, discriminator_A, discriminator_B, lambda_cycle=10.0,
                 lambda_identity=0.5):
        super(CycleGan, self).__init__()
        self.G_AB = generator_AB
        self.G_BA = generator_BA
        self.discriminator_A = discriminator_A
        self.discriminator_B = discriminator_B
        self.lambda_cycle = lambda_cycle
        self.lambda_identity = lambda_identity

    def compile(self, gen_A_optimizer, gen_B_optimizer, disc_A_optimizer, disc_B_optimizer, gen_loss_fn, dis_loss_fn):
        super(CycleGan, self).compile()
        self.gen_A_optimizer = gen_A_optimizer
        self.gen_B_optimizer = gen_B_optimizer
        self.disc_A_optimizer = disc_A_optimizer
        self.disc_B_optimizer = disc_B_optimizer
        self.gen_loss_fn = gen_loss_fn
        self.dis_loss_fn = dis_loss_fn
        self.cycle_loss_fn = tf.keras.losses.MeanAbsoluteError()
        self.identity_loss_fn = tf.keras.losses.MeanAbsoluteError()

    def train_step(self, batch_data):
        real_A, real_B = batch_data
        with tf.GradientTape() as gen_A_tape, tf.GradientTape() as gen_B_tape, tf.GradientTape() as dis_A_tape, tf.GradientTape() as dis_B_tape:
            # passing sample A from f(A->B) to map to B
            fake_b = self.G_AB(real_A, training=True)
            # passing sample B from f(B->A) to map to A
            fake_a = self.G_BA(real_B, training=True)

            # cycle back from fake b to sample A
            cycle_a = self.G_BA(fake_b, training=True)
            # cycle back from fake a to sample B
            cycle_b = self.G_AB(fake_a, training=True)

            # identity mapping
            same_a = self.G_BA(real_A, training=True)
            same_b = self.G_AB(real_B, training=True)

            # discriminator output
            disc_real_a = self.discriminator_A(real_A, training=True)
            disc_real_b = self.discriminator_B(real_B, training=True)

            fake_a = self.discriminator_A(fake_a, training=True)
            fake_b = self.discriminator_B(fake_b, training=True)

            gen_ab_loss = self.gen_loss_fn(fake_b)
            gen_ba_loss = self.gen_loss_fn(fake_a)

            cycle_loss_ab = self.cycle_loss_fn(real_B, cycle_b) * self.lambda_cycle
            cycle_loss_ba = self.cycle_loss_fn(real_A, cycle_a) * self.lambda_cycle

            id_loss_AB = (
                    self.identity_loss_fn(real_B, same_b) * self.lambda_cycle * self.lambda_identity
            )

            id_loss_BA = (
                    self.identity_loss_fn(real_A, same_a) * self.lambda_cycle * self.lambda_identity
            )

            # Total Generator Loss
            total_loss_ab = gen_ab_loss + cycle_loss_ab + id_loss_AB
            total_loss_ba = gen_ba_loss + cycle_loss_ba + id_loss_BA

            # Discriminator Loss
            disc_A_loss = self.dis_loss_fn(disc_real_a, fake_a)
            disc_B_loss = self.dis_loss_fn(disc_real_b, fake_b)

        # Computing gradient for each generator
        grad_AB = gen_A_tape.gradient(total_loss_ab, self.G_AB.trainable_weights)
        grad_BA = gen_B_tape.gradient(total_loss_ba, self.G_BA.trainable_weights)

        # Computing gradient of discriminator
        disc_grad_A = dis_A_tape.gradient(disc_A_loss, self.discriminator_A.trainable_weights)
        disc_grad_B = dis_B_tape.gradient(disc_B_loss, self.discriminator_B.trainable_weights)
        # apply grads to gen
        self.gen_A_optimizer.apply_gradients(zip(grad_AB, self.G_AB.trainable_weights))
        self.gen_B_optimizer.apply_gradients(zip(grad_BA, self.G_BA.trainable_weights))

        # Apply grads to disc
        self.disc_A_optimizer.apply_gradients(zip(disc_grad_A, self.discriminator_A.trainable_weights))
        self.disc_B_optimizer.apply_gradients(zip(disc_grad_B, self.discriminator_B.trainable_weights))

        return {
            "G_loss": total_loss_ab,
            "F_loss": total_loss_ba,
            "D_X_loss": disc_A_loss,
            "D_Y_loss": disc_B_loss,
        }


if __name__ == '__main__':
    model = Unetgenerator((28,28,3),32,channels=3)
    model.summary()