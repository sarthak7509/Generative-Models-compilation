import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

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
class GAN(tf.keras.Model):
    def __init__(self, discriminator, generator, latent_dim):
        super(GAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim

    def compile(self, d_optimizer, g_optimizer, loss_fn):
        super(GAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn
        self.d_loss_metric = tf.keras.metrics.Mean(name="d_loss")
        self.g_loss_metric = tf.keras.metrics.Mean(name="g_loss")

    @property
    def metrics(self):
        return [self.d_loss_metric, self.g_loss_metric]

    def train_step(self, real_images):
        # Sample random points in the latent space
        batch_size = tf.shape(real_images)[0]
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))

        # Decode them to fake images
        generated_images = self.generator(random_latent_vectors)

        # Combine them with real images
        combined_images = tf.concat([generated_images, real_images], axis=0)

        # Assemble labels discriminating real from fake images
        labels = tf.concat(
            [tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))], axis=0
        )
        # Add random noise to the labels - important trick!
        labels += 0.05 * tf.random.uniform(tf.shape(labels))

        # Train the discriminator
        with tf.GradientTape() as tape:
            predictions = self.discriminator(combined_images)
            d_loss = self.loss_fn(labels, predictions)
        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(
            zip(grads, self.discriminator.trainable_weights)
        )

        # Sample random points in the latent space
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))

        # Assemble labels that say "all real images"
        misleading_labels = tf.zeros((batch_size, 1))

        # Train the generator (note that we should *not* update the weights
        # of the discriminator)!
        with tf.GradientTape() as tape:
            predictions = self.discriminator(self.generator(random_latent_vectors))
            g_loss = self.loss_fn(misleading_labels, predictions)
        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))

        # Update metrics
        self.d_loss_metric.update_state(d_loss)
        self.g_loss_metric.update_state(g_loss)
        return {
            "d_loss": self.d_loss_metric.result(),
            "g_loss": self.g_loss_metric.result(),
        }
class Ganmonitor(tf.keras.callbacks.Callback):
  def __init__(self,gen,num_image=3,latent_dim=100):
    self.gen=gen
    self.num_image = num_image
    self.latent_dim = latent_dim
  def on_epoch_end(self, epoch, logs=None):
    seed = tf.random.normal([self.num_image,self.latent_dim])
    predictions = self.gen(seed,training=False)
    fig = plt.figure(figsize=(4, 4))
    for i in range(predictions.shape[0]):
      plt.subplot(4, 4, i+1)
      plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
      plt.axis('off')
    plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
    plt.show()
if __name__ == '__main__':
    disc = discriminator((28,28,3),[64,64,128,128],[5,5,5,5],[2,2,2,1],False,25)
    disc.summary()
    gen = generator(2,(7,7,64),[128,64, 64,1],[5,5,5,5],[1,1,2,2],False,0.5,mom=0.2)
    gen.summary()