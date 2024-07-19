import tensorflow as tf
from models import build_generator, build_discriminator
from utils import load_data, generate_and_save_images, show_saved_images
import config

# 使用分布策略
strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

# 启用混合精度训练
from tensorflow.keras.mixed_precision import set_global_policy
policy = tf.keras.mixed_precision.Policy('mixed_float16')
set_global_policy(policy)

with strategy.scope():
    generator = build_generator()
    discriminator = build_discriminator()

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-4,
        decay_steps=10000,
        decay_rate=0.96,
        staircase=True)

    generator_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    def discriminator_loss(real_output, fake_output):
        real_loss = cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss

    def generator_loss(fake_output):
        return cross_entropy(tf.ones_like(fake_output), fake_output)

def preprocess_image(image):
    image = tf.cast(image, tf.float32)  # 将image转换为浮点数类型
    image = (image - 127.5) / 127.5
    return tf.expand_dims(image, axis=-1)

def load_and_preprocess_data(data_dir):
    images, _ = load_data(data_dir)
    images = tf.data.Dataset.from_tensor_slices(images)
    images = images.map(preprocess_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    images = images.cache().shuffle(buffer_size=10000).batch(config.BATCH_SIZE).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return images

train_dataset = load_and_preprocess_data(config.DATA_DIR)

seed = tf.random.normal([config.NUM_EXAMPLES_TO_GENERATE, config.NOISE_DIM])

@tf.function
def train_step(images):
    noise = tf.random.normal([config.BATCH_SIZE, config.NOISE_DIM])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

def train(dataset, epochs):
    for epoch in range(epochs):
        for image_batch in dataset:
            train_step(image_batch)

        if (epoch + 1) % config.SAVE_INTERVAL == 0:
            generate_and_save_images(generator, epoch + 1, seed)

        print(f'Epoch {epoch + 1} completed')

if __name__ == "__main__":
    train(train_dataset, config.EPOCHS)
    show_saved_images()
