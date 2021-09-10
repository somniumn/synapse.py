## load modules
import cv2 as cv
import matplotlib.pyplot as plt
from absl import app
import os, time, sys
import numpy as np
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential, load_model, Model
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import plot_model

sys.path.append("./data_mnist")
from data_mnist.mnist import load_mnist


# 압축해제된 데이터 경로를 찾아 복사해서 붙여넣어주세요
src = './data_3000/'
# img_size = 56  # 이미지 사이즈
img_size = 28  # 이미지 사이즈
channels = 1
noise_dim = 100


# 이미지 읽기
def img_read(src, file):
    img = cv.imread(src + file, cv.COLOR_BGR2GRAY)
    return img


def get_data():
    # src 경로에 있는 파일 명을 저장합니다.
    files = os.listdir(src)
    X = []

    # 경로와 파일명을 입력으로 넣어 확인하고
    # 데이터를 255로 나눠서 0~1사이로 정규화 하여 X 리스트에 넣습니다.

    for file in files:
        X.append((img_read(src, file) - 127.5) / 127.5)

        # Train set(80%), Test set(20%)으로 나누기
    X_train, X_test = train_test_split(X, test_size=0.2, random_state=1, shuffle=True)

    # (x, 56, 56, 1) 차원으로 맞춰줌
    X_train = np.expand_dims(X_train, axis=3)
    X_test = np.expand_dims(X_test, axis=3)

    return X_train, X_test


# ---------------------
#  Generator 모델 구성 (input : noise / output : image)
# ---------------------
def build_generator():
    model = Sequential()

    model.add(layers.Dense(256, use_bias=False, input_shape=(100,)))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.BatchNormalization(momentum=0.8))
    model.add(layers.Dense(512))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.BatchNormalization(momentum=0.8))
    model.add(layers.Dense(1024))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.BatchNormalization(momentum=0.8))
    model.add(layers.Dense(np.prod(img_size * img_size), activation='tanh'))
    model.add(layers.Reshape((img_size, img_size, 1)))

    # noise 텐서 생성, model 에 noise 넣으면 이미지 나옴
    noise = Input(shape=(100,))
    img = model(noise)
    model.summary()
    return Model(noise, img)


# ---------------------
#  Discriminator 모델 구성 (input : image / output : 판별값(0에서 1사이의 숫자))
# ---------------------
def build_discriminator():
    model = tf.keras.Sequential()
    img_shape = (img_size, img_size, channels)
    model.add(layers.Flatten(input_shape=img_shape))
    model.add(layers.Dense(512))
    model.add(layers.LeakyReLU(alpha=0.2))

    model.add(layers.Dense(256))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.summary()
    # 이미지 들어갈 텐서 생성, model 에 넣으면 판별값 나옴
    img = Input(shape=img_shape)
    validity = model(img)

    return Model(img, validity)


def main(argv):
    del argv  # Unused.

    # 데이터 셋 불러옴 (이미지만 필요해서 y 라벨 필요 없음)
    # X_train, X_test = get_data()
    (X_train, Y_train), (X_test, Y_test) = load_mnist(flatten=False, normalize=True)
    # X_train = np.expand_dims(X_train, axis=3)
    # X_test = np.expand_dims(X_test, axis=3)

    print("X_train.shape = {}".format(X_train.shape))
    print("X_test.shape = {}".format(X_test.shape))

    # images 확인용
    fig = plt.figure(figsize=(10, 10))
    nplot = 5
    for i in range(1, nplot):
        ax = fig.add_subplot(1, nplot, i)
        ax.imshow(X_train[i, :, :, 0], cmap=plt.cm.bone)
    plt.show()

    # Optimizer
    optimizer = Adam(0.0010, 0.5)

    # generator 모델 생성과 컴파일(loss 함수와 optimizer 설정)
    generator = build_generator()
    generator.compile(loss='binary_crossentropy', optimizer=optimizer)

    # 노이즈 만들어서 generator 에 넣은 후 나오는 이미지 출력 (확인용)
    noise = tf.random.normal([1, 100])
    generated_image = generator(noise, training=False)
    plt.imshow(generated_image[0, :, :, 0], cmap='gray')

    # plot_model(generator, show_shapes=True)

    # discriminator 모델 생성과 컴파일(loss 함수와 optimizer 설정, accuracy 측정)
    discriminator = build_discriminator()
    discriminator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    # image 를 discriminator 에 넣었을 때 판별값 나옴 (예시. 확인용)
    decision = discriminator(generated_image)
    print(decision)

    # Combined Model
    # 랜덤으로 만든 이미지로부터 학습해서 새로운 이미지를 만들어내는 generator 의 데이터를 discriminator 가 분류.

    z = layers.Input(shape=(100,), name="noise_input")
    img = generator(z)

    # 모델을 합쳐서 학습하기 때문에 발란스 때문에 discriminator 는 학습을 꺼둠. 우리는 generator 만 학습
    discriminator.trainable = False

    # discriminator 에 이미지를 입력으로 넣어서 진짜이미지인지 가짜이미지인지 판별
    valid = discriminator(img)

    # generator 와 discriminator 모델 합침. (노이즈가 인풋으로 들어가서 판별결과가 아웃풋으로 나오게)
    # discriminator 를 속이도록 generator 를 학습
    combined = Model(z, valid)
    combined.compile(loss='binary_crossentropy', optimizer=optimizer)
    combined.summary()

    def train(epochs, batch_size=128, sample_interval=50):

        # 정답으로 사용 할 매트릭스. valid는 1, fake는 0
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))
        history = []
        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # batch_size만큼 이미지와 라벨을 랜덤으로 뽑음
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]

            # Sample noise 생성(batch_size 만큼)
            noise = np.random.normal(0, 1, (batch_size, 100))

            # noise 를 generator 에 넣어서 fake image 이미지 생성
            gen_imgs = generator.predict(noise)

            # discriminator 를 학습함. 진짜 이미지는 1이 나오게, 가짜 이미지는 0이 나오게
            # discriminator 가 이미지를 판별한 값과 valid 와 fake 가
            # 각각 같이 들어가서 binary_crossentropy 으로 계산되어 업데이트함.
            d_loss_real = discriminator.train_on_batch(imgs, valid)
            d_loss_fake = discriminator.train_on_batch(gen_imgs, fake)

            # real 을 넣었을 때와 fake 를 넣었을 때의 discriminator 의 loss 값과 accuracy 값의 평균을 구함.
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            # noise 생성
            noise = np.random.normal(0, 1, (batch_size, noise_dim))

            # noise 가 들어가서 discriminator 가 real image 라고 판단하도록 generator 를 학습
            g_loss = combined.train_on_batch(noise, valid)

            history.append({"D": d_loss[0], "G": g_loss})

            # sample_interval(1000) epoch 마다 loss 와 accuracy 와 이미지 출력
            if epoch % sample_interval == 0:
                print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100 * d_loss[1], g_loss))
                sample_images(epoch)

        return history

    # 이미지 출력
    # generator 가 얼마나 학습이 잘 되었는지는 단지 loss 값만으로는 파악이 어려움 -> 직접 image 확인해봐야 함
    def sample_images(epoch):
        n = 10  # how many digits we will display
        noise = np.random.normal(0, 1, (n, noise_dim))
        gen_imgs = generator.predict(noise)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        plt.figure(figsize=(15, 4))

        for i in range(n):
            ax = plt.subplot(2, n, i + 1)
            plt.imshow(gen_imgs[i].reshape(img_size, img_size), vmin=0, vmax=1, cmap=plt.cm.bone)
            # plt.title("Gen"+str(i))
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
    #           fig.savefig("images/%d.png" % epoch)
    #        plt.close()

    # GAN 실행
    history = train(epochs=20001, batch_size=64, sample_interval=5000)

    # summarize history for loss
    import pandas as pd
    hist = pd.DataFrame(history)
    plt.figure(figsize=(10, 5))
    for colnm in hist.columns:
        plt.plot(hist[colnm], label=colnm)
    plt.legend()
    plt.ylabel("loss")
    plt.xlabel("epochs")

    generated_image = generator(tf.random.normal([1, noise_dim]))
    plt.imshow(tf.reshape(generated_image, shape=(img_size, img_size)), cmap=plt.cm.bone)

    plt.show()


if __name__ == '__main__':
    app.run(main)
