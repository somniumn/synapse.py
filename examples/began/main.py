#https://drive.google.com/uc?export=download&id=1URxZOJTO38qb3v1hOT3usuAE7nTGQDnL
import os
from absl import app
import numpy as np
from tensorflow.python import keras
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import losses
from keras.optimizers import Adam
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Conv2D, Flatten, Dense, UpSampling2D, Reshape, Lambda, Input, Dropout
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from keras.datasets import mnist

# from colab import drive


from tensorflow.python.keras.losses import mean_absolute_error

# drive.mount('/content/drive')


# 생성된 MNIST 이미지를 rowsxcols Grid로 보여주는 plot 함수 정의
def plot_imgs(path, imgs, rows, cols):
    fig = plt.figure(figsize=(rows, cols))
    fig.set_figheight(3)
    fig.set_figwidth(3)
    gs = gridspec.GridSpec(rows, cols)
    gs.update(wspace=0.05, hspace=0.05)

    for i, img in enumerate(imgs):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')

        plt.imshow(img.reshape((28, 28)), cmap='gray')

    return fig


def main(argv):
    del argv  # Unused.

    BATCH_SIZE = 64
    IMG_SHAPE = (28, 28, 1)

    # mnist 데이터 셋 불러옴
    (X_train, __), (__, __) = mnist.load_data()

    X_train = X_train.astype('float32')
    X_train = np.expand_dims(X_train, axis=3)

    data_gen = ImageDataGenerator(rescale=1/255.)  # 이미지 전처리(Rescale 0 to 1)
    train_data_generator = data_gen.flow(X_train, batch_size=BATCH_SIZE)

    plt.imshow(X_train[0,:,:,0], cmap='gray')

    def build_encoder(input_shape, z_size, n_filters, n_layers):
        """Encoder구축

        Arguments:
            input_shape (int): 이미지의 shape
            z_size (int): 특징 공간의 차원 수
            n_filters (int): 필터 수

        """
        model = Sequential()
        model.add(Conv2D(n_filters, 3, activation='elu', input_shape=input_shape, padding='same'))
        model.add(Conv2D(n_filters, 3, padding='same'))
        for i in range(2, n_layers + 1):
            model.add(Conv2D(i * n_filters, 3, activation='elu', padding='same'))
            model.add(Conv2D(i * n_filters, 3, activation='elu', padding='same'))

        model.add(Conv2D(n_layers * n_filters, 3, padding='same'))
        model.add(Flatten())
        model.add(Dense(z_size))
        model.summary()
        return model

    def build_decoder(output_shape, z_size, n_filters, n_layers):
        """Decoder 구축

        Arguments:
            output_shape (np.array): 이미지 shape
            z_size (int): 특징 공간의 차원 수
            n_filters (int): 필터 수
            n_layers (int): 레이어 수

        """
        # UpSampling2D로 몇 배로 확대할지 계산
        scale = 2 ** (n_layers - 1)
        # 합성곱층의 처음 입력 사이즈를 scale로부터 역산
        fc_shape = (output_shape[0] // scale, output_shape[1] // scale, n_filters)
        # 완전연결 계층에서 필요한 사이즈를 역산
        fc_size = fc_shape[0] * fc_shape[1] * fc_shape[2]

        model = Sequential()
        # 완전연결 계층
        model.add(Dense(fc_size, input_shape=(z_size,)))
        model.add(Reshape(fc_shape))

        # 합성곱층 반복
        for i in range(n_layers - 1):
            model.add(Conv2D(n_filters, 3, activation='elu', padding='same'))
            model.add(Conv2D(n_filters, 3, activation='elu', padding='same'))
            model.add(UpSampling2D())

        # 마지막 층은 UpSampling2D가 불필요
        model.add(Conv2D(n_filters, 3, activation='elu', padding='same'))
        model.add(Conv2D(n_filters, 3, activation='elu', padding='same'))
        # 출력층에서는 1채널로
        model.add(Conv2D(1, 3, padding='same'))

        return model

    def build_generator(img_shape, z_size, n_filters, n_layers):
        decoder = build_decoder(img_shape, z_size, n_filters, n_layers)
        return decoder

    def build_discriminator(img_shape, z_size, n_filters, n_layers):
        encoder = build_encoder(img_shape, z_size, n_filters, n_layers)
        decoder = build_decoder(img_shape, z_size, n_filters, n_layers)
        return keras.models.Sequential((encoder, decoder))

    def build_discriminator_trainer(discriminator):
        img_shape = discriminator.input_shape[1:]
        real_inputs = Input(img_shape)
        fake_inputs = Input(img_shape)
        real_outputs = discriminator(real_inputs)
        fake_outputs = discriminator(fake_inputs)

        return Model(
            inputs=[real_inputs, fake_inputs],
            outputs=[real_outputs, fake_outputs]
        )

    n_filters = 64 # 필터 수
    n_layers = 3 # 레이어 수
    z_size = 32  # 특징 공간의 차원

    generator = build_generator(IMG_SHAPE, z_size, n_filters, n_layers)
    discriminator = build_discriminator(IMG_SHAPE, z_size, n_filters, n_layers)
    discriminator_trainer = build_discriminator_trainer(discriminator)

    generator.summary()

    # discriminator.layers[1]은 디코더를 나타냄
    discriminator.layers[1].summary()

    def build_generator_loss(discriminator):
        # discriminator를 사용해서 손실 함수 정의
        def loss(y_true, y_pred):
            # y_true는 더미
            reconst = discriminator(y_pred)
            return mean_absolute_error(reconst, y_pred)
        return loss

    # 초기 학습률(Generator)
    g_lr = 0.0001
    generator_loss = build_generator_loss(discriminator)
    generator.compile(loss=generator_loss, optimizer=Adam(g_lr))

    # 초기 학습률(Discriminator)
    # k_var는 수치(일반 변수)
    k_var = 0.0
    # k : Keras(TensorFlow) Variable
    k = K.variable(k_var)

    d_lr = 0.0001

    discriminator_trainer.compile(loss=[ mean_absolute_error, mean_absolute_error],loss_weights=[1., -k], optimizer=Adam(d_lr))

    def measure(real_loss, fake_loss, gamma):
        return real_loss + np.abs(gamma*real_loss - fake_loss)

    # k의 갱신에 이용할 파라미터
    GAMMA = 1
    Lambda = 0.002

    # 반복 수. 100000～1000000 정도로 지정
    TOTAL_STEPS = 1000

    # 모델과 확인용 생성 이미지를 저장할 폴더
    IMG_SAVE_DIR = '/content/drive/My Drive/data/imgs'
    # 확인용으로 4x4 개의 이미지를 생성
    IMG_SAMPLE_SHAPE = (4, 4)
    N_IMG_SAMPLES = np.prod(IMG_SAMPLE_SHAPE)

    # 저장할 폴더가 없다면 생성
    os.makedirs(IMG_SAVE_DIR, exist_ok=True)

    # 샘플이미지용 랜덤 시드
    sample_seeds = np.random.uniform(-1, 1, (N_IMG_SAMPLES, z_size))

    history = []
    logs = []

    for epoch, batch in enumerate(train_data_generator):
        # 학습 종료
        if epoch > TOTAL_STEPS:
            break
        # 임의의 값(noise) 생성, 잠재변수의 input으로 사용할 noise를 균등분포(Uniform Distribution)에서 BATCH_SIZE만큼 샘플링
        z_g = np.random.uniform(-1, 1, (BATCH_SIZE, z_size))  # 균등 분포 -1과 1사이에 랜덤값 추출
        z_d = np.random.uniform(-1, 1, (BATCH_SIZE, z_size))
        # 생성 이미지(구분자의 학습에 이용), z_g 입력받아 가짜 이미지 생성
        g_pred = generator.predict(z_d)

        # 생성자를 1스텝 학습시킨다
        generator.train_on_batch(z_g, batch)
        # discriminator 1스텝 학습시킨다
        _, real_loss, fake_loss = discriminator_trainer.train_on_batch([batch, g_pred], [batch, g_pred])

        # k 를 갱신, generator & discriminator loss 균형맞춤. discriminator가 얼마나  fake images에 집중할 것인지 컨트롤. 매 batch마다 업데이트.
        k_var += Lambda * (GAMMA * real_loss - fake_loss)
        K.set_value(k, k_var)

        # g_measure 을 계산하기 위한 loss 저장
        history.append({'real_loss': real_loss, 'fake_loss': fake_loss})

        # 100번에 1번씩 로그 표시
        if epoch % 100 == 0:
            # 과거 100 번의 measure 의 평균
            measurement = np.mean([measure(loss['real_loss'], loss['fake_loss'], GAMMA) for loss in history[-100:]])
            logs.append({'Epoch:': epoch, 'k': K.get_value(k), 'measure': measurement, 'real_loss': real_loss,
                         'fake_loss': fake_loss})
            print(logs[-1])

            # 생성된 이미지 저장 및 보이기
            img_path = '{}/generated_{}.png'.format(IMG_SAVE_DIR, epoch)
            fig = plot_imgs(img_path, generator.predict(sample_seeds), rows=IMG_SAMPLE_SHAPE[0], cols=IMG_SAMPLE_SHAPE[1])
            plt.savefig(img_path, bbox_inches='tight')
            plt.axis('off')
            plt.show()


if __name__ == '__main__':
    app.run(main)
