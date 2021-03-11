import tensorflow as tf
import numpy as np
import scipy.io as scio
import os
import pandas as pd

data01 = scio.loadmat('/userhome/MAD/t_gb01.mat')
gb01 = data01['t_gb01']
data02 = scio.loadmat('/userhome/MAD/t_gb02.mat')
gb02 = data02['t_gb02']
data03 = scio.loadmat('/userhome/MAD/t_gb03.mat')
gb03 = data03['t_gb03']
data04 = scio.loadmat('/userhome/MAD/t_gb04.mat')
gb04 = data04['t_gb04']
data05 = scio.loadmat('/userhome/MAD/t_gb05.mat')
gb05 = data05['t_gb05']

data01 = scio.loadmat('/userhome/MAD/t_sp01.mat')
sp01 = data01['t_sp01']
data02 = scio.loadmat('/userhome/MAD/t_sp02.mat')
sp02 = data02['t_sp02']
data03 = scio.loadmat('/userhome/MAD/t_sp03.mat')
sp03 = data03['t_sp03']
data04 = scio.loadmat('/userhome/MAD/t_sp04.mat')
sp04 = data04['t_sp04']

data = scio.loadmat('/userhome/MAD/t_csf.mat')
csf = tf.cast(data['t_csf'], dtype=tf.complex64)

Gabors = tf.cast(np.stack((gb01, gb02, gb03, gb04, gb05), axis=0), dtype=tf.complex64)
Spreads = tf.cast(np.stack((sp01, sp02, sp03, sp04), axis=0), dtype=tf.complex64)

M = 384
N = 512

BSIZE = 16
G = 0.5
kernel_16 = np.ones((16, 16), dtype=np.float32) / (16 * 16)
kernel_16 = kernel_16[..., np.newaxis, np.newaxis]

kernel_8 = np.ones((8, 8), dtype=np.float32) / (8 * 8)
kernel_8 = kernel_8[..., np.newaxis, np.newaxis]


@tf.function
def gaborconvolve(im1=0, Gabors=0, Spreads=0):
    nscale = 5
    norient = 4

    imagefft1 = tf.signal.fft2d(im1)

    EO = []
    # The main loop...
    for o in range(norient):
        for s in range(nscale):
            filter = tf.signal.fftshift(tf.math.multiply(tf.squeeze(Gabors[s, :, :]), tf.squeeze(Spreads[o, :, :])))
            EO.append(tf.signal.ifft2d(tf.multiply(imagefft1, filter)))
            # data=tf.squeeze(tf.signal.ifft2d(tf.multiply(imagefft1, filter))).numpy()
            # EO2.append(tf.signal.ifft2d(tf.multiply(imagefft2, filter)))

    return EO


@tf.function
def madsc(refc, dstc):
    ref_hi = tf.math.real(
        tf.signal.ifft2d(tf.signal.ifftshift(tf.math.multiply(tf.signal.fftshift(tf.signal.fft2d(refc)), csf))))
    dst_hi = tf.math.real(
        tf.signal.ifft2d(tf.signal.ifftshift(tf.math.multiply(tf.signal.fftshift(tf.signal.fft2d(dstc)), csf))))

    in011 = tf.expand_dims(dst_hi - ref_hi, axis=3)
    in021 = tf.expand_dims(ref_hi, axis=3)

    ######STD of Dst-Ref##########################
    in011_mean = tf.nn.conv2d(in011, kernel_16, strides=[1, 4, 4, 1], padding='SAME')
    in011_mean_r = tf.image.resize(in011_mean, [M, N], method='nearest')
    in01_std = tf.math.sqrt(
        tf.nn.conv2d(tf.math.multiply(in011 - in011_mean_r, in011 - in011_mean_r), kernel_16 * 256.0 / 255.0,
                     strides=[1, 4, 4, 1], padding='SAME'))

    ######Mean of Ref#############################
    in021_mean = tf.nn.conv2d(in021, kernel_16, strides=[1, 4, 4, 1], padding='SAME')

    ######Modified STD of Ref#############################
    in021_mean_8 = tf.nn.conv2d(in021, kernel_8, strides=[1, 4, 4, 1], padding='SAME')
    in021_mean_r = tf.image.resize(in021_mean_8, [M, N], method='nearest')

    in02_std = tf.math.sqrt(
        tf.nn.conv2d(tf.math.multiply(in021 - in021_mean_r, in021 - in021_mean_r), kernel_8 * 64.0 / 63.0,
                     strides=[1, 4, 4, 1], padding='SAME'))
    in02_std_mod = tf.image.resize(-1 * tf.nn.max_pool2d(-1 * in02_std, ksize=2, strides=2, padding='SAME'),
                                   [int(M / 4), int(N / 4)], method='nearest')

    Ci_ref = tf.math.log(tf.math.divide(in02_std_mod, in021_mean))
    Ci_dst = tf.math.log(tf.math.divide(in01_std, in021_mean))
    Ci_dst = tf.where(in021_mean < G, -100.0, Ci_dst)
    idx01_cond1 = (Ci_ref > -5)
    idx01_cond2 = (Ci_dst > Ci_ref)
    idx01 = tf.logical_and(idx01_cond1, idx01_cond2)

    idx02_cond1 = (Ci_ref <= -5)
    idx02_cond2 = (Ci_dst > -5)
    idx02 = tf.logical_and(idx02_cond1, idx02_cond2)

    msk = tf.zeros_like(Ci_ref)
    msk = tf.where(idx01, Ci_dst - Ci_ref, msk)
    msk = tf.where(idx02, Ci_dst + 5, msk)
    msk_r = tf.image.resize(msk, [M, N], method='nearest')
    #######Remove IF-ELSEs################################
    # Ci_dst_m2 = tf.image.resize(Ci_dst, [M, N], method='nearest')

    lmse = tf.nn.conv2d(tf.multiply(tf.cast(tf.expand_dims(refc - dstc, axis=3), tf.float32),
                                    tf.cast(tf.expand_dims(refc - dstc, axis=3), tf.float32)), kernel_16,
                        strides=[1, 1, 1, 1],
                        padding='SAME')
    mp = tf.math.multiply(msk_r, lmse)
    mpr = mp[:, 16:-17, 16:-17, :]
    Sc01 = tf.squeeze(tf.math.sqrt(tf.math.reduce_mean(tf.math.multiply(mpr, mpr), axis=[1, 2])) * 200)

    gbRef = tf.math.abs(gaborconvolve(refc, Gabors, Spreads))
    gbDst = tf.math.abs(gaborconvolve(dstc, Gabors, Spreads))
    I = 0
    # print('++++++')
    ww = np.array([0.5, 0.75, 1, 5, 6, 0.5, 0.75, 1, 5, 6, 0.5, 0.75, 1, 5, 6, 0.5, 0.75, 1, 5, 6])
    ww = 4.0 * (ww / np.sum(ww))
    mp = tf.zeros((1, int(M / 4), int(N / 4), 1), dtype=tf.float32)
    for i in range(20):
        ref = tf.expand_dims(gbRef[i], 3)
        mean_ref = tf.nn.conv2d(ref, kernel_16, strides=[1, 4, 4, 1], padding='SAME')
        mean_ref_r = tf.image.resize(mean_ref, [M, N], method='nearest')
        ref_diff_n2 = tf.nn.conv2d(tf.math.multiply(ref - mean_ref_r, ref - mean_ref_r), kernel_16 * 256.0,
                                   strides=[1, 4, 4, 1], padding='SAME')
        ref_diff_n3 = tf.nn.conv2d(tf.math.pow(ref - mean_ref_r, 3), kernel_16 * 256.0, strides=[1, 4, 4, 1],
                                   padding='SAME')
        ref_diff_n4 = tf.nn.conv2d(tf.math.pow(ref - mean_ref_r, 4), kernel_16 * 256.0, strides=[1, 4, 4, 1],
                                   padding='SAME')
        ref_std = tf.math.sqrt(ref_diff_n2 / 256.0)
        ref_std_ret = tf.math.sqrt(ref_diff_n2 / 255.0)

        ref_skw = tf.math.divide(ref_diff_n3 / 256.0, tf.math.pow(ref_std, 3))
        ref_krt = tf.math.divide(ref_diff_n4 / 256.0, tf.math.pow(ref_std, 4))
        ref_skw = tf.where(ref_std == 0, 0.0, ref_skw)
        ref_krt = tf.where(ref_std == 0, 0.0, ref_krt)

        dst = tf.expand_dims(gbDst[i], 3)
        mean_dst = tf.nn.conv2d(dst, kernel_16, strides=[1, 4, 4, 1], padding='SAME')
        mean_dst_r = tf.image.resize(mean_dst, [M, N], method='nearest')
        dst_diff_n2 = tf.nn.conv2d(tf.math.multiply(dst - mean_dst_r, dst - mean_dst_r), kernel_16 * 256.0,
                                   strides=[1, 4, 4, 1], padding='SAME')
        dst_diff_n3 = tf.nn.conv2d(tf.math.pow(dst - mean_dst_r, 3), kernel_16 * 256.0, strides=[1, 4, 4, 1],
                                   padding='SAME')
        dst_diff_n4 = tf.nn.conv2d(tf.math.pow(dst - mean_dst_r, 4), kernel_16 * 256.0, strides=[1, 4, 4, 1],
                                   padding='SAME')
        dst_std = tf.math.sqrt(dst_diff_n2 / 256.0)
        dst_std_ret = tf.math.sqrt(dst_diff_n2 / 255.0)

        dst_skw = tf.math.divide(dst_diff_n3 / 256.0, tf.math.pow(dst_std, 3))
        dst_krt = tf.math.divide(dst_diff_n4 / 256.0, tf.math.pow(dst_std, 4))
        dst_skw = tf.where(dst_std == 0, 0.0, dst_skw)
        dst_krt = tf.where(dst_std == 0, 0.0, dst_krt)

        mp += ww[i] * (tf.math.abs(ref_std - dst_std) + 2 * tf.math.abs(ref_skw - dst_skw) + tf.math.abs(
            ref_krt - dst_krt))

    # I += tf.math.reduce_mean(tf.math.pow(mp[:, 16:-17, 16:-17, :], 2))
    th2 = 3.35
    th1 = 2.55
    b1 = np.exp(-1.0 * th1 / th2)
    b2 = 1.0 / (np.log(10) * th2)
    mp = tf.squeeze(mp[:, 4:-5, 4:-5, :])
    I = tf.math.sqrt(tf.reduce_mean(mp, axis=[1, 2]))
    sig = 1.0 / (1.0 + b1 * tf.math.pow(Sc01, b2))
    madscores = tf.math.pow(I, 1 - sig) * tf.math.pow(Sc01, sig)

    return madscores


def _parse_function_A(filename1, filename2):
    image1 = tf.io.read_file(filename1)
    image_rgb1 = tf.image.decode_png(image1, channels=3)
    image11 = tf.cast(tf.image.rgb_to_grayscale(image_rgb1), dtype=tf.complex64)
    image2 = tf.io.read_file(filename2)
    image_rgb2 = tf.image.decode_bmp(image2, channels=3)
    image22 = tf.cast(tf.image.rgb_to_grayscale(image_rgb2), dtype=tf.complex64)
    return image11, image22


def madsccal(file_path, ref_dir, dst_dir):
    print("===SPATIAL SMAD Calculating===")
    data01 = pd.read_csv(file_path)
    ref_file_list = data01['ref_im'].tolist()
    dst_file_list = data01['dist_im'].tolist()

    # ref_file_list = os.listdir(ref_dir)
    # dst_file_list = os.listdir(dst_dir)

    ref_files = [os.path.join(ref_dir, f) for f in ref_file_list]
    dst_files = [os.path.join(dst_dir, f) for f in dst_file_list]
    TOTAL_NUM = len(dst_files)
    #TOTAL_NUM = 100
    BATCH_NUM = 48
    EPOCH_NUM = int(TOTAL_NUM / BATCH_NUM)
    dataset = tf.data.Dataset.from_tensor_slices((ref_files, dst_files))
    dataset = dataset.map(_parse_function_A)
    dataset = dataset.batch(BATCH_NUM)
    # iterator = dataset.make_one_shot_iterator()
    train_iter = iter(dataset)
    madval = np.zeros((TOTAL_NUM, 1), dtype=np.float32)
    for i in range(EPOCH_NUM):
        refc, dstc = (train_iter.get_next())
        refc = tf.squeeze(refc)
        dstc = tf.squeeze(dstc)
        madvalbat = madsc(refc, dstc)
        madval[i * BATCH_NUM:(i + 1) * BATCH_NUM, 0] = madvalbat.numpy()
        print(i)

    return madval


if __name__ == '__main__':
    x = madsccal('/userhome/MAD/kadis700k_names.csv', '/userhome/kadia/kadis700k/ref_imgs',
                 '/userhome/kadia/kadis700k/dist_imgs/')
    scio.savemat('madsc.mat',{'madsc':x})
