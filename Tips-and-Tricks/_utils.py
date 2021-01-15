import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import librosa
import librosa.display
import numpy as np

# ! Audio functions


def summary_audio(x):
    if x.ndim == 1:
        SUM = ('\n{0:>10s}: {1:>15.4f}').format('min', np.amin(x))
        SUM += ('\n{0:>10s}: {1:>15.4f}').format('1st Quar',
                                                 np.percentile(x, 25))
        SUM += ('\n{0:>10s}: {1:>15.4f}').format('median', np.median(x))
        SUM += ('\n{0:>10s}: {1:>15.4f}').format('mean', np.mean(x))
        SUM += ('\n{0:>10s}: {1:>15.4f}').format('3rd Quar',
                                                 np.percentile(x, 75))
        SUM += ('\n{0:>10s}: {1:>15.4f}').format('max', np.amax(x))
        SUM += ('\n{0:>10s}: {1:>15.4f}').format('sigma', np.std(x))
    elif x.ndim == 2:
        E = x[:, 0]
        D = x[:, 1]
        SUM = ('{0:>%ss} {1:>%ss}' % (27, 15)).format('L', 'R')
        SUM += ('\n{0:>10s}: {1:>15.4f} {2:>15.4f}').format('min',
                                                            np.amin(E), np.amin(D))
        SUM += ('\n{0:>10s}: {1:>15.4f} {2:>15.4f}').format('1st Quar',
                                                            np.percentile(E, 25), np.percentile(D, 25))
        SUM += ('\n{0:>10s}: {1:>15.4f} {2:>15.4f}').format('median',
                                                            np.median(E), np.median(D))
        SUM += ('\n{0:>10s}: {1:>15.4f} {2:>15.4f}').format('mean',
                                                            np.mean(E), np.mean(D))
        SUM += ('\n{0:>10s}: {1:>15.4f} {2:>15.4f}').format('3rd Quar',
                                                            np.percentile(E, 75), np.percentile(D, 75))
        SUM += ('\n{0:>10s}: {1:>15.4f} {2:>15.4f}').format('max',
                                                            np.amax(E), np.amax(D))
        SUM += ('\n{0:>10s}: {1:>15.4f} {2:>15.4f}').format('sigma',
                                                            np.std(E), np.std(D))
    else:
        raise ValueError('Invalid argument! It is not an audio..')
    print(SUM)


def audiovis(x, fs=44100, **kwargs):
    dx, dy = (1536, 512) if 'dims' not in kwargs else kwargs['dims']
    dpi = 72 if 'dpi' not in kwargs else kwargs['dpi']
    text = '' if 'text' not in kwargs else kwargs['text']

    samples = x.shape[0]
    time = samples/fs
    if 'time' not in kwargs:
        to, ti = [0, time]
    else:
        to, ti = kwargs['time'][[0, -1]]
    tmin, tmax = (to, ti) if 'tlim' not in kwargs else kwargs['tlim']
    t = np.linspace(to, ti, samples)

    if x.ndim == 1:
        print('audio mono')
        fig = plt.figure(figsize=(dx/dpi, dy/dpi))
        plt.plot(t, x)
        plt.xlim([tmin, tmax])
    elif x.ndim == 2:
        print('audio stereo')
        fig = plt.figure(figsize=(dx/dpi, 2*dy/dpi))
        plt.subplot(2, 1, 1)
        plt.plot(t, x[:, 0], color='#00ff88')
        plt.xlim([tmin, tmax])
        plt.subplot(2, 1, 2)
        plt.plot(t, x[:, 1], color='#0088ff')
        plt.xlim([tmin, tmax])
    else:
        raise ValueError('Invalid argument! It is not an audio..')
    plt.title(text)
    plt.show()


def spectrogram(x, fs=44100, **kwargs):
    dx, dy = (1536, 512) if 'dims' not in kwargs else kwargs['dims']
    dpi = 72 if 'dpi' not in kwargs else kwargs['dpi']
    fmin, fmax = [0, 8000] if 'flim' not in kwargs else kwargs['flim']
    text = '' if 'text' not in kwargs else kwargs['text']

    x = x.astype(np.float32)

    if x.ndim == 1:
        print('audio mono')
        fig = plt.figure(figsize=(dx/dpi, dy/dpi))
        X = np.abs(librosa.stft(x))
        X = librosa.amplitude_to_db(X, ref=np.max)
        librosa.display.specshow(X, sr=fs, y_axis='linear')
        plt.ylim([fmin, fmax])
        plt.colorbar(format='%+2.0f dB')
    elif x.ndim == 2:
        print('audio stereo')
        fig = plt.figure(figsize=(dx/dpi, 2*dy/dpi))
        plt.subplot(2, 1, 1)
        X = np.abs(librosa.stft(x[:, 0]))
        X = librosa.amplitude_to_db(X, ref=np.max)
        librosa.display.specshow(X, sr=fs, y_axis='linear')
        plt.ylim([fmin, fmax])
        plt.colorbar(format='%+2.0f dB')
        plt.subplot(2, 1, 2)
        X = np.abs(librosa.stft(x[:, 1]))
        X = librosa.amplitude_to_db(X, ref=np.max)
        librosa.display.specshow(X, sr=fs, y_axis='linear')
        plt.ylim([fmin, fmax])
        plt.colorbar(format='%+2.0f dB')
    else:
        raise ValueError('Invalid argument! It is not an audio..')
    plt.title(text)
    plt.show()

# ! Image functions


def summary_image(image):
    if image.ndim == 2:
        SUM = ('\n{0:>10s}: {1:>15.4f}').format('min', np.amin(image))
        SUM += ('\n{0:>10s}: {1:>15.4f}').format('1st Quar',
                                                 np.percentile(image, 25))
        SUM += ('\n{0:>10s}: {1:>15.4f}').format('median', np.median(image))
        SUM += ('\n{0:>10s}: {1:>15.4f}').format('mean', np.mean(image))
        SUM += ('\n{0:>10s}: {1:>15.4f}').format('3rd Quar',
                                                 np.percentile(image, 75))
        SUM += ('\n{0:>10s}: {1:>15.4f}').format('max', np.amax(image))
        SUM += ('\n{0:>10s}: {1:>15.4f}').format('sigma', np.std(image))
    elif image.ndim == 3:
        R = image[:, :, 0]
        G = image[:, :, 1]
        B = image[:, :, 2]
        SUM = ('{0:>%ss} {1:>%ss} {2:>%ss}' %
               (27, 15, 15)).format('R', 'G', 'B')
        SUM += ('\n{0:>10s}: {1:>15.4f} {2:>15.4f} {3:>15.4f}').format('min',
                                                                       np.amin(R), np.amin(G), np.amin(B))
        SUM += ('\n{0:>10s}: {1:>15.4f} {2:>15.4f} {3:>15.4f}').format('1st Quar',
                                                                       np.percentile(R, 25), np.percentile(G, 25), np.percentile(B, 25))
        SUM += ('\n{0:>10s}: {1:>15.4f} {2:>15.4f} {3:>15.4f}').format('median',
                                                                       np.median(R), np.median(G), np.median(B))
        SUM += ('\n{0:>10s}: {1:>15.4f} {2:>15.4f} {3:>15.4f}').format('mean',
                                                                       np.mean(R), np.mean(G), np.mean(B))
        SUM += ('\n{0:>10s}: {1:>15.4f} {2:>15.4f} {3:>15.4f}').format('3rd Quar',
                                                                       np.percentile(R, 75), np.percentile(G, 75), np.percentile(B, 75))
        SUM += ('\n{0:>10s}: {1:>15.4f} {2:>15.4f} {3:>15.4f}').format('max',
                                                                       np.amax(R), np.amax(G), np.amax(B))
        SUM += ('\n{0:>10s}: {1:>15.4f} {2:>15.4f} {3:>15.4f}').format('sigma',
                                                                       np.std(R), np.std(G), np.std(B))
    else:
        raise ValueError('Invalid argument! It is not an image..')
    print(SUM)


def histogram(image, **kwargs):
    x, y = (3*(256 + 100), 256) if 'dims' not in kwargs else kwargs['dims']
    dpi = 72 if 'dpi' not in kwargs else kwargs['dpi']
    bins = 256 if 'bins' not in kwargs else kwargs['bins']
    rw = 0.95 if 'rw' not in kwargs else kwargs['rw']
    interval = [0, 255] if 'interval' not in kwargs else kwargs['interval']

    fig = plt.figure(figsize=(x/dpi, y/dpi))

    gs1 = gridspec.GridSpec(1, 3)
    gs1.update(	left=0,
                right=1,
                bottom=0,
                top=1,
                wspace=0,
                hspace=0)
    ax1 = plt.subplot(gs1[:, 0])
    ax1.axis('off')
    ax2 = plt.subplot(gs1[:, 1:])

    img = image.copy()

    if img.ndim == 2:
        ax1.imshow(	img,
                    cmap='gray',
                    vmin=interval[0],
                    vmax=interval[1])
        ax2.hist(	img.ravel(),
                  bins=bins,
                  range=interval,
                  rwidth=rw,
                  color='k')
    elif img.ndim == 3:
        imgo = (img - interval[0])/(interval[1] - interval[0])
        imgo[imgo < 0] = 0
        imgo[imgo > 1] = 1

        R = img[:, :, 0]
        G = img[:, :, 1]
        B = img[:, :, 2]

        ax1.imshow(imgo)
        ax2.hist(	R.ravel(),
                  bins=bins,
                  range=interval,
                  rwidth=rw,
                  fc=(1, 0, 0.25, 0.5))
        ax2.hist(	G.ravel(),
                  bins=bins,
                  range=interval,
                  rwidth=rw,
                  fc=(0.25, 1, 0, 0.5))
        ax2.hist(	B.ravel(),
                  bins=bins,
                  range=interval,
                  rwidth=rw,
                  fc=(0, 0.25, 1, 0.5))
    else:
        raise ValueError('Invalid argument! It is not an image..')
    plt.show()


def panel(images, gspec, **kargs):
    M, N = gspec

    x, y = (1024, 512) if 'dims' not in kargs else kargs['dims']
    dpi = 72 if 'dpi' not in kargs else kargs['dpi']
    texts = [] if 'texts' not in kargs else kargs['texts']
    tx, ty = (10, 10) if 'text_pos' not in kargs else kargs['text_pos']
    tc = 'white' if 'text_color' not in kargs else kargs['text_color']
    ts = 12 if 'text_size' not in kargs else kargs['text_size']
    interval = [0, 255] if 'interval' not in kargs else kargs['interval']

    fig = plt.figure(figsize=(x/dpi, y/dpi))
    gs = gridspec.GridSpec(N, M)
    gs.update(	left=0, right=1,
               bottom=0, top=1,
               wspace=0, hspace=0)
    images = (images - interval[0])/(interval[1] - interval[0])
    images[images < 0] = 0
    images[images > 1] = 1
    for n in range(N):
        for m in range(M):
            ax = plt.subplot(gs[n, m])
            ax.axis('off')
            try:
                image = images[n*M + m]
            except:
                image = images[0]*0
            if image.ndim == 2:
                ax.imshow(image, cmap='gray')
            else:
                ax.imshow(image)
            try:
                text = texts[n*M + m]
            except:
                text = ''
            ax.text(	tx, ty, text,
                     color=tc, size=ts,
                     horizontalalignment='left',
                     verticalalignment='top')
    plt.show()
