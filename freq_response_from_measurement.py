import sys
import json
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED
from pathlib import PurePath
import numpy as np
from scipy.io import wavfile
from matplotlib import pyplot as plt
from scipy.fftpack import fft, fftfreq
from scipy.signal import ellip, lfilter, freqz


def nlms_alg(x, y, N, signal):
    dx = np.zeros(shape=(1, N))
    hx = np.zeros(shape=(1, N))
    mux = 0.00002
    px = 0.001

    L = np.max(x.size)
    L1 = np.floor(0.8 * L)

    z = np.empty(shape=(1, N))
    for k in range(0, L):
        signal.progress.emit(int(k*100 / L))
        dx[0] = np.append(x[k], dx[0, 0:-1])
        es = np.matmul(dx[0], hx[0].T)
        b = y[k] - es
        z[0, k:k+1] = b
        if k < L1:
            px = px + (x[k] * x[k] - px) * 0.001
            ex = mux * b / (px + 1 / (10 ** 5))
            hx = hx + ex * dx

    return z, hx


def amplitude_calculator(data, boost, fs):
    data_abs = np.empty(shape=(100, int(fs / 2)))
    for k in range(0, 100):
        data_abs[k, :] = np.abs(fft(data[(fs * 5 + k * int(fs / 2)): (fs * 5 + (k + 1) * int(fs / 2))]))
    data_db = 20 * np.log10(np.sum(data_abs, 0) / 100) + boost
    return data_db


def read_measurement_json(arg):
    with open(PurePath(arg), 'r') as f:
        measurements = json.load(f)

    return measurements


def get_anc_normalization_curve(measurement, bit_depth, signal):
    signal.status.emit(f"Processing...")
    fs, passive = wavfile.read(measurement['base']['file'])
    passive = passive.astype(bit_depth)
    passive_norm = passive.T / np.max(passive)

    passive_db = amplitude_calculator(passive_norm[measurement['base']['ch'], 200001:2900000], 0, fs)

    lines_label = []
    anc_db = []
    for m in measurement['anc']:
        print(m)
        fs, anc = wavfile.read(m['file'])
        anc = anc.astype(bit_depth)
        anc_norm = anc.T / np.max(anc)
        anc_db.append({'fs': fs,
                       'data': (amplitude_calculator(anc_norm[m['ch'], 200001:2900000], 0, fs) - passive_db)})
        if m['boost'] == 0.0:
            lines_label.append(m['legend'])
        else:
            lines_label.append(f'{m["legend"]:s} {m["boost"]:.2f}dB')

    print(anc_db)

    return anc_db, lines_label


def get_frequency_response_curve(measurement, bit_depth, signal):
    recordings_db = []
    lines_label = []
    for m in measurement['recording']:
        print(m)
        signal.status.emit(f"Processing...")
        fs, recording = wavfile.read(PurePath(m['file']))
        recording = recording.astype(bit_depth)
        print('figure props: ' + bit_depth)
        recording_norm = recording.T / np.max(recording)
        ch = m['ch']
        recordings_db.append({'fs': fs,
                              'data': amplitude_calculator(recording_norm[ch, 200001:2900000], m['boost'], fs)})

        if m['boost'] == 0.0:
            lines_label.append(m['legend'])
        else:
            lines_label.append(f'{m["legend"]:s} {m["boost"]:.2f}dB')

    print(recordings_db)

    return recordings_db, lines_label


def nlms_alg_plot(measurement, signal):
    executor = ThreadPoolExecutor(max_workers=2)
    [bx, ax] = ellip(2, 0.0005, 40, 20 / 24000, 'high')

    impulse_resp = []
    recordings_db = []
    lines_label = []
    for m in measurement['recording']:
        print(f"Handle wav file: {m['file']:s}")
        signal.status.emit(f"{m['file']:s} is processing...")
        fs, wav = wavfile.read(PurePath(m['file']))
        wav = wav.astype('float32')
        wav = wav.T[:, 24000:]
        # x = numba_lfilter(bx, ax, wav[1, :])
        # y = numba_lfilter(bx, ax, wav[0, :])
        x = executor.submit(lfilter, bx, ax, wav[0, :])
        y = executor.submit(lfilter, bx, ax, wav[1, :])
        wait([x, y], return_when=ALL_COMPLETED)
        x = x.result()
        y = y.result()
        z, hx = nlms_alg(x, y, 1200, signal)
        impulse_resp.append(hx[0])
        u = np.conj(freqz(hx[0], 1, 24000)).T
        u_abs = np.where(np.abs(u) > 1.0e-10, np.abs(u), 1.0e-10)
        u_db = np.log10(u_abs).T * 20
        recordings_db.append(u_db[1])
        lines_label.append(m['legend'])

    return recordings_db, impulse_resp, lines_label


def nlms_alg_plot_residual(measurement, signal):
    executor = ThreadPoolExecutor(max_workers=2)
    [bx, ax] = ellip(2, 0.0005, 40, 20 / 24000, 'high')
    recordings_db = []
    lines_label = []
    for m in measurement['recording']:
        print(f"Handle wav file: {m['file']:s}")
        signal.status.emit(f"{m['file']:s} is processing...")
        fs, wav = wavfile.read(PurePath(m['file']))
        wav = wav.astype('float32')
        wav = wav.T[:, 240000::]
        # x = numba_lfilter(bx, ax, wav[0, :]) # 5e Err MIC FIR
        # y = numba_lfilter(bx, ax, wav[1, :]) # 5e Kemar MIC FIR
        x = executor.submit(lfilter, bx, ax, wav[0, :])
        y = executor.submit(lfilter, bx, ax, wav[1, :])
        wait([x, y], return_when=ALL_COMPLETED)
        x = x.result()
        y = y.result()
        print("(5e)error mic:")
        print(x)
        print("(5e)kemar mic:")
        print(y)
        z, hx = nlms_alg(x, y, 1200, signal) # Compensation by NLMS algorithm
        dd = y - lfilter(hx[0], 1, x) # Compute residual noise
        # u1 = numba_conj(numba_freqz(y, 1, 24000)).T
        # u2 = numba_conj(numba_freqz(dd, 1, 24000)).T
        u1_data = executor.submit(freqz, y, 1, 24000)
        u2_data = executor.submit(freqz, dd, 1, 24000)
        wait([u1_data, u2_data], return_when=ALL_COMPLETED)
        u1 = executor.submit(np.conj, u1_data.result())
        u2 = executor.submit(np.conj, u2_data.result())
        wait([u1, u2], return_when=ALL_COMPLETED)
        u1 = u1.result().T
        u2 = u2.result().T
        u1_abs = np.where(np.abs(u1) > 1.0e-10, np.abs(u1), 1.0e-10)
        u1_db = np.log10(u1_abs).T * 20
        u2_abs = np.where(np.abs(u2[0:24000]) > 1.0e-10, np.abs(u2[0:24000]), 1.0e-10)
        u2_db = np.log10(u2_abs).T * 20
        recordings_db.append(u1_db[1])
        lines_label.append(f"{m['legend']:s} Base")
        recordings_db.append(u2_db[1])
        lines_label.append(f"{m['legend']:s} Residual")

    return recordings_db, lines_label
    # s = range(0, 24000)
    # plt.subplot(1, 1, 1)
    # lines = plt.semilogx(s, u1_db[1] * 20, u2_db[1] * 20, linewidth=1)
    # plt.gca().title.set_text('Frequency Response (5e Recording Residual)')
    # plt.yticks(np.arange(-50, 50, step=10))
    # plt.axis([20, 24000, -50, 50])
    # plt.grid(True, which='both')
    # plt.show()


def main():
    if len(sys.argv) < 2:
        print("Usage:" + sys.argv[0] + '[measurement.json]')
        return

    measurements = read_measurement_json(sys.argv[1])
    print(measurements)
    figure_props = measurements['figure_properties']

    measurements_db = []
    lines_label = []
    for m in measurements['measurements']:
        print(m)
        m_db = []
        label = []
        if m['type'].lower() == "ANC_NORM".lower():
            m_db, labels = get_anc_normalization_curve(m, figure_props['bit_depth'])
        elif m['type'].lower() == "FREQ_RESP".lower():
            m_db, labels = get_frequency_response_curve(m, figure_props['bit_depth'])

        measurements_db.extend(m_db)
        lines_label.extend(labels)

    # Draw
    axis = figure_props['x_axis']
    axis.extend(figure_props['y_axis'])
    plt.title(figure_props['title'], fontsize=12)
    plt.xlabel(figure_props['x_label'], fontsize=8)
    plt.ylabel(figure_props['y_label'], fontsize=8)
    plt.axis(axis)
    plt.yticks(range(*figure_props['y_ticks']), fontsize=8)
    plt.xticks(fontsize=8)
    plt.grid(True, which='both')
    lines = []
    for c in measurements_db:
        total_samples = c['data'].shape[0]
        limit = int((total_samples / 2) - 1)
        freqs = fftfreq(c['data'].shape[0], 1 / c['fs'])
        line, = plt.semilogx(freqs[:limit], c['data'][:limit], linewidth=1)
        lines.append(line)

    plt.legend(lines, lines_label, fontsize=10)
    plt.show()


def draw(measurements, signal):
    print(measurements)

    nlms_measurments_db = []
    nlms_impulse = []
    nlms_lines_label = []
    measurements_db = []
    lines_label = []
    nlms_res_db = []
    nlms_res_lines_label = []
    figure_props = measurements['figure_properties']

    for m in measurements['measurements']:
        print(m)

        if m['type'].lower() == "ANC_NORM".lower():
            _db, _labels = get_anc_normalization_curve(m, figure_props['bit_depth'], signal)
            measurements_db.extend(_db)
            lines_label.extend(_labels)
        elif m['type'].lower() == "FREQ_RESP".lower():
            _db, _labels = get_frequency_response_curve(m, figure_props['bit_depth'], signal)
            measurements_db.extend(_db)
            lines_label.extend(_labels)
        elif m['type'].lower() == "NLMS".lower():
            _nlms_db, _nlms_impulse, _nlms_labels = nlms_alg_plot(m, signal)
            nlms_measurments_db.extend(_nlms_db)
            nlms_impulse.extend(_nlms_impulse)
            nlms_lines_label.extend(_nlms_labels)
        elif m['type'].lower() == "NLMS_RES".lower():
            _nlms_db, _nlms_labels = nlms_alg_plot_residual(m, signal)
            nlms_res_db.extend(_nlms_db)
            nlms_res_lines_label.extend(_nlms_labels)

    return [measurements_db, lines_label, nlms_impulse, nlms_measurments_db,
            nlms_lines_label, nlms_res_db, nlms_res_lines_label], figure_props


if __name__ == "__main__":
    main()
