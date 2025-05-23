import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, CheckButtons
from scipy.signal import butter, filtfilt


class HarmonicNoiseGUI:
    """
    GUI for plotting a harmonic signal with optional noise and filtering.
    """
    DEFAULTS = {
        'amplitude': 1.0,
        'frequency': 2.0,
        'phase': 0.0,
        'noise_mean': 0.0,
        'noise_cov': 0.1
    }

    def __init__(self):
        # Cached noise and its parameters
        self._noise_cache = None
        self._cached_params = (None, None)

        # Time vector
        self.t = np.linspace(0, 2, 1000)

        # Initialize figure and axes
        self.fig, self.ax = plt.subplots()
        plt.subplots_adjust(left=0.25, bottom=0.45)

        # Initial signal lines (pure, noisy, filtered)
        init_pure = self._generate_pure(**self.DEFAULTS)
        self.line_noisy, = self.ax.plot(self.t, init_pure, label='Noisy Signal')
        self.line_pure, = self.ax.plot(self.t, init_pure, '--', label='Pure Signal')
        self.line_filtered, = self.ax.plot(self.t, np.full_like(self.t, np.nan), label='Filtered Signal')

        self._setup_widgets()
        self._configure_plot()

    def _generate_pure(self, amplitude, frequency, phase, **_):
        return amplitude * np.sin(2 * np.pi * frequency * self.t + phase)

    def _generate_noise(self, mean, cov):
        # Generate new noise only if parameters changed
        if (mean, cov) != self._cached_params or self._noise_cache is None:
            self._noise_cache = np.random.normal(mean, np.sqrt(cov), size=self.t.shape)
            self._cached_params = (mean, cov)
        return self._noise_cache

    def _butter_filter(self, data, cutoff=3.0, fs=500.0, order=4):
        b, a = butter(order, cutoff / (0.5 * fs), btype='low')
        return filtfilt(b, a, data)

    def _setup_widgets(self):
        axcolor = 'lightgoldenrodyellow'
        slider_defs = [
            ('Amplitude', 0.1, 5.0, self.DEFAULTS['amplitude']),
            ('Frequency', 0.1, 10.0, self.DEFAULTS['frequency']),
            ('Phase', 0.0, 2 * np.pi, self.DEFAULTS['phase']),
            ('Noise Mean', -1.0, 1.0, self.DEFAULTS['noise_mean']),
            ('Noise Cov', 0.01, 1.0, self.DEFAULTS['noise_cov']),
        ]
        self.sliders = {}
        for i, (label, mn, mx, init) in enumerate(slider_defs):
            ax = plt.axes([0.25, 0.35 - i * 0.05, 0.65, 0.03], facecolor=axcolor)
            slider = Slider(ax, label, mn, mx, valinit=init)
            slider.on_changed(self._update_plot)
            self.sliders[label.lower().replace(' ', '_')] = slider

        # Reset button
        btn_ax = plt.axes([0.8, 0.025, 0.1, 0.04])
        self.button = Button(btn_ax, 'Reset', color=axcolor, hovercolor='0.975')
        self.button.on_clicked(self._reset)

        # Checkboxes for noise and filtering
        chk_ax = plt.axes([0.025, 0.6, 0.15, 0.15])
        labels = ['Show Noise', 'Show Filter']
        states = [True, False]
        self.check = CheckButtons(chk_ax, labels, states)
        self.check.on_clicked(self._update_plot)

    def _configure_plot(self):
        self.ax.set_title('Harmonic Signal with Noise and Filtering')
        self.ax.set_xlabel('Time')
        self.ax.set_ylabel('Amplitude')
        self.ax.legend(loc='upper right')

    def _update_plot(self, _=None):
        # Read parameters
        amp = self.sliders['amplitude'].val
        freq = self.sliders['frequency'].val
        phase = self.sliders['phase'].val
        mean = self.sliders['noise_mean'].val
        cov = self.sliders['noise_cov'].val
        show_noise, show_filt = self.check.get_status()

        # Generate signals
        pure = self._generate_pure(amp, freq, phase)
        noise = self._generate_noise(mean, cov)
        noisy = pure + noise
        display_signal = noisy if show_noise else pure
        filtered = self._butter_filter(display_signal)

        # Update plot data
        self.line_pure.set_ydata(pure)
        self.line_noisy.set_ydata(display_signal)
        self.line_filtered.set_ydata(filtered if show_filt else np.full_like(self.t, np.nan))

        self.fig.canvas.draw_idle()

    def _reset(self, _=None):
        # Reset sliders
        for slider in self.sliders.values():
            slider.reset()
        # Reset checkboxes to default
        if not self.check.get_status()[0]:
            self.check.set_active(0)
        if self.check.get_status()[1]:
            self.check.set_active(1)
        # Force plot update
        self._update_plot()

    def run(self):
        plt.show()


if __name__ == '__main__':
    gui = HarmonicNoiseGUI()
    gui.run()
