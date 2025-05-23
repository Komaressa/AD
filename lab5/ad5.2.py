from bokeh.io import curdoc
from bokeh.models import ColumnDataSource, Slider, Select, CheckboxButtonGroup, Div
from bokeh.layouts import row, column
from bokeh.plotting import figure
import numpy as np

# === Дані ===
t = np.linspace(0, 4 * np.pi, 1000)
data = {'x': t, 'y_signal': np.zeros_like(t), 'y_filtered': np.zeros_like(t)}
source = ColumnDataSource(data=data)

# === Сигнал ===
def create_signal(ampl, freq, phase, noise_mu, noise_sigma, with_noise):
    y = ampl * np.sin(freq * t + phase)
    if with_noise:
        y += np.random.normal(noise_mu, noise_sigma, size=t.shape)
    return y

# === Власний фільтр: ковзне середнє ===
def my_filter(data, window):
    res = np.zeros_like(data)
    for i in range(len(data)):
        res[i] = np.mean(data[max(0, i - window + 1):i + 1])
    return res

# === Butterworth фільтр ===
def butter_filter(data, freq, wn_mult, fs_mult):
    from scipy.signal import iirfilter, lfilter
    fs = 1000 / (4 * np.pi) * fs_mult
    Wn = freq * 2 * wn_mult
    b, a = iirfilter(N=4, Wn=Wn, fs=fs, btype='low', ftype='butter')
    return lfilter(b, a, data)

# === Віджети ===
sliders = {
    'amplitude': Slider(title='Амплітуда', start=0.1, end=5.0, step=0.1, value=1),
    'frequency': Slider(title='Частота', start=0.5, end=10, step=0.1, value=2),
    'phase': Slider(title='Фаза', start=-np.pi, end=np.pi, step=0.1, value=0),
    'noise_mu': Slider(title='Сер. шуму', start=0, end=1, step=0.1, value=0),
    'noise_sigma': Slider(title='Дисперсія шуму', start=0, end=5, step=0.1, value=0)
}

checkboxes = CheckboxButtonGroup(labels=["Додати шум", "Фільтрувати", "Власний фільтр"], active=[])
wn_mult = Slider(title='Wn множник', start=0.5, end=2.0, step=0.1, value=1)
fs_mult = Slider(title='Fs множник', start=0.5, end=2.0, step=0.1, value=1)

# === Графіки ===
p_signal = figure(height=300, width=700, title="Сигнал")
p_signal.line('x', 'y_signal', source=source, line_width=2, color='blue')

p_filtered = figure(height=300, width=700, title="Фільтрований сигнал")
p_filtered.line('x', 'y_filtered', source=source, line_width=2, color='green')

# === Загальна логіка оновлення (без параметрів) ===
def update():
    amp = sliders['amplitude'].value
    freq = sliders['frequency'].value
    phase = sliders['phase'].value
    noise_mu = sliders['noise_mu'].value
    noise_sigma = sliders['noise_sigma'].value

    with_noise = 0 in checkboxes.active
    apply_filter = 1 in checkboxes.active
    use_custom_filter = 2 in checkboxes.active

    y = create_signal(amp, freq, phase, noise_mu, noise_sigma, with_noise)
    y_filtered = (
        my_filter(y, window=10)
        if (apply_filter and use_custom_filter)
        else butter_filter(y, freq, wn_mult.value, fs_mult.value)
        if apply_filter
        else np.zeros_like(t)
    )

    source.data = {'x': t, 'y_signal': y, 'y_filtered': y_filtered}

# Обгортка для Bokeh, з правильною сигнатурою
def update_handler(attr, old, new):
    update()

# Підключення обробників подій
for s in sliders.values():
    s.on_change('value', update_handler)
checkboxes.on_change('active', update_handler)
wn_mult.on_change('value', update_handler)
fs_mult.on_change('value', update_handler)

# === Інтерфейс ===
controls_col = column(
    Div(text="<h2>Налаштування</h2>"),
    *sliders.values(), checkboxes, wn_mult, fs_mult
)
layout = column(row(controls_col), p_signal, p_filtered)

update()  # Без аргументів — викликає "update", а не "update_handler"
curdoc().add_root(layout)
curdoc().title = "Гармонічний сигнал"
