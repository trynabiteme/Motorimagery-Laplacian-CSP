
#
import numpy as np
import mne
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import roc_auc_score
from mne.decoding import CSP
from scipy import signal

# %%
# --- Step 1. Import raw BrainVision data ---
# vhdr_file = os.path.join('data', 'calibration_motorimageryVPkg.vhdr')
# Update this path to your actual file:
vhdr_file = 'calibration_motorimageryVPkg.vhdr'
raw = mne.io.read_raw_brainvision(vhdr_file, preload=True)

print(f"Loaded raw data: {len(raw.ch_names)} channels, {raw.n_times} samples.")

# Set the montage: use file-embedded montage if available; otherwise, use a standard 10-20 montage.
montage = raw.get_montage()
if montage is None:
    montage = mne.channels.make_standard_montage('standard_1020')
    raw.set_montage(montage)
    print("No montage found in file; using standard_1020 montage.")
else:
    print("Montage loaded from file.")


#laplacian filter
# Create Laplacian filtered channels for C3 and C4
c3_neigh = ['FC3', 'C1', 'CP3', 'C5']
c3_main = raw.copy().pick(['C3']).get_data()[0]
c3_neighbors = raw.copy().pick(c3_neigh).get_data()
lap_c3= c3_main-c3_neighbors.mean(axis=0)


c4_neigh = ['FC4', 'C2', 'CP4', 'C6']
c4_main=raw.copy().pick(['C4']).get_data()[0]
c4_neighbors=raw.copy().pick(c4_neigh).get_data()
lap_c4 = c4_main-c4_neighbors.mean(axis=0)
# Combine Laplacian channels and add to raw object
ch_lap_info = mne.create_info(ch_names = ['C3 lap', 'C4 lap'],sfreq=raw.info['sfreq'],ch_types='eeg')

lap_raw_array  = mne.io.RawArray(np.vstack((lap_c3, lap_c4)), ch_lap_info)
raw.add_channels([lap_raw_array], force_update_info=True)

# %%
# --- Step 2. Epoch the data ---
# Extract events from annotations.
# MNE returns an event_id dictionary mapping annotation descriptions to integer codes.
events, event_id = mne.events_from_annotations(raw)
print(f"Extracted {len(events)} events with event_id: {event_id}")

# Define new event mapping based on your event_id output.
# Here, we assume "Stimulus/S  1" corresponds to 'left' and "Stimulus/S  2" to 'right'.
if "Stimulus/S  1" in event_id and "Stimulus/S  2" in event_id:
    new_event_id = {"left": event_id["Stimulus/S  1"], "right": event_id["Stimulus/S  2"]}
else:
    raise ValueError(
        "Expected keys ('Stimulus/S  1' and 'Stimulus/S  2') not found in event_id. Check annotation codes.")

# Define the epoching time window (in seconds) relative to each event.
tmin, tmax = -0.5, 5  # e.g., 500ms to 5s post-stimulus

# Create epochs containing only the 'left' and 'right' events.
epochs = mne.Epochs(raw, events, event_id=new_event_id, tmin=tmin, tmax=tmax,
                    baseline=(None, 0), preload=True)
print("Created epochs:")
print(epochs)
print(print(mne.__version__))

#%%

epochs_cropped = epochs.copy().crop(tmin=0.75, tmax=3.5)
samples = len(epochs_cropped.times)  # number of samples in the cropped epoch
#  Compute Power Spectral Density (PSD)
all_channels = ['C3', 'C4', 'C3 lap', 'C4 lap']
psd_data = {}
avg_psd = {}

samples = len(epochs.times)  # use epoch length for n_per_seg (zero-padding enabled)

for cond in new_event_id.keys():
    epochs_cond = epochs[cond]
    psds, freqs = mne.time_frequency.psd_welch(
        epochs_cond, picks=all_channels, fmin=1, fmax=40,
        n_fft=256, n_per_seg=samples, average=None, verbose=False)
    psd_data[cond] = psds
    avg_psd[cond] = np.mean(psds, axis=(0, -1))
    print(f"Condition '{cond}': computed PSD for {len(epochs_cond)} epochs.")


#%%
# --- Step 4. Compute AUC at each frequency bin for each channel ---

auc_data = {}

if "left" in psd_data and "right" in psd_data:
    for ch_idx, ch in enumerate(all_channels):
        auc_vals = []
        for fi in range(len(freqs)):
            left_vals = psd_data["left"][:, ch_idx, fi]
            right_vals = psd_data["right"][:, ch_idx, fi]
            X = np.concatenate([left_vals, right_vals])
            Y = np.concatenate([np.ones_like(left_vals), np.zeros_like(right_vals)])
            try:
                auc = roc_auc_score(Y, X)
                if auc < 0.5:
                    auc = 1 - auc
            except ValueError:
                auc = np.nan
            auc_vals.append(auc)
        auc_data[ch] = np.array(auc_vals)

# Compute a common color scale for AUC across channels.
common_vmin = min(np.nanmin(auc_data[ch]) for ch in all_channels)
common_vmax = max(np.nanmax(auc_data[ch]) for ch in all_channels)

#Plot raw C3 and C4 PSD and AUC
fig = plt.figure(figsize=(12, 8))
gs = gridspec.GridSpec(2, 3,
                       width_ratios=[1, 1, 0.05],
                       height_ratios=[3, 1],
                       wspace=0.3, hspace=0.4)

ax_psd0 = fig.add_subplot(gs[0, 0])  # PSD for channel C3 (top left)
ax_psd1 = fig.add_subplot(gs[0, 1])  # PSD for channel C4 (top right)
ax_auc0 = fig.add_subplot(gs[1, 0])   # AUC for channel C3 (bottom left)
ax_auc1 = fig.add_subplot(gs[1, 1])   # AUC for channel C4 (bottom right)

ax_psd0.plot(freqs, 10 * np.log10(avg_psd["left"][0, :]), label="left", color='blue')
ax_psd0.plot(freqs, 10 * np.log10(avg_psd["right"][0, :]), label="right", color='red')
ax_psd0.set_title("PSD - Channel C3")
ax_psd0.set_xlabel("Frequency (Hz)")
ax_psd0.set_ylabel("Power (dB)")
ax_psd0.grid(True)
ax_psd0.legend(loc='upper right')

ax_psd1.plot(freqs, 10 * np.log10(avg_psd["left"][1, :]), label="left", color='blue')
ax_psd1.plot(freqs, 10 * np.log10(avg_psd["right"][1, :]), label="right", color='red')
ax_psd1.set_title("PSD - Channel C4")
ax_psd1.set_xlabel("Frequency (Hz)")
ax_psd1.set_ylabel("Power (dB)")
ax_psd1.grid(True)
ax_psd1.legend(loc='upper right')

ax_auc0.plot(freqs, auc_data["C3"], color='black', lw=2)
ax_auc0.set_title("AUC vs Frequency - C3")
ax_auc0.set_xlabel("Frequency (Hz)")
ax_auc0.set_ylabel("AUC")
ax_auc0.set_ylim(common_vmin, common_vmax)
ax_auc0.grid(True)

ax_auc1.plot(freqs, auc_data["C4"], color='black', lw=2)
ax_auc1.set_title("AUC vs Frequency - C4")
ax_auc1.set_xlabel("Frequency (Hz)")
ax_auc1.set_ylabel("AUC")
ax_auc1.set_ylim(common_vmin, common_vmax)
ax_auc1.grid(True)

plt.tight_layout(rect=[0, 0, 0.95, 1])
plt.savefig('raw_channels.pdf')
plt.show()
plt.close('all')

# Plot Laplacian channels PSD and AUC
fig = plt.figure(figsize=(16, 10))
gs = gridspec.GridSpec(2, 3,
                       width_ratios=[1, 1, 0.05],
                       height_ratios=[3, 1],
                       wspace=0.3, hspace=0.4)

ax_ch3 = fig.add_subplot(gs[0, 0])
ax_ch4 = fig.add_subplot(gs[0, 1])
ax_ch3_auc = fig.add_subplot(gs[1, 0])
ax_ch4_auc = fig.add_subplot(gs[1, 1])

ax_ch3.plot(freqs, 10 * np.log10(avg_psd["left"][2, :]), label="left", color='blue')
ax_ch3.plot(freqs, 10 * np.log10(avg_psd["right"][2, :]), label="right", color='red')
ax_ch3.set_title("PSD - Channel C3 lap")
ax_ch3.set_xlabel("Frequency (Hz)")
ax_ch3.set_ylabel("Power (dB)")
ax_ch3.grid(True)
ax_ch3.legend(loc='upper right')

ax_ch4.plot(freqs, 10 * np.log10(avg_psd["left"][3, :]), label="left", color='blue')
ax_ch4.plot(freqs, 10 * np.log10(avg_psd["right"][3, :]), label="right", color='red')
ax_ch4.set_title("PSD - Channel C4 lap")
ax_ch4.set_xlabel("Frequency (Hz)")
ax_ch4.set_ylabel("Power (dB)")
ax_ch4.grid(True)
ax_ch4.legend(loc='upper right')

ax_ch3_auc.plot(freqs, auc_data["C3 lap"], color='black', lw=2)
ax_ch3_auc.set_title("AUC vs Frequency - C3 lap")
ax_ch3_auc.set_xlabel("Frequency (Hz)")
ax_ch3_auc.set_ylabel("AUC")
ax_ch3_auc.set_ylim(common_vmin, common_vmax)
ax_ch3_auc.grid(True)

ax_ch4_auc.plot(freqs, auc_data["C4 lap"], color='black', lw=2)
ax_ch4_auc.set_title("AUC vs Frequency - C4 lap")
ax_ch4_auc.set_xlabel("Frequency (Hz)")
ax_ch4_auc.set_ylabel("AUC")
ax_ch4_auc.set_ylim(common_vmin, common_vmax)
ax_ch4_auc.grid(True)

plt.savefig('laplace_channels.pdf')
plt.show()
plt.close('all')


#Apply CSP for class separation
filt = raw.copy().filter(l_freq=9, h_freq=13, method='iir')
event_filtered, event_id_filtered = mne.events_from_annotations(filt)
new_event_id_filtered = {"left": event_id_filtered["Stimulus/S  1"], "right": event_id_filtered["Stimulus/S  2"]}

epochs_cropped = mne.Epochs(filt, event_filtered,event_id=new_event_id_filtered,tmin=0.5, tmax=3.5,baseline=None, preload=True)
epochs_left = epochs_cropped['left']
epochs_right = epochs_cropped['right']

picks = mne.pick_types(epochs_left.info, eeg=True, exclude=['C3 lap', 'C4 lap'])

left_x = epochs_left.get_data(picks=picks)
right_x = epochs_right.get_data(picks=picks)
X = np.concatenate([left_x, right_x], axis=0)
Y = np.concatenate([np.ones(len(left_x)), np.zeros(len(right_x))])
csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)
csp.fit(X, Y)
filters = csp.filters_
patterns = csp.patterns_
weigh= np.array([filters[0], filters[1]])
cspInfo= mne.create_info(['CSP1', 'CSP2'], raw.info['sfreq'], ch_types='eeg')
epochs_csp1_left, epochs_csp2_left = zip(*[(weigh[0] @ ep, weigh[1] @ ep) for ep in left_x])
epochs_csp1_right, epochs_csp2_right = zip(*[(weigh[0] @ ep, weigh[1] @ ep) for ep in right_x])
epochs_csp1_right = np.stack(epochs_csp1_right)
epochs_csp2_right = np.stack(epochs_csp2_right)
epochs_csp1_left = np.stack(epochs_csp1_left)
epochs_csp2_left = np.stack(epochs_csp2_left)

freqs_csp = np.linspace(1, 40, 100)

def psd_csp(epochs_csp1, epochs_csp2, freqs, fs):
    psd1, psd2 = [], []
    for signal1, signal2 in zip(epochs_csp1, epochs_csp2):
        f1, p1 = signal.welch(signal1, fs=fs, nperseg=256)
        f2, p2 = signal.welch(signal2, fs=fs, nperseg=256)
        psd1.append(np.interp(freqs, f1, p1))
        psd2.append(np.interp(freqs, f2, p2))
    return np.vstack(psd1), np.vstack(psd2)

psd_csp1_left, psd_csp2_left = psd_csp(epochs_csp1_left, epochs_csp2_left, freqs_csp, raw.info['sfreq'])
psd_csp1_right, psd_csp2_right = psd_csp(epochs_csp1_right, epochs_csp2_right, freqs_csp, raw.info['sfreq'])

avg_psd_csp1_left = psd_csp1_left.mean(axis=0)
avg_psd_csp2_left = psd_csp2_left.mean(axis=0)
avg_psd_csp1_right = psd_csp1_right.mean(axis=0)
avg_psd_csp2_right = psd_csp2_right.mean(axis=0)

psd_csp1_left = np.array(psd_csp1_left)
psd_csp1_right = np.array(psd_csp1_right)
psd_csp2_left = np.array(psd_csp2_left)
psd_csp2_right = np.array(psd_csp2_right)

def auc_csp(left_psd, right_psd):
    auc_values = []
    for left_freq_bin, right_freq_bin in zip(left_psd.T, right_psd.T):
        y_true = np.concatenate([np.ones_like(left_freq_bin), np.zeros_like(right_freq_bin)])
        y_scores = np.concatenate([left_freq_bin, right_freq_bin])

        score = roc_auc_score(y_true, y_scores)
        auc_values.append(score if score >= 0.5 else 1 - score)

    return np.array(auc_values)

auc_for_csp1 = auc_csp(psd_csp1_left, psd_csp1_right)
auc_for_csp2 = auc_csp(psd_csp2_left, psd_csp2_right)


#%%

# --- Step 5. Visualization using gridspec ---
# Create a gridspec with 2 rows and 2 columns.
# Columns 0 and 1: PSD (top) and AUC (bottom) for each channel.

fig_csp = plt.figure(figsize=(16, 10))
gs_csp = gridspec.GridSpec(2, 2,
                           width_ratios=[1, 1],
                           height_ratios=[3, 1],
                           wspace=0.3, hspace=0.4)

csp1_pds = fig_csp.add_subplot(gs_csp[0, 0])
csp2_pds = fig_csp.add_subplot(gs_csp[0, 1])
ax_auc_csp1 = fig_csp.add_subplot(gs_csp[1, 0])
ax_auc_csp2 = fig_csp.add_subplot(gs_csp[1, 1])

csp1_pds.plot(freqs_csp, 10 * np.log10(avg_psd_csp1_left), label="left", color='blue')
csp1_pds.plot(freqs_csp, 10 * np.log10(avg_psd_csp1_right), label="right", color='red')
csp1_pds.set_title("PSD - CSP1")
csp1_pds.set_xlabel("Frequency (Hz)")
csp1_pds.set_ylabel("Power (dB)")
csp1_pds.grid(True)
csp1_pds.legend(loc='upper right')

csp2_pds.plot(freqs_csp, 10 * np.log10(avg_psd_csp2_left), label="left", color='blue')
csp2_pds.plot(freqs_csp, 10 * np.log10(avg_psd_csp2_right), label="right", color='red')
csp2_pds.set_title("PSD - CSP2")
csp2_pds.set_xlabel("Frequency (Hz)")
csp2_pds.set_ylabel("Power (dB)")
csp2_pds.grid(True)
csp2_pds.legend(loc='upper right')

ax_auc_csp1.plot(freqs_csp, auc_for_csp1, color='black', lw=2)
ax_auc_csp1.set_title("AUC vs Frequency - CSP1")
ax_auc_csp1.set_xlabel("Frequency (Hz)")
ax_auc_csp1.set_ylabel("AUC")
ax_auc_csp1.grid(True)

ax_auc_csp2.plot(freqs_csp, auc_for_csp2, color='black', lw=2)
ax_auc_csp2.set_title("AUC vs Frequency - CSP2")
ax_auc_csp2.set_xlabel("Frequency (Hz)")
ax_auc_csp2.set_ylabel("AUC")
ax_auc_csp2.grid(True)

plt.savefig('csp_channels.pdf')
plt.show()
plt.close('all')
