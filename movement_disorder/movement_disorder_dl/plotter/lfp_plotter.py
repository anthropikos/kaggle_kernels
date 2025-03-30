
import numpy as np
import matplotlib.pyplot as plt

def plot_lfp(multi_channel_lfp:np.ndarray, first_n_sec=None, sampling_freq=None):

    if first_n_sec is None: first_n_sec = 0.5
    if sampling_freq is None: sampling_freq = 2048

    n_channels = multi_channel_lfp.shape[0]
    nrows, ncols, inches, xy_scale = n_channels, 1, 3, 10
    
    fig, axs = plt.subplots(
        nrows=n_channels, 
        ncols=ncols, 
        figsize=(ncols*inches*xy_scale, nrows*inches),
        sharex=True,
        # sharey=True,
    )
    
    for idx, dbs_channel_data in enumerate(multi_channel_lfp):
        ax = axs[idx]
        
        dbs_channel_data = dbs_channel_data[:int(first_n_sec*sampling_freq)]
        
        
        x = np.arange(len(dbs_channel_data)) / sampling_freq * 1000
        y = dbs_channel_data
        ax.scatter(x, y, marker='.', c=f'C{idx}')
        ax.plot(x, y, alpha=0.2, c=f'C{idx}')
        
        ax.set_xlabel('Time (milliseconds)')
        ax.set_ylabel('Membrane potential (micro-volt)')
        ax.set_title(f'DBS Channel {idx}')

    fig.suptitle(f'Multi-channel LFP', fontsize=12)
    fig.set_layout_engine('constrained')
    
    return fig