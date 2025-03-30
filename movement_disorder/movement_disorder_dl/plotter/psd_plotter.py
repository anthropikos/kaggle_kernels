
def plot_psd_ax(psd, ax, title, label):
    ax.plot(psd, label=label )
    ax.set_yscale('log')
    ax.legend()
    ax.set_xlabel('Time (milliseconds)')
    ax.set_ylabel('Power Density (log-scale) (a.u.)')
    ax.set_title(str(title))
    return ax