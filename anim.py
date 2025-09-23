import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter

def create_anim_gif(name: str, frames) -> None:
    fig, ax_plot = plt.subplots(1, 1, figsize=(5, 5))
    fig.tight_layout(pad=2)
    # ax_plot.set_xlim(1e3, 10000)
    ax_plot.set_ylim(0, 0.5)
    ax_plot.grid()
    line, = ax_plot.plot(frames[0][0], frames[0][1], 'k-', marker='v', markerfacecolor='blue', markeredgewidth=0, label='Model')
    ax_plot.set_title(name)
    writer = PillowWriter(fps=3)

    label = fig.text(0.99, 0.01, '', ha='right', va='bottom')

    with writer.saving(fig, f'{name}.gif', 100):
        for i in range(len(frames)):
            line.set_data(frames[i][0], frames[i][1])
            label.set_text(f'frame={i}')
            writer.grab_frame()
    plt.clf()