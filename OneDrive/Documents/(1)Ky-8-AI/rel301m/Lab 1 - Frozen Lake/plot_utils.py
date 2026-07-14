# Draw greedy policy from state value function
import numpy as np
import matplotlib.pyplot as plt
    
def plot_policy(policy, arrows, label = ''):
    h = np.array(policy).shape[0]
    w = np.array(policy).shape[1]
    # actions: up -0, down - 1, left - 2, right- 3
    fig, ax = plt.subplots(figsize=(w, h))
    for r, row in enumerate(policy):
        for c, cell in enumerate(row):
            if np.max(np.array(cell)) == 0:
                ax.add_patch(plt.Circle((c, h-r),0.2))
                continue
            for idx, value in enumerate(cell):
                if value > 0: 
                    scale = 0.3 * value
                    plt.arrow(c, h-r, scale*arrows[str(idx)][0], scale*arrows[str(idx)][1], head_width=0.1)
    plt.axis([0.5, w-0.5, 0.5, h+0.5])
    plt.xticks(np.arange(-0.5, w-0.5, 1.0))
    plt.yticks(np.arange(0.5, h+0.5, 1.0))
    plt.title(label)
    plt.grid()
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    plt.show()
    #return greedy_policy