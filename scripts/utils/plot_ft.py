import numpy as np
from matplotlib import pyplot as plt


def numpy_ewma_vectorized(data, window):
    alpha = 2 / (window + 1.0)
    alpha_rev = 1-alpha

    scale = 1/alpha_rev
    n = data.shape[0]

    r = np.arange(n)
    scale_arr = scale**r
    offset = data[0]*alpha_rev**(r+1)
    pw0 = alpha*alpha_rev**(n-1)

    mult = data*pw0*scale_arr
    cumsums = mult.cumsum()
    out = offset + cumsums*scale_arr[::-1]
    return out


def smooth(scalars, weight):  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = []
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value
    return np.array(smoothed)


def plot_data(filepath, color, label, start=0, end=-1, alpha=0.25):
    ft = np.load(filepath)
    contact = np.linalg.norm(np.abs(ft[:, :, 6:9]), axis=2)
    mean = np.mean(contact, axis=0)[start:start + end]
    mean = smooth(mean, 0.6)
    # mean = numpy_ewma_vectorized(mean, 5)
    std = np.std(contact, axis=0)[start:start + end]

    t = np.linspace(0, std.shape[0], mean.shape[0]) / 40.0
    plt.plot(t, mean, color=color, label=label, linewidth=1)
    plt.fill_between(t, np.maximum(0, mean-std), mean+std, antialiased=True, linewidth=0.5, alpha=alpha, facecolor=color)


# # ACT VS Impedance Controller
demo_ft = "/root/osx-ur/dependencies/act/datasets/sim_bimanual_wiping_ds_contact_force.npy"
no_ft = "/root/osx-ur/dependencies/act/rollouts/sim_bimanual_wiping/no_ft/contact_force.npy"
ft = "/root/osx-ur/dependencies/act/rollouts/sim_bimanual_wiping/ft/contact_force.npy"
pure_act = "/root/osx-ur/dependencies/act/rollouts/sim_bimanual_wiping_joint/no_ft/contact_force.npy"
act_ft = "/root/osx-ur/dependencies/act/rollouts/sim_bimanual_wiping_joint/ft2/contact_force.npy"

# # ACT VS Compliance Controller
# demo_ft = "/root/osx-ur/dependencies/act/datasets/sim_bimanual_wiping_ds_contact_force.npy"
# no_ft = "/root/osx-ur/dependencies/act/rollouts/sim_bimanual_wiping_compliance/no_ft/contact_force.npy"
# ft = "/root/osx-ur/dependencies/act/rollouts/sim_bimanual_wiping_compliance/ft/contact_force.npy"
# pure_act = "/root/osx-ur/dependencies/act/rollouts/sim_bimanual_wiping_joint/no_ft/contact_force.npy"
# act_ft = "/root/osx-ur/dependencies/act/rollouts/sim_bimanual_wiping_joint/ft2/contact_force.npy"

plt.figure(figsize=(9, 5))

plot_data(demo_ft, "dimgray", "Demonstrations", start=280, end=300)
# plot_data(demo_ft, "tab:cyan", "Demonstrations Joints", start=280, end=300)
plot_data(pure_act, "tab:green", "ACT", start=400, end=300, alpha=0.1)
plot_data(act_ft, "darkgreen", "ACT + F/T", start=380, end=300, alpha=0.1)
plot_data(no_ft, "tab:red", "Comp-ACT w/o F/T", start=280, end=300)
plot_data(ft, "tab:blue", "Comp-ACT", start=280, end=300)

handles, labels = plt.gca().get_legend_handles_labels()
order = [1, 2, 4, 3, 0]

# Increase tick size and label size
plt.tick_params(axis='both', which='major', labelsize=14, width=2, length=6)
plt.tick_params(axis='both', which='minor', labelsize=14, width=1, length=4)
plt.figure
plt.ylabel("Normal Contact Force (N)", fontsize=14)
plt.xlabel("Time (s)", fontsize=14)
plt.ylim(-1, 30)
plt.grid()
plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order], fontsize=14)
# plt.legend()
plt.tight_layout()

plt.savefig("/root/osx-ur/dependencies/act/rollouts/sim_bimanual_wiping_ft_comparison.png", dpi=600)
plt.show()
