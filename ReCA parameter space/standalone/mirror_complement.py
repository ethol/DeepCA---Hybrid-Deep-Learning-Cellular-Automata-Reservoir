import evodynamic.experiment as experiment
import evodynamic.connection.cellular_automata as ca
import evodynamic.cells.activation as act
import evodynamic.connection as connection
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np

width = 50
height_fig = 25
normal = [(a,) for a in [110]]
reflection = [(a,) for a in [124]]
complement = [(a,) for a in [137]]
reflection_complement = [(a,) for a in [193]]

# init = np.random.randint(2, size=(width, 1)).astype(np.float64)
init = np.zeros((width, 1), dtype=np.float64)
init[int(len(init)/2)] = 1.0
init_r = np.array(init, copy=True)
init_c = np.array(init, copy=True)
init_rc = np.array(init, copy=True)

for i in range(0, len(init)):
    init_r[i] = init[len(init) - i - 1]
    init_rc[i] = init[len(init) - i - 1]

for i in range(0, len(init)):
    if init[i] == 1.0:
        init_c[i] = 0.0
    else:
        init_c[i] = 1.0

for i in range(0, len(init)):
    if init_rc[i] == 1.0:
        init_rc[i] = 0.0
    else:
        init_rc[i] = 1.0


def runCA(rule, init):
    exp = experiment.Experiment()
    g_ca = exp.add_group_cells(name="g_ca", amount=width)
    neighbors, center_idx = ca.create_pattern_neighbors_ca1d(3)
    g_ca_bin = g_ca.add_binary_state(state_name='g_ca_bin', init=init)
    g_ca_bin_conn = ca.create_conn_matrix_ca1d('g_ca_bin_conn', width, \
                                               neighbors=neighbors, \
                                               center_idx=center_idx)

    exp.add_connection("g_ca_conn",
                       connection.WeightedConnection(g_ca_bin, g_ca_bin,
                                                     act.rule_binary_ca_1d_width3_func,
                                                     g_ca_bin_conn, fargs_list=rule))

    exp.initialize_cells()

    im_ca = np.zeros((height_fig, width))
    im_ca[0] = exp.get_group_cells_state("g_ca", "g_ca_bin")[:, 0]

    for i in range(1, height_fig):
        exp.run_step()
        im_ca[i] = exp.get_group_cells_state("g_ca", "g_ca_bin")[:, 0]

    exp.close()
    return im_ca


im_ca_n = runCA(normal, init)
im_ca_r = runCA(reflection, init_r)
im_ca_c = runCA(complement, init_c)
im_ca_rc = runCA(reflection_complement, init_rc)

fig, axes = plt.subplots(2, 2)
plt.subplots_adjust(wspace=0.10, hspace=-0.2)

axes[0][0].imshow(im_ca_n, cmap="binary")
axes[0][0].axis('off')
axes[0][0].set_title(f'Rule {normal[0][0]} ')
axes[0][1].imshow(im_ca_r, cmap="binary")
axes[0][1].axis('off')
axes[0][1].set_title(f'Rule {complement[0][0]} (reflection)')
axes[1][0].imshow(im_ca_c, cmap="binary")
axes[1][0].axis('off')
axes[1][0].set_title(f'Rule {reflection[0][0]} (complement)')
axes[1][1].imshow(im_ca_rc, cmap="binary")
axes[1][1].axis('off')
axes[1][1].set_title(f'Rule {reflection_complement[0][0]} (ref. & comp.)')


# plt.title(f'Rule {fargs_list[0][0]}, {height_fig} step')

# plt.show()

fig.savefig(f'mirror_complement.png', bbox_inches="tight", dpi=300)
