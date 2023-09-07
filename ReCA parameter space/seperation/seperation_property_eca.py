import random

import evodynamic.experiment as experiment
import evodynamic.connection.cellular_automata as ca
import evodynamic.connection.random_boolean_net as rbn
import evodynamic.connection.custom as conn_custom
import evodynamic.cells.activation as act
import evodynamic.connection as connection
import matplotlib.pyplot as plt
import numpy
from matplotlib import colors
import numpy as np
import pandas as pd
from datetime import datetime
from rapidfuzz.distance import DamerauLevenshtein
import tensorflow as tf

exp = pd.read_csv('data/exp.csv')
exp_run = pd.read_csv('data/exp_run.csv')
run = pd.read_csv('data/run.csv')


def ca_run(init, width, ca_rule):
    exp = experiment.Experiment()
    g_ca = exp.add_group_cells(name="g_ca", amount=width)
    neighbors, center_idx = ca.create_pattern_neighbors_ca1d(3)
    g_ca_bin = g_ca.add_binary_state(state_name='g_ca_bin', init=init)
    g_ca_bin_conn = ca.create_conn_matrix_ca1d('g_ca_bin_conn', width,
                                               neighbors=neighbors,
                                               center_idx=center_idx)

    exp.add_connection("g_ca_conn",
                       connection.WeightedConnection(g_ca_bin, g_ca_bin,
                                                     act.rule_binary_ca_1d_width3_func,
                                                     g_ca_bin_conn, fargs_list=ca_rule))

    exp.initialize_cells()

    im_ca = np.zeros((height_fig, width))
    im_ca[0] = exp.get_group_cells_state("g_ca", "g_ca_bin")[:, 0]

    for i in range(1, height_fig):
        exp.run_step()
        im_ca[i] = exp.get_group_cells_state("g_ca", "g_ca_bin")[:, 0]

    exp.close()
    return im_ca


def hhrbn_run(init, width, ca_rule, matrix):
    tf.compat.v1.disable_eager_execution()

    exp = experiment.Experiment()
    g_rbn = exp.add_group_cells(name="g_rbn", amount=width)
    g_rbn_bin = g_rbn.add_binary_state(state_name='g_rbn_bin', init=init)

    g_rbn_bin_conn = conn_custom.create_custom_matrix('g_rbn_bin_conn', matrix)

    exp.add_connection("g_rbn_conn",
                       connection.WeightedConnection(g_rbn_bin, g_rbn_bin, act.rule_binary_ca_1d_width3_func,
                                                     g_rbn_bin_conn,
                                                     fargs_list=ca_rule))

    exp.initialize_cells()

    im_ca = np.zeros((height_fig, width))
    im_ca[0] = exp.get_group_cells_state("g_rbn", "g_rbn_bin")[:, 0]

    for i in range(1, height_fig):
        exp.run_step()
        im_ca[i] = exp.get_group_cells_state("g_rbn", "g_rbn_bin")[:, 0]

    exp.close()

    return im_ca


complex = [41, 54, 106, 110]
chaotic = [18, 22, 30, 45, 122, 126, 146, 60, 90, 105, 150]
periodic = [2, 7, 9, 10, 11, 15, 24, 25, 26, 34, 35, 42, 46, 56, 57,
            58, 62, 94, 108, 130, 138, 152, 154, 162, 170, 178,
            184, 1, 3, 4, 5, 6, 13, 14, 27, 28, 29, 33, 37, 38, 43,
            44, 72, 73, 74, 77, 78, 104, 132, 134, 140, 142,
            156, 164, 172, 12, 19, 23, 36, 50, 51, 76, 200, 204, 232]
rules = periodic

for rule in rules:
    number_of_runs = 100
    ca_width = 100
    height_fig = 500

    sub_ca = False
    central_locked = True

    ca_rule = [(a,) for a in [rule]]

    hamming_list_1bit = []
    hamming_list_5bit = []
    hamming_list_9bit = []
    hamming_list_5bit_rand = []
    hamming_list_9bit_rand = []

    DL_list_1bit = []
    DL_list_5bit = []
    DL_list_9bit = []
    DL_list_5bit_rand = []
    DL_list_9bit_rand = []

    im_ca_n = []
    im_ca_alt = []
    exp_row = {"id": exp["id"].max() + 1 if not exp.empty else 0,
               "width": ca_width,
               "height": height_fig,
               "start": pd.Timestamp.now(),
               "end": 0,
               "rule": ca_rule[0][0],
               "substrate": "ECA" if sub_ca else "PLCA" if central_locked else "HHRBN",
               "total_runs": number_of_runs}

    for i in range(0, number_of_runs):
        init = np.random.randint(2, size=(ca_width, 1))

        init_alt = [np.array(init, copy=True), np.array(init, copy=True), np.array(init, copy=True),
                    np.array(init, copy=True),
                    np.array(init, copy=True)]

        # change a central bit
        init_alt[0][int(ca_width / 2),] = init_alt[0][int(ca_width / 2),] ^ 1

        # change the central 5 bits
        init_alt[1][int(ca_width / 2) - 3:int(ca_width / 2) + 2, ] = \
            init_alt[1][int(ca_width / 2) - 3:int(ca_width / 2) + 2, ] ^ 1

        # change the central 9 bits
        init_alt[2][int(ca_width / 2) - 5:int(ca_width / 2) + 4, ] = \
            init_alt[2][int(ca_width / 2) - 5:int(ca_width / 2) + 4, ] ^ 1

        # change 5 random non overlapping bits
        past = []
        count = 0
        while count < 5:
            rand = random.randint(0, ca_width - 1)
            if rand not in past:
                count += 1
                past.append(rand)
                init_alt[3][rand] = init_alt[3][rand] ^ 1

        # change 9 random non overlapping bits
        past = []
        count = 0
        while count < 9:
            rand = random.randint(0, ca_width - 1)
            if rand not in past:
                count += 1
                past.append(rand)
                init_alt[4][rand] = init_alt[4][rand] ^ 1

        im_ca_alt = [[], [], [], [], []]



        if sub_ca:
            im_ca_n = ca_run(init, ca_width, ca_rule)

            for j in range(0, len(init_alt)):
                im_ca_alt[j] = ca_run(init_alt[j], ca_width, ca_rule)

        else:
            conn_matrix = np.zeros(shape=(ca_width, ca_width))
            count = 0
            for arr in conn_matrix:
                arr[random.randint(0, ca_width - 1)] += 1

                if central_locked:
                    arr[count] += 2
                else:
                    arr[random.randint(0, ca_width-1)] += 2

                arr[random.randint(0, ca_width - 1)] += 4
                count += 1

            print(conn_matrix)

            im_ca_n = hhrbn_run(init, ca_width, ca_rule, conn_matrix)

            for j in range(0, len(init_alt)):
                im_ca_alt[j] = hhrbn_run(init_alt[j], ca_width, ca_rule, conn_matrix)


        hamming = [[], [], [], [], []]
        D_Levenshtein = [[], [], [], [], []]
        for j in range(0, len(im_ca_alt)):
            for k in range(0, len(im_ca_n)):
                hamming[j].append(sum(im_ca_alt[j][k].astype(np.uint8) ^ im_ca_n[k].astype(np.uint8)))
                D_Levenshtein[j].append(DamerauLevenshtein.distance(im_ca_alt[j][k], im_ca_n[k]))

        hamming_list_1bit.append(hamming[0])
        hamming_list_5bit.append(hamming[1])
        hamming_list_9bit.append(hamming[2])
        hamming_list_5bit_rand.append(hamming[3])
        hamming_list_9bit_rand.append(hamming[4])

        DL_list_1bit.append(D_Levenshtein[0])
        DL_list_5bit.append(D_Levenshtein[1])
        DL_list_9bit.append(D_Levenshtein[2])
        DL_list_5bit_rand.append(D_Levenshtein[3])
        DL_list_9bit_rand.append(D_Levenshtein[4])

        firstid = exp_run["id"].max() + 1 if not exp_run.empty else 0
        exp_run_row_orig = {"id": firstid, "exp_id": exp_row["id"], "run_nr": i, "type": "origin",
                            "init": ''.join(str(x[0]) for x in init)}
        exp_run = pd.concat([exp_run, pd.DataFrame.from_records(exp_run_row_orig, index=[0])], ignore_index=True)

        exp_run_row_1bit = {"id": exp_run["id"].max() + 1, "exp_id": exp_row["id"], "run_nr": i,
                            "type": "1_bit_central",
                            "init": ''.join(str(x[0]) for x in init_alt[0])}
        exp_run = pd.concat([exp_run, pd.DataFrame.from_records(exp_run_row_1bit, index=[0])], ignore_index=True)

        exp_run_row_5bit = {"id": exp_run["id"].max() + 1, "exp_id": exp_row["id"], "run_nr": i,
                            "type": "5_bit_central",
                            "init": ''.join(str(x[0]) for x in init_alt[1])}
        exp_run = pd.concat([exp_run, pd.DataFrame.from_records(exp_run_row_5bit, index=[0])], ignore_index=True)

        exp_run_row_9bit = {"id": exp_run["id"].max() + 1, "exp_id": exp_row["id"], "run_nr": i,
                            "type": "9_bit_central",
                            "init": ''.join(str(x[0]) for x in init_alt[2])}
        exp_run = pd.concat([exp_run, pd.DataFrame.from_records(exp_run_row_9bit, index=[0])], ignore_index=True)

        exp_run_row_5bit_rand = {"id": exp_run["id"].max() + 1, "exp_id": exp_row["id"], "run_nr": i,
                                 "type": "5_bit_random",
                                 "init": ''.join(str(x[0]) for x in init_alt[3])}
        exp_run = pd.concat([exp_run, pd.DataFrame.from_records(exp_run_row_5bit_rand, index=[0])], ignore_index=True)

        exp_run_row_9bit_rand = {"id": exp_run["id"].max() + 1, "exp_id": exp_row["id"], "run_nr": i,
                                 "type": "9_bit_random",
                                 "init": ''.join(str(x[0]) for x in init_alt[4])}
        exp_run = pd.concat([exp_run, pd.DataFrame.from_records(exp_run_row_9bit_rand, index=[0])], ignore_index=True)

        steps = np.arange(height_fig)
        temp_run = pd.DataFrame(
            {"step_nr": steps, "diff_hamming": hamming_list_1bit[i], "diff_Damerau_Levenshtein": DL_list_1bit[i]})
        temp_run["exp_run_id"] = exp_run_row_1bit["id"]
        run = pd.concat([run, temp_run], ignore_index=True)

        temp_run = pd.DataFrame(
            {"step_nr": steps, "diff_hamming": hamming_list_5bit[i], "diff_Damerau_Levenshtein": DL_list_5bit[i]})
        temp_run["exp_run_id"] = exp_run_row_5bit["id"]
        run = pd.concat([run, temp_run], ignore_index=True)

        temp_run = pd.DataFrame(
            {"step_nr": steps, "diff_hamming": hamming_list_9bit[i], "diff_Damerau_Levenshtein": DL_list_9bit[i]})
        temp_run["exp_run_id"] = exp_run_row_9bit["id"]
        run = pd.concat([run, temp_run], ignore_index=True)

        temp_run = pd.DataFrame({"step_nr": steps, "diff_hamming": hamming_list_5bit_rand[i],
                                 "diff_Damerau_Levenshtein": DL_list_5bit_rand[i]})
        temp_run["exp_run_id"] = exp_run_row_5bit_rand["id"]
        run = pd.concat([run, temp_run], ignore_index=True)

        temp_run = pd.DataFrame({"step_nr": steps, "diff_hamming": hamming_list_9bit_rand[i],
                                 "diff_Damerau_Levenshtein": DL_list_9bit_rand[i]})
        temp_run["exp_run_id"] = exp_run_row_9bit_rand["id"]
        run = pd.concat([run, temp_run], ignore_index=True)

    # pandas
    exp_row["end"] = pd.Timestamp.now()
    exp = pd.concat([exp, pd.DataFrame.from_records(exp_row, index=[0])], ignore_index=True)

    exp.to_csv("data/exp.csv", index=False)
    exp_run.to_csv("data/exp_run.csv", index=False)
    run.to_csv("data/run.csv", index=False)

hamming_avg = [[], [], [], [], []]
#
# import os
#
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
# # print(numpy.array(hamming_list_1bit)[:, 0])
# # print(hamming_list_1bit)
# for i in range(0, height_fig):
#     r = numpy.array(hamming_list_1bit)[:, i]
#     hamming_avg[0].append(sum(r) / len(r))
#     r = numpy.array(hamming_list_5bit)[:, i]
#     hamming_avg[1].append(sum(r) / len(r))
#     r = numpy.array(hamming_list_9bit)[:, i]
#     hamming_avg[2].append(sum(r) / len(r))
#     r = numpy.array(hamming_list_5bit_rand)[:, i]
#     hamming_avg[3].append(sum(r) / len(r))
#     r = numpy.array(hamming_list_9bit_rand)[:, i]
#     hamming_avg[4].append(sum(r) / len(r))
#
# # print(hamming_avg)
#
# im_ca = [np.array(im_ca_n, copy=True), np.array(im_ca_n, copy=True), np.array(im_ca_n, copy=True),
#          np.array(im_ca_n, copy=True), np.array(im_ca_n, copy=True)]
#
# for index in range(0, 5):
#     for i, j in np.ndindex(im_ca[index].shape):
#         if im_ca[index][i, j] != im_ca_alt[index][i, j]:
#             im_ca[index][i, j] = 2.0
#
# cmap = colors.ListedColormap(["white", "lightgray", "darkred"])
# plt.figure()
# grid = plt.GridSpec(2, 11)
# g0 = plt.subplot(grid[0, 0])
# g1 = plt.subplot(grid[0, 1])
# g2 = plt.subplot(grid[0, 2])
# g3 = plt.subplot(grid[0, 3])
# g4 = plt.subplot(grid[0, 4])
# g5 = plt.subplot(grid[0, 5])
# g6 = plt.subplot(grid[0, 6])
# g7 = plt.subplot(grid[0, 7])
# g8 = plt.subplot(grid[0, 8])
# g9 = plt.subplot(grid[0, 9])
# g10 = plt.subplot(grid[0, 10])
#
# plot = plt.subplot(grid[1, :])
#
# g0.imshow(im_ca_n, cmap="binary")
# g0.axis('off')
# g0.set_title("Original")
# g1.imshow(im_ca_alt[0], cmap="binary")
# g1.axis('off')
# g1.set_title("1 bit diff")
# g2.imshow(im_ca[0], cmap=cmap)
# g2.axis('off')
# g2.set_title("1 bit diff")
# g3.imshow(im_ca_alt[1], cmap="binary")
# g3.axis('off')
# g3.set_title("5 bit diff local")
# g4.imshow(im_ca[1], cmap=cmap)
# g4.axis('off')
# g4.set_title("5 bit diff local")
# g5.imshow(im_ca_alt[2], cmap="binary")
# g5.axis('off')
# g5.set_title("9 bit diff local")
# g6.imshow(im_ca[2], cmap=cmap)
# g6.axis('off')
# g6.set_title("9 bit diff local")
# g7.imshow(im_ca_alt[3], cmap="binary")
# g7.axis('off')
# g7.set_title("5 bit diff random")
# g8.imshow(im_ca[3], cmap=cmap)
# g8.axis('off')
# g8.set_title("5 bit diff random")
# g9.imshow(im_ca_alt[4], cmap="binary")
# g9.axis('off')
# g9.set_title("9 bit diff random")
# g10.imshow(im_ca[4], cmap=cmap)
# g10.axis('off')
# g10.set_title("9 bit diff random")
#
# plot.plot(hamming_avg[0], label="1 diff")
# plot.plot(hamming_avg[1], label="5 diff")
# plot.plot(hamming_avg[2], label="9 diff")
# plot.plot(hamming_avg[3], label="5 random diff")
# plot.plot(hamming_avg[4], label="9 random diff")
# plot.legend()
#
# plt.title(f'Rule {ca_rule[0][0]}, {height_fig} steps')
#
# plt.show()

# fig.savefig(f'diff.png', bbox_inches="tight", dpi=300)
