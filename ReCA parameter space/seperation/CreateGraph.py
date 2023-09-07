import pandas as pd
import matplotlib.pyplot as plt

drop_zero = True
# 9, 34, 57, 73, 38
rule = 57
width = 100
height = 500

exp = pd.read_csv(f'data/exp.csv')
exp_run = pd.read_csv(f'data/exp_run.csv')
run = pd.read_csv(f'data/run.csv')

graph_data = exp[(exp.rule == rule) &
                 (exp.width == width) &
                 (exp.height == height) &
                 (exp.substrate == "ECA")
                 ]
print(graph_data.head(50))

meta = pd.merge(graph_data, exp_run, left_on="id", right_on="exp_id")
print(meta.head(50))

mergius_maximus = pd.merge(meta, run, left_on="id_y", right_on="exp_run_id")
print(mergius_maximus.head(50))

if drop_zero:
    # print(mergius_maximus.exp_run_id.nunique())
    to_drop = mergius_maximus[(mergius_maximus.step_nr == height - 1) & (mergius_maximus.diff_hamming == 0)]
    # print(to_drop.head(50))
    for id in to_drop.exp_run_id:
        # print(id)
        mergius_maximus = mergius_maximus[mergius_maximus.exp_run_id != id]

print(mergius_maximus.groupby("type").exp_run_id.nunique())


plottable = mergius_maximus[["type", "step_nr", "diff_hamming", "diff_Damerau_Levenshtein"]].groupby(
    ["type", "step_nr"]).agg(["mean", "std"]).reset_index()
# https://towardsdatascience.com/how-to-flatten-multiindex-columns-and-rows-in-pandas-f5406c50e569
plottable.columns = ['_'.join(col) for col in plottable.columns.values]

# plottable = plottable[plottable.type_ == "1_bit_central"]

fig, axs = plt.subplots(2)
axs[0].set_title('Hamming')
axs[1].set_title('Damerau Levenshtain')
for key, grp in plottable.groupby("type_"):
    grp.plot(ax=axs[0], x="step_nr_", y='diff_hamming_mean', label=key)
         # , yerr="diff_hamming_std"
         # )
    grp.plot(ax=axs[1], x="step_nr_", y='diff_Damerau_Levenshtein_mean', label=key)
    # , yerr="diff_Damerau_Levenshtein_std"
    # )


plt.legend()
plt.show()
