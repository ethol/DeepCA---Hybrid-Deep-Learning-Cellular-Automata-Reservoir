import pandas as pd

exp = pd.DataFrame({}, columns=["id", "width", "height", "start", "end", "rule", "substrate", 'total_runs'])
exp = exp.astype(
    {'id': 'int',
     'width': 'int',
     'height': 'int',
     'start': 'int',
     'end': 'int',
     'rule': 'int',
     'substrate': 'str',
     'total_runs': 'int'
     }
)

exp_run = pd.DataFrame({}, columns=["id", "exp_id", "run_nr", "type", "init"])
exp_run = exp_run.astype(
    {'id': 'int',
     'exp_id': 'int',
     'run_nr': 'int',
     'type': 'str',
     'init': 'str'}
)

run = pd.DataFrame({}, columns=["exp_run_id", "step_nr", "diff_hamming", "diff_Damerau_Levenshtein"])
run = run.astype(
    {'exp_run_id': 'int',
     'step_nr': 'int',
     'diff_hamming': 'int',
     'diff_Damerau_Levenshtein': 'int'}
)
# exp.to_csv("data/exp.csv", index=False)
# exp_run.to_csv("data/exp_run.csv", index=False)
# run.to_csv("data/run.csv", index=False)
