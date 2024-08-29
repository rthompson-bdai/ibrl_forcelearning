
import plotly
import plotly.graph_objects as go

from os import listdir
from os.path import isfile, join
import pickle


def graph_success_from_folder(folder1, folder2):
    fig = go.Figure()
    subdirs = [join(folder1, f) for f in listdir(folder1)]
    subdir_names = [f for f in listdir(folder1)]
    success_rates = []
    for subdir in subdirs: 
        successes = []
        if len(listdir(subdir)) == 0:
            continue
        files = [f for f in listdir(subdir) if (isfile(join(subdir, f)) and f[-3:] == 'pkl')]
        for f in files:
            reward = pickle.load(open(join(subdir, f), 'rb'))
            successes.append(reward[-1])
        success_rate = sum(successes)/len(successes)
        success_rates.append(success_rate)

    fig.add_trace(go.Bar(x=subdir_names, y=success_rates, name="Train Factors"))

    subdirs = [join(folder2, f) for f in listdir(folder2)]
    subdir_names = [f for f in listdir(folder2)]
    success_rates = []
    for subdir in subdirs: 
        successes = []
        if len(listdir(subdir)) == 0:
            continue
        files = [f for f in listdir(subdir) if (isfile(join(subdir, f)) and f[-3:] == 'pkl')]
        for f in files:
            reward = pickle.load(open(join(subdir, f), 'rb'))
            successes.append(reward[-1])
        success_rate = sum(successes)/len(successes)
        success_rates.append(success_rate)

    fig.add_trace(go.Bar(x=subdir_names, y=success_rates, name="Test Factors"))
    fig.update_layout(barmode='group')
    fig.write_image("bc_successses.png")


if __name__ == "__main__":
    graph_success_from_folder('bc_eval_train/metaworld', 'bc_eval_test/metaworld')