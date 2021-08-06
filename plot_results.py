import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('data/results.csv', index_col=0)

rosen = df[[c for c in df.columns if 'rosen' in c]]
rosen.columns = [x.split('_')[-1] for x in rosen.columns]

rastrigin = df[[c for c in df.columns if 'rastrigin' in c]]
rastrigin.columns = [x.split('_')[-1] for x in rastrigin.columns]

ackley = df[[c for c in df.columns if 'ackley' in c]]
ackley.columns = [x.split('_')[-1] for x in ackley.columns]


data = {
    'rosen': rosen,
    'rastrigin': rastrigin,
    'ackley': ackley
}

for f, d in data.items():
    print(f'Mean Results for the {f} function')
    print(d.mean().to_string(), end='\n\n')

label_map = {
    'rs': 'random search',
    'nses': 'novelty search',
    'es': 'evolutionary strategies',
    'me': 'map elites',
    'qdes': 'quality diversity',
    'cmaes': 'cma-es'
}

for f, d in data.items():
    bin_d = (d < 0.05).cumsum()
    fig = plt.figure(figsize=(13, 7))
    labels = [label_map[x] for x in bin_d.columns]
    plt.plot(bin_d, label=labels)
    plt.xlabel('# Iterations')
    plt.ylabel('cumulative # of solutions with distance < 5% of the target')
    plt.legend(fontsize='large')
    plt.title(f'Results for the {f} function')
    plt.savefig(f'imgs/{f}_line_plot', pad_inches=0)


for f, d in data.items():
    fig = plt.figure(figsize=(13, 7))
    labels = [label_map[x] for x in bin_d.columns]
    _ = plt.boxplot(d, labels=labels)
    plt.xlabel('Algorithms')
    plt.ylabel('Mean of the solutions')
    plt.title(f'Results for the {f} function')
    plt.savefig(f'imgs/{f}_box_plot', pad_inches=0)
