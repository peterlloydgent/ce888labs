import matplotlib
matplotlib.use('Agg')
import pandas as pd
import seaborn as sns
import numpy as np

def getSampleSet(sample, sample_size):
    return_array = []
    for i in range(0,sample_size):
        rand = np.random.randint(0, sample_size)
        return_array.append(sample[rand])
    return_array.sort()

    return return_array

def bootstrap(sample, sample_size, iterations):
    samples = []
    mean_averages = []
    i = 0
    while i < iterations:
        samples.append(getSampleSet(sample,sample_size))
        print(samples[i])
        mean_averages.append(np.mean(samples[i]))
        print("mean for iteration " + str(i) + " : " + str(mean_averages[i]))
        i+=1
    # remove 2.5% percentile from either end and return upper or lower average.
    mean_averages.sort()
    data_mean = np.percentile(mean_averages,95)
    lower = mean_averages[0]
    upper = mean_averages[1]
	return data_mean, lower, upper


if __name__ == "__main__":
	df = pd.read_csv('./salaries.csv')

	data = df.values.T[1]
	boots = []
	for i in range(100, 100000, 1000):
		boot = bootstrap(data, data.shape[0], i)
		boots.append([i, boot[0], "mean"])
		boots.append([i, boot[1], "lower"])
		boots.append([i, boot[2], "upper"])

	df_boot = pd.DataFrame(boots, columns=['Boostrap Iterations', 'Mean', "Value"])
	sns_plot = sns.lmplot(df_boot.columns[0], df_boot.columns[1], data=df_boot, fit_reg=False, hue="Value")

	sns_plot.axes[0, 0].set_ylim(0,)
	sns_plot.axes[0, 0].set_xlim(0, 100000)

	sns_plot.savefig("bootstrap_confidence.png", bbox_inches='tight')
	sns_plot.savefig("bootstrap_confidence.pdf", bbox_inches='tight')


	#print ("Mean: %f")%(np.mean(data))
	#print ("Var: %f")%(np.var(data))
	


	