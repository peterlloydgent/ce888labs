from pandas import read_csv
import matplotlib.pyplot as plt
import seaborn as sns
vehicle_file = open("vehicles.csv",'r')

csv = read_csv('vehicles.csv')
current_fleet = csv["Current fleet"]
new_fleet = csv["New Fleet"]

sns_plot2 = sns.distplot(current_fleet,kde=False, rug=True).get_figure()
axes = plt.gca()
axes.set_xlabel("Miles per gallon")
axes.set_ylabel("Frequency")
sns_plot2.savefig("current_fleet_histogram.png",bbox_inches = "tight")

plt.clf()

current_fleet_index = read_csv('current_fleet.csv')

sns_plot = sns.lmplot(current_fleet_index.columns[0],current_fleet_index.columns[1],data=current_fleet_index,fit_reg=False)
sns_plot.axes[0,0].set_ylim(0,)
sns_plot.axes[0,0].set_xlim(0,)
sns_plot.savefig("current_fleet_scatterplot.png",bbox_inches = "tight")

# new fleet

new_fleet_index = read_csv('new_fleet.csv')
#scatter plot
sns_plot = sns.lmplot(new_fleet_index.columns[0],new_fleet_index.columns[1],data=new_fleet_index,fit_reg=False)
sns_plot.axes[0,0].set_ylim(0,)
sns_plot.axes[0,0].set_xlim(0,)
sns_plot.savefig("new_fleet_scatterplot.png",bbox_inches = "tight")

plt.clf()
sns_plot3 = sns.distplot(new_fleet[0:79],kde=False, rug=True).get_figure()
axes = plt.gca()
axes.set_xlabel("Miles per gallon")
axes.set_ylabel("Frequency")
sns_plot3.savefig("new_fleet_histogram.png",bbox_inches = "tight")
