#!/bin/bash
import numpy as np
import matplotlib.pyplot as plt
import os

folder_name = "plots/"
data_path = "data.csv"


if __name__ == "__main__":
	# create folder
	try:
		os.mkdir(folder_name);
	except OSError as error:
		pass

	# create plots
	data_names = np.genfromtxt(data_path, dtype='str', delimiter=',', max_rows=1)
	data = np.loadtxt(data_path, delimiter=",", skiprows=1)
	episodes = data[:,0]
	for i in range(1, len(data[0])):
		title = data_names[i]
		plt_path = folder_name + title.lower().replace(" ", "_") + ".pdf"
		print("Creating plot '%s': %s ..."%(title, plt_path));
		plt.title(title)
		plt.xlabel("Episodes")
		plt.plot(episodes, data[:,i], linewidth=0.8)
		plt.savefig(plt_path)
		plt.clf()

	print("Done!");
