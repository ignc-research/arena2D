#Evaluation skript
import numpy as np 
import matplotlib.pyplot as plt
import csv
import os
import pandas as pd



path = 'evaluation/'

#make dir for plots
try:
    os.mkdir(path)
except OSError:
    print ("Creation of the directory %s failed" % path)
else:
    print ("Successfully created the directory %s " % path)


#read in evaluation data
data = pd.read_csv('evaluation.csv')
ending = np.array(data['Ending'])
episode = np.array(data['Episode'])
robot_position = np.array(data[['Robot_Position_x','Robot_Position_y']])
human_robot_distance = np.array(data[data.columns[4:]])
human_robot_distance = human_robot_distance[~np.isnan(human_robot_distance).any(axis=1)]

#read out some values
human_counter = np.count_nonzero(ending == 'human')
wall_counter = np.count_nonzero(ending == 'wall')
time_out_counter = np.count_nonzero(ending == 'time')
goal_counter = np.count_nonzero(ending == 'goal')
num_episodes = human_counter + wall_counter + time_out_counter + goal_counter

#get time_to_goal and traveled distanse per episode
time_to_goal = []
traveled_distance = []
for current_episode in range(num_episodes):
	idx_start = np.argwhere(episode == current_episode) if current_episode != 0 else 0
	idx_end = np.argwhere(episode == current_episode+1)
	if ending[idx_end] == 'goal':
		position_for_episode = robot_position[int(idx_start):int(idx_end)]
		position_for_episode = position_for_episode[~np.isnan(position_for_episode).any(axis=1)]
		time_to_goal.append(position_for_episode.shape[0])
		dist = 0
		for row_idx in range(position_for_episode.shape[0]-1):
			v1 = position_for_episode[row_idx]
			v2 = position_for_episode[row_idx+1]
			dist = dist + np.linalg.norm(v1-v2)
		traveled_distance.append(dist)
#remove all nan 
robot_position = robot_position[~np.isnan(robot_position).any(axis=1)]

#time_to_goal = np.array(time_to_goal)
#traveled_distance = np.array(traveled_distance)
#read safty distance from settings.st
with open("settings.st", "r") as f:
    settings = f.readlines()
for line in settings:
    if 'safety_distance_human' in line:
        start = line.find("=")
        end = line.find("#")
        safety_distance_human = float(line[start+1:end])


#print hit numbers
#print('for evaluation the robot did 1000 episodes')
#print('Robot hit ' + str(human_counter) + ' times a human')
#print('Robot hit ' + str(wall_counter) + ' times a wall')
#print('Robot reaches ' + str(time_out_counter) + ' times NOT the goal')

text_file = open(path + "evaluation_stats.txt", "w")
text_file.write('For the evaluation the robot did ' + str(num_episodes) + ' episodes \n')
text_file.write('The robot reached ' + str(goal_counter) + ' the goal \n')
text_file.write('The robot hit ' + str(human_counter) + ' times a human \n')
text_file.write('The robot hit ' + str(wall_counter) + ' times a wall \n')
text_file.write('The robot didn\'t reach the goal in time for ' + str(time_out_counter) + ' times \n')
text_file.close()



#hist plot of distances robot-human
mu = np.mean(human_robot_distance)
std=np.std(human_robot_distance)

plt.figure()
plt.hist(human_robot_distance.flatten(),bins=50,align = 'right',edgecolor='black', linewidth=0.8)
plt.axvline(safety_distance_human, color='r', linestyle='dashed', linewidth=1.5)
plt.annotate('safety distance = %.2f'%(safety_distance_human), xy=(0, 0.85), xycoords='axes fraction',color = 'red')
plt.annotate('$\mu = %.2f$'%(mu) + ' \n$\sigma^2 = %.2f$'%(std), xy=(0.85, 0.85), xycoords='axes fraction',bbox=dict(boxstyle="round", fc="w"))
plt.title('Hist of human distances')
plt.xlabel('distance')
plt.ylabel('counts')
plt.savefig(path+ 'human_distance_hist')
plt.clf()

#box plot auf time to reach goal 
plt.figure(1)
plt.subplot(1,2,1)
plt.title('Time for reaching the goal')
plt.ylabel('number of actions to reach the goal')
plt.boxplot(time_to_goal)

plt.subplot(1,2,2)
plt.title('distance to reach goal')
plt.ylabel('distance')
plt.boxplot(traveled_distance)
plt.savefig(path + 'goal_distance_time_box')
plt.clf()

print('evaluation done')

