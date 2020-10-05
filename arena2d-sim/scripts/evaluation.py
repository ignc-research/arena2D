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
robot_direction = np.array(data[['Robot_Direction_x','Robot_Direction_y']])
human_robot_distance = np.array(data[data.columns[8:]])
human_robot_distance = human_robot_distance[~np.isnan(human_robot_distance).any(axis=1)]
goal_distance = np.array(data['Goal_Distance'])
goal_distance = goal_distance[~np.isnan(goal_distance)] # remove nan
goal_angle = np.array(data['Goal_Angle'])
goal_angle = goal_angle[~np.isnan(goal_angle)] #remove nan
#read out some values
human_counter = np.count_nonzero(ending == 'human')
wall_counter = np.count_nonzero(ending == 'wall')
time_out_counter = np.count_nonzero(ending == 'time')
goal_counter = np.count_nonzero(ending == 'goal')
num_episodes = human_counter + wall_counter + time_out_counter + goal_counter

#check if there were humans in level
human_exist = human_robot_distance.size != 0

#read safty distance and time-out time from settings.st
with open("settings.st", "r") as f:
	settings = f.readlines()
base = False
for line in settings:
	if 'safety_distance_human' in line:
		start = line.find("=")
		end = line.find("#")
		safety_distance_human = float(line[start+1:end])
	if 'max_time = ' in line:
		start = line.find("=")
		end = line.find("#")
		max_time = float(line[start+1:end])
	if 'time_step = ' in line:
		start = line.find("=")
		end = line.find("#")
		time_step = float(line[start+1:end])

		
max_step_time_out = max_time/time_step

#get time_to_goal and traveled distanse per episode
time_to_goal = []
traveled_distance = []
frac_direct_traveled_dist = []
temp=0
for current_episode in range(num_episodes):
	idx_start = np.argwhere(episode == current_episode) if current_episode != 0 else 0
	idx_end = np.argwhere(episode == current_episode+1)
	if ending[idx_end] == 'goal':
		#correct robot position index mistake if episode_end != time_out
		if ending[idx_start] != 'time':
			idx_start = idx_start + 2 #if last episode was time then correction is NOT nessesary 
		idx_end = idx_end + 2 #must always be corrected if ending[idx_end] != time
			
		position_for_episode = robot_position[int(idx_start):int(idx_end)]
		position_for_episode = position_for_episode[~np.isnan(position_for_episode).any(axis=1)]

		time_to_goal.append(position_for_episode.shape[0] - 1)
		vector_array = position_for_episode[:-1] - position_for_episode[1:]
		distance = np.sum(np.sum(np.abs(vector_array)**2,axis=-1)**(1./2))
		traveled_distance.append(distance)
		#first episode goal info missing
		if current_episode != 0:
			frac_direct_traveled_dist.append(distance - goal_distance[current_episode-1])



#remove all nan 
robot_position = robot_position[~np.isnan(robot_position).any(axis=1)]
robot_direction = robot_direction[~np.isnan(robot_direction).any(axis=1)]
time_to_goal = np.array(time_to_goal)



#hist plot of distances robot-human
if human_exist:
	mu = np.mean(human_robot_distance)
	std=np.std(human_robot_distance)

	plt.figure(1)
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
plt.figure(2)
#plt.subplot(2,2,1)
#plt.title('Time for reaching the goal')
#plt.ylabel('number of actions to reach the goal')
#plt.boxplot(time_to_goal)
#plt.subplot(2,1,1)
plt.title(r'$\frac {number steps}{max number steps}$')
plt.ylabel('relative time')
plt.boxplot(time_to_goal/max_step_time_out)
plt.savefig(path + 'goal__time_box')
plt.clf()

plt.figure(3)
plt.subplot(1,2,1)
plt.title('distance to reach goal')
plt.ylabel('distance')
plt.boxplot(traveled_distance)
plt.subplot(1,2,2)
plt.title(r'$real\_distance - direct\_distance$')
plt.ylabel('relative distance')
plt.boxplot(frac_direct_traveled_dist)
plt.savefig(path + 'goal_distance_box')
plt.clf()

#calculate and plot ending counters per bins and in total and write text file
bin_size = 100
ending = ending[~pd.isnull(data['Ending'])] #remove all nan
ending = ending[ending != 'reset'] # remove all reset
ending = np.reshape(ending,(-1,bin_size))# reshape in bins of 100

wall_counter_bin = []
human_counter_bin = []
time_out_counter_bin = []
goal_counter_bin = []

for row in ending:
	wall_counter_bin.append(np.count_nonzero(row == 'wall')/bin_size*100)
	human_counter_bin.append(np.count_nonzero(row == 'human')/bin_size*100)
	time_out_counter_bin.append(np.count_nonzero(row == 'time')/bin_size*100)
	goal_counter_bin.append(np.count_nonzero(row == 'goal')/bin_size*100)

#write text file
text_file = open(path + "evaluation_stats.txt", "w")
text_file.write('For the evaluation the robot did ' + str(num_episodes) + ' episodes \n')
text_file.write('The robot reached ' + str(goal_counter) + ' the goal \n')
if human_exist:
	text_file.write('The robot hit ' + str(human_counter) + ' times a human \n')
else:
	text_file.write('no human in level \n')
text_file.write('The robot hit ' + str(wall_counter) + ' times a wall \n')
text_file.write('The robot didn\'t reach the goal in time for ' + str(time_out_counter) + ' times \n')
text_file.write('---------------------------------------------------------------- \n')
text_file.write('Success rate: ' + str(round(goal_counter/num_episodes*100,1)) + '%')
text_file.write(', Variance: ' + str(round(np.std(goal_counter_bin),1)) + '%\n')
text_file.write('Human hit rate: ' + str(round(human_counter/num_episodes*100,1)) + '%')
text_file.write(', Variance: ' + str(round(np.std(human_counter_bin),1)) + '%\n')
text_file.write('Wall hit rate: ' + str(round(wall_counter/num_episodes*100,1)) + '%')
text_file.write(', Variance: ' + str(round(np.std(wall_counter_bin),1)) + '%\n')
text_file.write('Timeout rate: ' + str(round(time_out_counter/num_episodes*100,1)) + '%')
text_file.write(', Variance: ' + str(round(np.std(time_out_counter_bin),1)) + '%\n')
text_file.close()
#plot
plt.figure(4)
plt.title('episode endings over bins with size ' + str(bin_size))
plt.ylabel('number of counted endings')
plt.xlabel('bin')
plt.plot(wall_counter_bin,label='wall')
plt.plot(human_counter_bin,label='human')
plt.plot(time_out_counter_bin,label='time out')
plt.plot(goal_counter_bin,label='goal')
plt.legend()
plt.savefig(path + 'endings')
plt.clf()


print('evaluation done')
