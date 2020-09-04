#Evaluation skript
import numpy as np 
import matplotlib.pyplot as plt
import csv
import os


path = 'evaluation/'

try:
    os.mkdir(path)
except OSError:
    print ("Creation of the directory %s failed" % path)
else:
    print ("Successfully created the directory %s " % path)

with open('evaluation.csv','r') as dest_f:
    data_iter = csv.reader(dest_f,
                           delimiter = ',',
                           quotechar = '"')
    data = [data for data in data_iter]
    
human_robot_distance = []
distances = []
time_to_goal = []
traveled_distance = []
counter = 0
episode = 0
for line in data:
    if not line:
        pass
    elif line[0] == 'Episode ': 
        episode = episode + 1
        human_robot_distance.append(distances)
        distances = []
    elif line[0] == 'Action counter ':
        time_to_goal.append(float(line[1]))
    elif line[0] == 'Travelled distance ':
        traveled_distance.append(float(line[1]))
    elif line[0] == 'Goal counter ':
        goal_counter = float(line[1])
    elif line[0] == 'Human counter ':
        human_counter = float(line[1])
    elif line[0] == 'Wall counter ':
        wall_counter = float(line[1])
    elif line[0] == 'Time out counter':
        time_out_counter = float(line[1])
    else:
        distances.append([float(i) for i in line])
        
num_episodes = human_counter + wall_counter + time_out_counter + goal_counter

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

hist_arr = []
for episode in human_robot_distance:
    hist,bin_edges = np.histogram(episode,bins=20,range=[0,5])
    hist_arr.append(hist)


hist_arr = np.array(hist_arr).mean(axis=0)
plt.figure()
plt.bar(bin_edges[1:],hist_arr)
plt.title('Hist of human distances')
plt.xlabel('distance')
plt.ylabel('Number of steps where distance fall in bin')
plt.savefig(path+ 'human_distance_hist')
plt.clf()

print('evaluation done')

exit()
