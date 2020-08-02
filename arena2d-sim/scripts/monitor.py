#!/usr/bin/python
import _thread
import signal
import re
import sys
import os
import shutil
import time
from tensorboard.main import run_main
from torch.utils.tensorboard import SummaryWriter

run_folder = "./monitor_runs/"
writer_folder = "arena2d_monitor"
csv_file = "data.csv"
file_polling_delay = 10 # wait n seconds between file polling
csv_delimiter = ','

def stats_writer():
	writer = SummaryWriter(log_dir=run_folder+writer_folder)
	bytes_read = 0
	num_lines = 0
	col_names = []
	last_unfinished_line = "" 
	# polling file for changes
	while True:
		with open(csv_file, 'r') as f:
			f.seek(bytes_read)
			r = f.read()
			if len(r) > 0:
				bytes_read += len(r) # a line end is always added by function read() (for some reason)
				complete_lines = (last_unfinished_line + r).splitlines()
				if r[-1] != "\n":
					last_unfinished_line = complete_lines[-1]
					complete_lines = complete_lines[:-1]
				if len(complete_lines) > 0:
					# get header
					if num_lines == 0:
						col_names = complete_lines[0].split(csv_delimiter)
						complete_lines = complete_lines[1:]

					for l in complete_lines:
						values = l.split(csv_delimiter)
						episode = int(float(values[0]))
						for index in range(1, len(col_names)):
							writer.add_scalar(col_names[index], float(values[index]), episode);

					num_lines += len(complete_lines)

		time.sleep(file_polling_delay)	

def signal_handler(sig, frame):
	# remove run folder
	shutil.rmtree(run_folder)
	sys.exit(0)

if __name__ == '__main__':
	# remove old run folder
	shutil.rmtree(run_folder, ignore_errors=True)

	# create directory
	try:
		os.mkdir(run_folder);
	except OSError as error:
		pass
	
	# create a writer thread
	_thread.start_new_thread(stats_writer, ())

	# set sigint handler
	signal.signal(signal.SIGINT, signal_handler)
	
	# open tensorboard main
	sys.argv[0] = re.sub(r'(-script\.pyw|\.exe)?$', '', sys.argv[0])
	sys.argv.append("--logdir")
	sys.argv.append(run_folder)
	sys.exit(run_main())

