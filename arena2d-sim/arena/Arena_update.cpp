/* Author: Cornelius Marx */
#include "Arena.hpp"

void Arena::update()
{
	// update key control
	int action = -1;
	if(!_trainingMode){
		if(_playSimulation){
			action = Robot::STOP;
		}
		if(_keysPressed[UP]){
			action = Robot::FORWARD;
			if(_keysPressed[LEFT]){
				action = Robot::FORWARD_LEFT;
			}
			else if(_keysPressed[RIGHT]){
				action = Robot::FORWARD_RIGHT;
			}
		}else if(_keysPressed[DOWN]){
			action = Robot::BACKWARD;
		}
		else{
			if(_keysPressed[LEFT]){
				action = Robot::FORWARD_STRONG_LEFT;
			}
			else if(_keysPressed[RIGHT]){
				action = Robot::FORWARD_STRONG_RIGHT;
			}
		}
		if(action != -1){
			for(int i = 0; i < _numEnvs; i++){
				Robot::getActionTwist((Robot::Action)action, _actions[i]);
			}
		}
	}

	int episodes_before = _episodeCount;
	if(_trainingMode && _pyAgentUsed){
		_agentMeasure.startTime();
		if(_agentFuncs[PYAGENT_FUNC_PRE_STEP] != NULL){
			// creating list
			PyObject * args = PyTuple_New(1);
			PyTuple_SetItem(args, 0, packAllPyObservation());
			PyObject * result = PyObject_CallObject(_agentFuncs[PYAGENT_FUNC_PRE_STEP], args);
			Py_DECREF(args);
			if(result == NULL){
				PyErr_Print();
				ERROR_F("Call to function '%s' in Python agent failed", PYAGENT_FUNC_NAMES[PYAGENT_FUNC_PRE_STEP]);
				cmdStopTraining(ConsoleParameters(0, NULL));
			}
			else{
				bool return_error = false;
				if(PyList_Check(result)){// get multiple actions
					if(PyList_Size(result) != _numEnvs){// check number of elements
						ERROR_F("Call to function '%s' in Python agent failed: Size of returned list (%d) does not match number of environments (%d)!",
								PYAGENT_FUNC_NAMES[PYAGENT_FUNC_PRE_STEP], (int)PyList_Size(result), _numEnvs);
						cmdStopTraining(ConsoleParameters(0, NULL));
					}
					else{
						action = 0;
						for(int i = 0; i < _numEnvs; i++){
							PyObject * item = PyList_GetItem(result, i);
							if(PyTuple_Check(item)){
								_actions[i].linear = PyFloat_AsDouble(PyTuple_GetItem(item, 0));
								_actions[i].angular = PyFloat_AsDouble(PyTuple_GetItem(item, 1));
							}
							else if(PyLong_Check(item)){
								Robot::getActionTwist((Robot::Action)PyLong_AsLong(item), _actions[i]);
							}
							else{
								return_error = true;
								break;
							}
						}
					}
				}
				else{
					Twist t;
					if(PyLong_Check(result)){// get single action
						action = 0;
						Robot::getActionTwist((Robot::Action)PyLong_AsLong(result), t);
					}
					else if(PyTuple_Check(result)){// get single twist (tuple)
						action = 0;
						t.linear = PyFloat_AsDouble(PyTuple_GetItem(result, 0));
						t.angular = PyFloat_AsDouble(PyTuple_GetItem(result, 1));
					}
					else{
						return_error = true;
					}

					if(action >= 0){
						for(int i = 0; i < _numEnvs; i++){
							_actions[i] = t;
						}
					}
				}
				Py_DECREF(result);
				if(action < 0){
					ERROR_F("Call to function '%s' in Python agent failed: Expected List or Long for return value!", PYAGENT_FUNC_NAMES[PYAGENT_FUNC_PRE_STEP]);
					cmdStopTraining(ConsoleParameters(0, NULL));
				}

				if(return_error){
					ERROR_F("Unexpected return type from function '%s' in Python agent: Expected Long or Tuple!", PYAGENT_FUNC_NAMES[PYAGENT_FUNC_PRE_STEP]);
					cmdStopTraining(ConsoleParameters(0, NULL));
				}
			}
		}
		_agentMeasure.endTime();
	}


	_physicsTimer.checkLastUpdate();

	if(action >= 0){
		// check if episodes are already done
		for(int i = 0; i < _numEnvs; i++){
			Environment::EpisodeState s = _envs[i].getEpisodeState();
			if(s != Environment::RUNNING){
				_dones[i] = true;
			}
			else{
				_dones[i] = false;
			}
		}
		// threaded simulation step
		_simulationMeasure.startTime();
		for(int i = 0; i < _numThreads; i++){
			_threads[i].step();
		}

		for(int i = 0; i < _numThreads; i++){
			_threads[i].wait_finish();
		}
		_simulationMeasure.endTime();

		// check whether episodes has just ended
		bool episode_over = false;
		for(int i = 0; i < _numEnvs; i++){
			Environment::EpisodeState s = _envs[i].getEpisodeState();
			if(s != Environment::RUNNING && !_dones[i]){// episode is now over and has not been over before
				_dones[i] = true;
				// adding success value to success buffer
				_meanSuccess.push((s == Environment::POSITIVE_END) ? 1 : 0);
				_meanSuccess.calculateMean();

				// adding reward to total reward buffer
				_meanReward.push(_envs[i].getTotalReward());
				_meanReward.calculateMean();

				_episodeCount++;

				// show results
				printEpisodeResults(_envs[i].getTotalReward());

				episode_over = true;
			}
		}
		if(episode_over){
			if(_SETTINGS->video.enabled){
				refreshEpisodeCounter();
				refreshRewardCounter();
			}
		}

		// measuring FPS
		_physicsTimer.update(false);

		// call agents post step function
		bool reset_envs = true;
		if(_trainingMode && _pyAgentUsed){
			_agentPostMeasure.startTime();
			if(_agentFuncs[PYAGENT_FUNC_POST_STEP] != NULL){
				PyObject * args = PyTuple_New(5);
				PyTuple_SetItem(args, 0, packAllPyObservation());						//observation
				PyTuple_SetItem(args, 1, packAllPyRewards());							//reward
				PyTuple_SetItem(args, 2, packAllPyDones());								//done
				PyTuple_SetItem(args, 3, PyFloat_FromDouble(_meanReward.getMean()));	//mean reward
				PyTuple_SetItem(args, 4, PyFloat_FromDouble(_meanSuccess.getMean()));	//mean success
				PyObject * result = PyObject_CallObject(_agentFuncs[PYAGENT_FUNC_POST_STEP], args);
				Py_DECREF(args);
				if(result == NULL){
					PyErr_Print();
					ERROR_F("Call to function '%s' in Python agent failed", PYAGENT_FUNC_NAMES[PYAGENT_FUNC_POST_STEP]);
					cmdStopTraining(ConsoleParameters(0, NULL));
				}
				else{
					int done = PyLong_AsLong(result); 
					Py_DECREF(result);
					if(done > 0){// stop training
						cmdStopTraining(ConsoleParameters(0, NULL));
						INFO("Training done!");
					}
					else if(done < 0){ // do not reset environments
						reset_envs = false;
					}
				}
			}
			_agentPostMeasure.endTime();
		}

		// reset environments if allowed by agent
		if(reset_envs){
			for(int i = 0; i < _numEnvs; i++){
				Environment::EpisodeState s = _envs[i].getEpisodeState();
				if(s != Environment::RUNNING){
					_levelResetMeasure.startTime();
					_envs[i].reset(false);
					_levelResetMeasure.endTime();
				}
			}
			if(_SETTINGS->video.enabled && !_videoDisabled){
				refreshLevelResetTime();
			}
		}

		// write to csv file if episode count changed
		if(!_noTrainingRecord && _trainingMode && _pyAgentUsed && _episodeCount != episodes_before)
		{
			// write header first?
			const bool write_header = (_csvWriter.getNumLines() == 0);
			std::vector<const char*> names(3);
			// default metrics
			if(write_header)
			{
				names[0] = "Episodes";
				names[1] = "Success";
				names[2] = "Mean Reward";
			}
			std::vector<float> values(3);
			values[0] = (float)_episodeCount;
			values[1] = _meanSuccess.getMean();
			values[2] = _meanReward.getMean();
			// call get_stats python function
			if(_agentFuncs[PYAGENT_FUNC_GET_STATS] != NULL){
				const char * func_name = PYAGENT_FUNC_NAMES[PYAGENT_FUNC_GET_STATS];
				PyObject * result = PyObject_CallObject(_agentFuncs[PYAGENT_FUNC_GET_STATS], NULL);
				if(result == NULL){
					ERROR_F("Call to function '%s' in Python agent failed!", func_name);
				}
				else{
					// check if list
					if(!PyList_Check(result)){
						ERROR_F("Expected a list for return value of function '%s'!", func_name);
					} else{
						int num_stats = PyList_Size(result);
						for(int i = 0; i < num_stats; i++)
						{
							PyObject * item = PyList_GetItem(result, i);
							if(!PyTuple_Check(item)){
								ERROR_F("Expected tuple for item %d in returned list from function '%s'!", i, func_name);
								break;
							}
							PyObject * value = PyTuple_GetItem(item, 1);
							if(write_header){
								PyObject * name = PyTuple_GetItem(item, 0);
								if(!PyUnicode_Check(name)){
									ERROR_F("Expected string at first position in tuple %d in returned list from function '%s'!", i, func_name);
									break;
								}
								names.push_back(PyUnicode_AsUTF8(name));
							}
							float fvalue = 0;
							if(PyFloat_Check(value)) {
								fvalue = PyFloat_AsDouble(value);
							}else if(PyLong_Check(value)){
								fvalue = (float)PyLong_AsLong(value);
							}else{
								ERROR_F("Expected float or int at second position in tuple %d in returned list from function '%s'!", i, func_name);
								break;
							}
							values.push_back(fvalue);

						}
					}
					Py_DECREF(result);
				}
			}
			if(write_header){// write header first
				_csvWriter.writeHeader(names);
			}
			if(values.size() != _csvWriter.getNumCols()){
				ERROR_F("Number of metrics changed to %d (before: %d)!", values.size(), _csvWriter.getNumCols());
			}
			_csvWriter.write(values);
			_csvWriter.flush();
		}
	}
}
