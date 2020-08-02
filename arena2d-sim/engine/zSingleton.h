//created on: 16th Jan 2017
//Author: Cornelius Marx

#ifndef SINGLETON_H
#define SINGLETON_H

template <class T>
class zTSingleton
{
protected:

	//members
	static T *_singleton;  

public:
	//destructor
	virtual ~zTSingleton ()
	{
	}

	//get instance
	inline static T* get ()
	{
		// instance already existing?
		if (_singleton == 0)
			_singleton = new T;   // create new instance

		return (_singleton);
	}

	//delete singleton instance
	static void del ()
	{
		if (_singleton)
		{
			delete (_singleton);
			_singleton = 0;
		}
	}
};

//create static variable
template <class T>
T* zTSingleton<T>::_singleton = 0;

#endif
