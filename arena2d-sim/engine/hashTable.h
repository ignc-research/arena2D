/* author: Cornelius Marx */
#ifndef HASH_TABLE_H
#define HASH_TABLE_H

#include <string.h>
#include "list.h"

#ifdef __cplusplus
extern "C"{
#endif
//simple hashtable that controls <key,value> pairs
//thereby key is considered to be a c string value can be any kind of data

typedef unsigned int uint;//unsigned integer
typedef struct {
	void * key;
	uint key_size;
    uint val_size;
	void * value;
} sPair;

typedef struct {
	sList ** table;
	uint size; //number of buckets in table
	uint num_elements; //number of pairs in table
	uint (*hash_func)(const void *, uint, void *);//hash function; parameters: void * key, uint key_len, void * info
	void * hash_info;//additional information passed to hash_function
} sHashTable;

//init hashtable with given capacity and hash function
sHashTable* h_init(uint capacity, uint (*hash_function)(const void *, uint, void *), void * hash_info);

//delete the hashtable and all of its pairs, returns data-array
void** h_free(sHashTable* h, uint * num_elements);

//resize number of buckets 
void h_resize(sHashTable * h, uint new_capacity);

//adds a key value pair to the hashtable
void h_add(sHashTable * h, const void * key, uint key_size, void * value, uint val_size);

//adds a given pair to the hashtable
void h_addPair(sHashTable * h, sPair * p);

//remove a key value pair from hash table, @return data-pointer
void* h_remove(sHashTable * h, const void * key, uint key_size);

//sets the data pointer for the given key and returns the old data
void* h_set(sHashTable * h, const void * key, uint key_size, void * new_value, uint val_size);

//get value of given key, returns NULL if no such key was found
void* h_get(sHashTable * h, const void * key, uint key_length, uint* val_length);

//remove all pairs in hashtable, returns array of value-objects of pairs
void** h_clear(sHashTable * h, uint *num_elements);

//print table
void h_printTable(sHashTable * h, void (*print_value)(void * value,void * inf),void* info);

///private functions (do not call externally)

//hashing key
uint _h_hashKey(sHashTable * h, const void * key, uint key_length);

//getPair, returning list element and bucket-position
sListItem* _h_getPair(sHashTable * h, const void * key, uint key_length, uint * bucket_pos);

//initialize table with given size
sList** _h_initTable(uint capacity);

#ifdef __cplusplus
}
#endif

#endif
