/* author: Cornelius Marx */
#ifndef LIST_H
#define LIST_H

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

//implements a double linked list for general purpose
#ifdef __cplusplus
extern "C"{
#endif

//sListItem
typedef struct sListItem sListItem;
struct sListItem{
	void * data;
	sListItem * next;
	sListItem * prev;
};

//sList
typedef struct{
	int size;
	sListItem * first;
	sListItem * last;	
} sList;


//creating new list
sList* list_init();

//freeing list (without removing items themselves)
void list_free(sList * l);

//freeing list and item-data (by calling deleteItems())
void list_freeAll(sList * l);

//removing all elements, all data should be removed before calling this function
//call deleteItems if you want to delete the data stored in the list items by free(it->data)
void list_clearItems(sList * l);

//delete data stored in items and remove items from list
void list_deleteItems(sList * l);

//inserting element at first
void list_pushFront(sList * l, void * data);

//inserting element at last
void list_pushBack(sList * l, void * data);

//inserting element before given element
void list_insertAt(sList * l, sListItem * i, void * data);

//removing first element, returns the data, that element contains
void* list_popFront(sList * l);

//removing last element, returns the data, that element contains
void* list_popBack(sList * l);

//removing given element, returns the data, that element contains
void* list_removeItem(sList * l, sListItem * i);

//creating a new element
sListItem* list_createItem(void * data);

#ifdef __cplusplus
}
#endif

#endif
