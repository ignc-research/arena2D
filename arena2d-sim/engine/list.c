/* author: Cornelius Marx */
#include "list.h"

//init
sList * list_init()
{
	sList * l = (sList*) malloc(sizeof(sList));
	l->size = 0;
	l->first = NULL;
	l->last = NULL;
	return l;
}

//free
void list_free(sList * l)
{
	//removing all items
	list_clearItems(l);

	//free list
	free(l);
}

//freeAll
void list_freeAll(sList * l)
{
	//removing all items
	list_deleteItems(l);

	//free list
	free(l);
}

//clearItems
void list_clearItems(sList * l)
{
	sListItem * i = l->first;
	while(i != NULL)
	{
		sListItem * tmp = i->next;
		free(i);
		i = tmp;
	}
	l->first = NULL;
	l->last = NULL;
	l->size = 0;
}

void list_deleteItems(sList * l)
{
	sListItem * i = l->first;
	while(i != NULL)
	{
		sListItem * tmp = i->next;
		if(i->data != NULL)
			free(i->data);
		free(i);
		i = tmp;
	}
	l->first = NULL;
	l->last = NULL;
	l->size = 0;
}

//createItem
sListItem* list_createItem(void * data)
{
	sListItem * i = (sListItem*) malloc(sizeof(sListItem));
	i->data = data;
	i->next = NULL;
	i->prev = NULL;
	return i;
}

//pushFront
void list_pushFront(sList * l, void * data)
{
	//creating new element
	sListItem * i = list_createItem(data);
	
	//linking item
	sListItem * tmp_first = l->first;
	l->first = i;
	if(l->last == NULL)
	{
		l->last = i;
	}
	else//i is not the first element
	{
		i->next = tmp_first;
		tmp_first->prev = i;
	}
	l->size++;
}

//pushBack
void list_pushBack(sList * l, void * data)
{
	//creating new element
	sListItem * i = list_createItem(data);
	
	//linking item
	sListItem * tmp_last = l->last;
	l->last = i;
	if(l->first == NULL)
	{
		l->first = i;
	}
	else//i is not the first element
	{
		i->prev = tmp_last;
		tmp_last->next = i;
	}
	l->size++;
}

//insertAt
void list_insertAt(sList * l, sListItem * i, void * data)
{
	if(i == NULL)
	{
		list_pushFront(l, data);
		return;
	}
	//creating new element
	sListItem * i_new = list_createItem(data);

	//linking item
	sListItem * tmp_iprev = i->prev;
	i->prev = i_new;
	i_new->next = i;
	if(tmp_iprev == NULL)//i was the first element
	{
		l->first = i_new;
	}
	else
	{
		tmp_iprev->next = i;
	}
	l->size++;
}

//popFront
void* list_popFront(sList * l)
{
	sListItem * ifirst= l->first;
	if(ifirst == NULL)//no items to delete
		return NULL;
	
	void * data = ifirst->data;
	sListItem * next = ifirst->next;
	if(next == NULL)
	{
		l->last = NULL;	
		l->first = NULL;
	}
	else
	{
		next->prev = NULL;
		l->first = next;
	}

	free(ifirst);
	l->size--;
	return data;
}

//popBack
void * list_popBack(sList * l)
{
	sListItem * ilast= l->last;
	if(ilast == NULL)//no items to delete
		return NULL;
	
	void * data = ilast->data;
	sListItem * prev = ilast->prev;
	if(prev == NULL)
	{
		l->last = NULL;	
		l->first = NULL;
	}
	else
	{
		prev->next = NULL;
		l->last = prev;
	}

	free(ilast);
	l->size--;
	return data;
}

//removeItem
void * list_removeItem(sList * l, sListItem * i)
{
	void * data = i->data;
	
	sListItem * iprev = i->prev;
	sListItem * inext = i->next;
	if(iprev == NULL)
	{
		l->first = inext;
	}
	else
	{
		iprev->next = inext;
	}

	if(inext == NULL)
	{
		l->last = iprev;
	}
	else
	{
		inext->prev = iprev;
	}

	free(i);
	l->size--;
	return data;
}
