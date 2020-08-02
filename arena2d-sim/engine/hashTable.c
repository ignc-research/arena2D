/* author: Cornelius Marx */
#include "hashTable.h"

//init
sHashTable* h_init(uint capacity, uint (*hash_function)(const void *, uint, void *), void * hash_info)
{
	sHashTable * h = (sHashTable*) malloc(sizeof(sHashTable));
	h->table = _h_initTable(capacity); 
	h->size = capacity;
	h->hash_func = hash_function;
	h->num_elements = 0;
	h->hash_info = hash_info;

	return h;
}

//_initTable
sList ** _h_initTable(uint capacity)
{
	sList ** l = (sList**) malloc(sizeof(sList*) * capacity);
	for(uint i = 0; i < capacity; i++)
	{
		l[i] = NULL;
	}

	return l;
}

//free
void ** h_free(sHashTable * h, uint * num_elements)
{
	void ** v = h_clear(h, num_elements);
	free(h->table);
	free(h);
	return v;
}

//resize
void h_resize(sHashTable * h, uint new_capacity)
{
	sList ** l = h->table;
	uint size = h->size;
	uint num_e = h->num_elements;
	//backup pairs in list
	sPair ** pairs = NULL;
	if(num_e > 0)
	{
		pairs = (sPair**) malloc(sizeof(sPair*) * num_e);
		uint pair_count = 0;
		for(uint i = 0; i < size; i++)
		{
			sList * li = l[i];
			if(li != NULL)
			{
				for(sListItem * e = li->first; e != NULL; e = (sListItem*)e->next)
				{	
					pairs[pair_count] = (sPair*) e->data; 	
					pair_count++;
				}
				list_free(li);
			}
		}
	}
	
	free(l);
	h->table = _h_initTable(new_capacity);
	h->size = new_capacity;	
	if(num_e > 0)
	{
		//insert elements in new hash table
		for(uint i = 0; i < num_e; i++)
		{
			h_addPair(h, pairs[i]);	
		}
	}
	free(pairs);//free pair array
}

//addPair
void h_addPair(sHashTable * h, sPair * p)
{
	//call hash-function
	uint bucket_pos = _h_hashKey(h, p->key, p->key_size); 
	sList * l = h->table[bucket_pos];
	if(l == NULL)//create new list if it doesn't exist already
	{
		l = list_init();
		h->table[bucket_pos] = l;
	}
	//insert in list
	list_pushBack(l, (void*)p);
	h->num_elements++;
}

//add
void h_add(sHashTable * h, const void * key, uint key_length, void * value, uint val_length)
{
	//create new pair from given key and value
	sPair * p = (sPair*)malloc(sizeof(sPair));
	p->value = value;
	p->key = malloc(key_length);
	//copy key
	memcpy(p->key, key, key_length);

    p->key_size = key_length;
    p->val_size = val_length;
	h_addPair(h, p);
}

//set
void* h_set(sHashTable * h, const void * key, uint key_length, void * new_data, uint val_length)
{
	sListItem * l = _h_getPair(h, key, key_length, NULL);
    //if not in hashmap already add
    if(l == NULL){
        h_add(h,key,key_length,new_data, val_length);
        return new_data;
    }
	sPair * p = (sPair*)l->data;
	//overwrite exisiting value pointer
	void * old_value = p->value;
	p->value = new_data;
    p->val_size = val_length;
	return old_value;
}

//get
void* h_get(sHashTable * h, const void * key, uint key_length, uint* val_length)
{
	sListItem * l = _h_getPair(h, key, key_length, NULL);
	if(l == NULL)//value not found
		return NULL;
	if(val_length != NULL)
		*val_length = ((sPair*) l->data) -> val_size;
	return ((sPair*)l->data)->value;
}

//remove
void* h_remove(sHashTable*h, const void * key, uint key_length)
{
	uint bucket_pos = 0;
	sListItem * l = _h_getPair(h, key, key_length, &bucket_pos);
	if(l == NULL)
		return NULL;
	
	void * value = ((sPair*)l->data)->value;
	void * dkey = ((sPair*)l->data)->key;
	free(dkey);
	free(l->data);
	h->num_elements--;
	sList * list = h->table[bucket_pos]; 
	list_removeItem(list, l);
	if(list->size == 0)//list is empty
	{
		//remove whole list
		list_free(list);
		h->table[bucket_pos] = NULL;
	}
	return value; 
}

//clear
void** h_clear(sHashTable * h, uint * num_elements)
{
	if(num_elements != NULL)
		*num_elements = h->num_elements;
	if(h->num_elements == 0)//no elements in table
		return NULL;
	void ** values = (void**) malloc(sizeof(void*) * h->num_elements);
	uint value_count = 0;
	uint size = h->size;
	//getting all values
	for(uint i = 0; i < size; i++)
	{
		sList * list= h->table[i];
		if(list != NULL)
		{
			for(sListItem * l = list->first; l != NULL; l = (sListItem*)l->next)
			{
				values[value_count] = ((sPair*)l->data)->value;
				value_count++;
				//free key
				free(((sPair*)l->data)->key);

				//free pair
				free(l->data);
			}
			//free list
			list_free(list);
			h->table[i] = NULL;
		}
	}
	h->num_elements = 0;
	return values;
}

void h_printTable(sHashTable * h, void (*print_value)(void * value,void * inf),void* info)
{
	uint size = h->size;
	for(uint i = 0; i < size; i++)
	{
		printf("%d: ", i);
		sList * list = h->table[i];
		if(list != NULL)
		{
			printf("(%d): ", list->size);
			for(sListItem * l = list->first; l != NULL; l = (sListItem*)l->next)
			{
				(*print_value)(((sPair*)l->data)->value, info);
				if(l != list->last)
					printf(", ");
			}
		}
		else
			printf("NULL");

		printf("\n");
	}
}

//hashKey
uint _h_hashKey(sHashTable * h, const void * key, uint key_length)
{
	return (*(h->hash_func))(key, key_length, h->hash_info)%h->size;
}

//getPair
sListItem* _h_getPair(sHashTable * h, const void * key, uint key_length, uint * bucket_pos)
{
	int index = _h_hashKey(h, key, key_length);
	if(bucket_pos != NULL)
		*bucket_pos = index;
	sList * list = h->table[index];
	if(list == NULL)
		return NULL;
	for(sListItem * l = list->first; l != NULL; l = (sListItem*)l->next)
	{
		sPair * p = (sPair*)l->data;
		if(key_length == p->key_size)//identical length?
		{
			if(memcmp(p->key, key, key_length) == 0)//key is equal to given one -> value found!
			{
				return l;
			}
		}
	}
	return NULL;
}


