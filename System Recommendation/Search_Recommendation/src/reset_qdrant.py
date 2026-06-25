from config import client

def reset_collection():
    collection_name='wayfair_items'

    confirm = input("Delete Collection wayfair_items? y / n")

    if confirm.lower() == 'y':
        if client.collection_exists(collection_name=collection_name):
            print(f"collection {collection_name} been removed")
        else:
            print("None Collection being removed")
    else:
        print("Operation cancel")

if __name__ == '__main__':
    reset_collection()
