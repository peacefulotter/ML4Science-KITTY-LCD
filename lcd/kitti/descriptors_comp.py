import faiss

# arr_point_descriptors( nb pc neighbourhoods, 256) : a pcs grouped into 1024 point neighbourhoods with their corresponding descriptors
# arr_patch_descriptors(nb img neighbourhoods, 256) : patches grouped into 32x32 patch neighbourhoods with their corresponding descriptors
def create_pc_to_img_array(arr_point_descriptors, arr_patch_descriptors):
    if arr_point_descriptors.shape[1] != arr_patch_descriptors.shape[1]:
        print('descriptor lengths do not match')
        return None

    d = arr_point_descriptors.shape[1] #dimension of descriptors and of index
    index = faiss.IndexFlatL2(d)   # build the index: uses euclidean distance to search
    print(index.is_trained)
    index.add(arr_patch_descriptors)                  # add vectors to the index
    print(index.ntotal)
    k = 1
    #index.search returns the index within arr_point_descriptor that is closest to arr_patch_descriptors
    dist, corresponding_indexes = index.search(arr_point_descriptors, k) #search for k nearest neighbours
    return corresponding_indexes # corresponding images(nb img neighbours) = arr of indexes inside arr_point_descriptors

