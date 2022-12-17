import faiss

# arr_point_descriptors(size(pc neighbourhoods), 256) : a pc grouped into 1024 point neighbourhoods
# arr_patch_descriptors(size(img neighbourhoods), 256) : an image grouped into 32x32 patch neighbourhoods)
def create_pc_to_img_array(arr_point_descriptors, arr_patch_descriptors):
    d = arr_point_descriptors.shape[1] #dimension of descriptors and of index
    index = faiss.IndexFlatL2(d)   # build the index: uses euclidean distance to search
    print(index.is_trained)
    index.add(arr_point_descriptors)                  # add vectors to the index
    print(index.ntotal)
    k = 1
    dist, corresponding_indexes = index.search(arr_patch_descriptors, k) #search for k nearest neighbours
    return arr_point_descriptors.shape[0], corresponding_indexes # returns a pair of corresponding indexes between patches and



