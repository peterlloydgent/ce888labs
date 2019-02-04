import numpy as np

def shuffle(array):
    np.random.shuffle(array)

def mergeArrays(sample1,sample2):
    return np.concatenate([sample1,sample2])

def  perm_test(new1,new2,size,perm_times):
    counter = 0
    for i in range(0,perm_times):
        new = np.array(mergeArrays(new1,new2))
        shuffle(new)
        two_way_split = np.split(new,2)
        pold_mean = two_way_split[0].mean()
        pnew_mean = two_way_split[1].mean()
        tperm = pnew_mean - pold_mean
        if tperm > size:
            counter+=1
    return counter/perm_times

def power(sample1, sample2, reps, size, alpha):
    count = 0
    for i in range(0,reps):
        new_array1 = np.random.choice(sample1,sample1.size)
        new_array2 = np.random.choice(sample2,sample2.size)
        p = perm_test(new_array1,new_array2,size,reps)
        if(p<1-alpha):
            count +=1
    return count/reps


if __name__ == "__main__":
    array1 = np.array([0,0,0,1,1,0,0])
    array2 = np.array([0,1,1,0,0,0,1])
    print(power(array1,array2,1000,0.2,0.95))
