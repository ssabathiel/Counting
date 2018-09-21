import numpy as np
import scipy.misc as smp
import random


# Good side about pixel manipulation: http://pythoninformer.com/python-libraries/numpy/numpy-and-images/

def create_object_set(it_step):


    img_size = 28
    max_objects = 8
    n_objects = random.randint(0, max_objects)
    n_neighbours = 2
    data = np.zeros((img_size, img_size), dtype=np.uint8)
    obj_size_var = 2
    complete_breaky = 0
    for n in range(n_objects):
        breaky = 0
        max_put_try = 50
        put_try=0
        current_obj_size_var =  random.randint(0,obj_size_var)
        n_neighbours = n_neighbours - current_obj_size_var
        while (breaky == 0):
            rand_pixel_1 = random.randint(0,img_size-1)
            rand_pixel_2 = random.randint(0, img_size-1)
            # Check if object within borders
            if (rand_pixel_1 + n_neighbours <= img_size and rand_pixel_2 + n_neighbours <= img_size and rand_pixel_1 - n_neighbours > 0 and rand_pixel_2 - n_neighbours > 0):
                breaky=1
                # Check if any objects are overlapping
                for i in range(2 * n_neighbours + 2):
                    for j in range(2 * n_neighbours + 2):
                        if (rand_pixel_1 - n_neighbours -1  + i != img_size and rand_pixel_2 - n_neighbours - 1  + j != img_size):
                            if(data[rand_pixel_1 - n_neighbours -1  + i, rand_pixel_2 - n_neighbours - 1 + j] == 255):
                                breaky=0


            put_try += 1
            if(put_try >= max_put_try):
                breaky = 1
                complete_breaky = 1
                print("ATTENTION: OBJECTS COULD NOT FIT INTO WINDOW. CHOOSE DIFFERENT SIZE FOR WINDOW OR OBJECTS")
        data[rand_pixel_1,rand_pixel_2] = 255

        for i in range(2*n_neighbours):
            for j in range(2 * n_neighbours ):
                data[rand_pixel_1 - n_neighbours + i, rand_pixel_2 - n_neighbours + j ] = 255



    #print("n_objects = ", n_objects)
    img = smp.toimage(data)  # Create a PIL image
    file_name = 'Object_Sets/' + str(it_step) + '.jpg'
    img.save(file_name)
    #img.show()  # View in default viewer


    # Data 2D-image --> 1D-array ( for NN as Input )
    data_flatten = data.flatten()
    # Build one-hot array from number of objects
    n_obj_one_hot = np.zeros(max_objects)
    n_obj_one_hot[n_objects-1] = 1

    return data_flatten, n_obj_one_hot




# Prepare Training data
mult_img, mult_class = create_object_set(0)
for i in range(10000):
    data_flatten, n_obj_one_hot = create_object_set(i)
    mult_img = np.vstack( [mult_img,data_flatten])
    mult_class = np.vstack( (mult_class, n_obj_one_hot))
np.save('Object_Sets_Array/trX', mult_img)
np.save('Object_Sets_Array/trY', mult_class)
print(mult_img.shape)
print(mult_class.shape)

# Prepare Test data
mult_imgT, mult_classT = create_object_set(0)
for i in range(2000):
    data_flatten, n_obj_one_hot = create_object_set(i)
    mult_imgT = np.vstack( [mult_imgT,data_flatten])
    mult_classT = np.vstack( (mult_classT, n_obj_one_hot))
np.save('Object_Sets_Array/teX', mult_img)
np.save('Object_Sets_Array/teY', mult_class)





recovered_file = np.load('Object_Sets_Array/trX.npy')
print(mult_imgT.shape)
print(mult_classT.shape)
#print(recovered_file.shape)

























