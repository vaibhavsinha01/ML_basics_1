# tensor basics
# tensors are multidimensional arrays
import tensorflow as tf
import numpy as np

tensor_zero_d=tf.constant(4)
tensor_bool = tf.constant([True,True,False])
tensor_string = tf.constant(["Hello World","hi"])
print(tensor_string)
print(tensor_bool)
print(tensor_zero_d)
tensor_one_d=tf.constant([2,0,-3,8,90],dtype=tf.float32)
casted_tensor_one_d=tf.cast(tensor_one_d,dtype=tf.int16)  # this is the cast function
print(tensor_one_d)
print(casted_tensor_one_d)
tensor_two_d=tf.constant([
    [1,2.,0],
    [3,5,-1],
    [1,5,6],
    [2,3,8]
])
print(tensor_two_d)
print(tensor_two_d[0:3,2:3])

tensor_three_d=tf.constant([
    [[1,2,0],[3,5,1]],[[10,2,0],[1,0,2]],[[5,8,0],[2,7,0]],[[2,1,9],[4,-3,32]]
])
print(tensor_three_d)

print(tensor_three_d.shape)
print(tensor_three_d.ndim) # tell us about the dimensions

# convert np array in a tensor
np_array=np.array([1,2,4])
print(np_array)
converted_tensor=tf.convert_to_tensor(np_array)
print(converted_tensor)

eye_tensor = tf.eye(
    num_rows=3,
    num_columns=None,
    batch_shape=None,
    dtype=tf.dtypes.float32,
    name=None
)
eye_tensor_2 = tf.eye(
    num_rows=5,
    num_columns=3,
    batch_shape=None,
    dtype=tf.dtypes.float32,
    name=None
)
eye_tensor_3 = tf.eye(
    num_rows=5,
    num_columns=None,
    batch_shape=[2,],
    dtype=tf.dtypes.float32,
    name=None
)
print(eye_tensor)
print(eye_tensor*3)
print(eye_tensor_2)
print(eye_tensor_3)

# fill method

fill_tensor=tf.fill([1,3,4],5,name=None)
ones_tensor = tf.ones([5,3],dtype=tf.dtypes.float32,name=None)
zero_tensor = tf.zeros([5,3],dtype=tf.dtypes.float32,name=None)
ones_like_tensor = tf.ones_like(fill_tensor) # same shape of 1 as the input similar for zero like

print(fill_tensor)
print(zero_tensor)
print(ones_tensor)
print(ones_like_tensor)

# 50:00 start learning from rank of a matrixs

# this gives the rank of a matrix
t=tf.constant([[[1,1,1],[2,2,2]],[[3,3,3],[4,4,4]]])
t1=tf.constant([[1,1,1],[2,2,2]])
print(t)
print(t1)
print()
print(tf.rank(t)) # 3
print(tf.rank(t1)) # 2
print(tf.size(t)) # 12
# size method gives the size of a tensor

random_tensor=tf.random.normal(
    [3,2],
    mean=0.0,
    stddev=1.0,
    dtype=tf.dtypes.float32,
    seed=None,
    name=None
)
print(random_tensor)

uniform_tensor=tf.random.uniform(
    [3,2],
    minval=0,
    maxval=100,
    dtype=tf.dtypes.float32,
    seed=None,
    name=None
)
print(uniform_tensor)

tf.random.set_seed(5)
print(tf.random.uniform(shape=[3,],maxval=5,dtype=tf.int32,seed=10))
print(tf.random.uniform(shape=[3,],maxval=5,dtype=tf.int32,seed=10))
print(tf.random.uniform(shape=[3,],maxval=5,dtype=tf.int32,seed=10))
print(tf.random.uniform(shape=[3,],maxval=5,dtype=tf.int32,seed=10))

