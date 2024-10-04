# tensors and variables
import tensorflow as tf
import numpy as np

# indexing

# tensor_indexed=tf.constant([3,6,2,4,6,66,7])
# print(tensor_indexed)
# print(tensor_indexed[0])
# print(tensor_indexed[2:7:2])
# print(tf.range(2,5+1))

# tensor_two_d=tf.constant([[1,2,0],[3,5,-1],[1,5,6],[2,3,8]])
# print(tensor_two_d[1:3,1:3])

# mathematic operations in tensorflow

x_abs = tf.constant([-2.25,3.25])
print(tf.abs(x_abs))
print(tf.abs(tf.constant(-0.2)))

x_abs_complex = tf.constant([-2.25 + 4.75j])
print(tf.abs(x_abs_complex))

x1=tf.constant([5,3,6,6,4,6],dtype=tf.int32)
x2=tf.constant([7,6,2,6,0,11],dtype=tf.int32)
print(tf.add(x1,x2))
print(tf.subtract(x1,x2))
print(tf.multiply(x1,x2))
print(tf.divide(x1,x2)) # use the divide nonan to take care of the division by 0 error
print(tf.math.divide_no_nan(x1,x2))

# tf.math.maximum gives the maximum of 2 tensors and tf.math.minimum gives the minimum of 2 tensors

x = tf.constant([0,0,0,0])
y = tf.constant([-5,-2,0,3])
print(tf.math.minimum(x,y))
print(tf.math.maximum(x,y))

# argmax

x_argmax = tf.constant([[2,20,30,3,6],[3,11,16,1,8],[14,45,23,5,27]])
print(x_argmax.shape)
print(tf.math.argmax(x_argmax,0)) # here it compares all three index wise
print(tf.math.argmax(x_argmax,1)) # here we compare max location of the list
x_argmax = tf.constant([200,120,30,3,6])
print(tf.math.argmax(x_argmax)) # this gives the position of the maximum value's index
print(tf.math.argmin(x_argmax)) # position of the minimum value

#power method

x=tf.constant([[2,2],[3,3]])
y=tf.constant([[8,16],[2,3]])
print(tf.pow(x,y)) # elements of x to the power of elements of y

print(tf.pow(tf.constant(2),tf.constant(3)))

# sum
tensor_two_d=tf.constant([
    [1,2.,0],
    [3,5,-1],
    [1,5,6],
    [2,3,8]
])
print(tf.math.reduce_sum(tensor_two_d,axis=None,keepdims=False,name=None)) # here this is the total sum of all the elements of 2d tensor
print(tf.math.reduce_max(tensor_two_d,axis=None,keepdims=False,name=None))
print(tf.math.reduce_min(tensor_two_d,axis=None,keepdims=False,name=None))
print()
print(tf.math.top_k(tensor_two_d,k=2))

