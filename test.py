#!/usr/bin/env python

import memory
import autograd.numpy as np
import addressing as add

mem1 = np.zeros((4,5))

write_weight = np.array([0.5,0.25,0.15,0.1]) 
e_t = np.array([1,1,1,1,1])
a_t = np.array([1,2,1,2,1])

mem2 = memory.write(mem1,write_weight,e_t,a_t)

read_weight = np.array([0.25,0.5,0.1,0.15])
print "read weight: ", read_weight
read_result = memory.read(mem2,read_weight)
print "read_result: "
print read_result

a = np.array([1,2,3,4])
b = np.array([1,2,3,4])

print "sim"
print add.cosine_sim(a,b)

# make it so that things will be more similar
write_weight = np.array([1,0,0,0]) 
e_t = np.array([1,1,1,1,1])
a_t = np.array([1,2,3,4,5])

mem3 = memory.write(mem2,write_weight,e_t,a_t)

k_t = np.array([1,2,3,4,5])
b_t = 1

print "focus test: "
focus = add.content_focus(k_t,b_t,mem3)
print focus
print np.sum(focus)

g_t = 0.5 
s_t = np.array([0,0,0,1])
gamma_t = 1
w_old = focus # we need to have 
w_content = focus

print add.location_focus(g_t,s_t,gamma_t,w_old,w_content)

print "success!"

