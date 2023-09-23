# try my own cutom activation function - based on the ideas that: 
# - swish is good because it allows for negative contribution, less destructive transformetion of data, smoother
# - swish is bas because it has 2 places where a neuron can contribute 0: a 'dead' are and a 'sensitive' area -  if a neuon needs to contribute little, gradient descent will swtich ebtwen them unstably
# - relu and swich good because they have dead zones so useless neurons can die = sparsity

# combine these and you want:
# an activation witha  dead zone
# an activation with only on zero area (the dead zone), in fact, strictly increasing to avoid any kind of gradient descent confusion - we want it so that if a gradient says go in one direction, continuing in that direction makes it moresoe, rather than local maxima
# unbounded on one end (see swiglu paper)
# bounded on one end (see swiglu paper)

# it would be nice to have the dead zone also be the bounded area so they can keep being pushed more and more dead but we can have that And a negative area AND a postiive one without breaking the strictly increasing requirement.

# so in the end we get a sigmoid function where the rightward plateau is shifted to the centre of the graph so we have a clear, stable dead zone around x=0,y=0, we converge to a negative zone of -1 as x decreases below 0, 
# and to make it unbounded we swap add a transated elu function

@jax.jit
def modified_sigmoid(x_, alpha=1.0, d=5, epsilon=0.5):
    x=x_*5
    shift = d
    shifted_x = x - shift  # Shift the input by 5 units to the right

    sigmoid_part = (jax.nn.sigmoid(x + d)-1)*epsilon # shift rightward plateau to x=0, y=0
    linear_part = jnp.where(shifted_x >= 0, shifted_x + shift, alpha * (np.exp(shifted_x) - 1) + shift)

    return (sigmoid_part + linear_part)-(d-1)