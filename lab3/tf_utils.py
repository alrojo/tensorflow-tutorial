import tensorflow as tf
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.framework import ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import math_ops


###
# custom loss function, similar to tensorflows but uses 3D tensors
# instead of a list of 2D tensors
def sequence_loss_tensor(logits, targets, weights, num_classes,
                         average_across_timesteps=True,
                         softmax_loss_function=None, name=None):
    """Weighted cross-entropy loss for a sequence of logits (per example).
    """
    with ops.op_scope([logits, targets, weights], name, "sequence_loss_by_example"):
        probs_flat = tf.reshape(logits, [-1, num_classes])
        targets = tf.reshape(targets, [-1])
        if softmax_loss_function is None:
            crossent = nn_ops.sparse_softmax_cross_entropy_with_logits(
                    probs_flat, targets)
        else:
            crossent = softmax_loss_function(probs_flat, targets)
        crossent = crossent * tf.reshape(weights, [-1])
        crossent = tf.reduce_sum(crossent)
        total_size = math_ops.reduce_sum(weights)
        total_size += 1e-12 # to avoid division by zero
        crossent /= total_size
        return crossent


###
# a custom masking function, takes sequence lengths and makes masks
def mask(sequence_lengths):
    # based on this SO answer: http://stackoverflow.com/a/34138336/118173
    batch_size = tf.shape(sequence_lengths)[0]
    max_len = tf.reduce_max(sequence_lengths)

    lengths_transposed = tf.expand_dims(sequence_lengths, 1)

    rng = tf.range(max_len)
    rng_row = tf.expand_dims(rng, 0)

    return tf.less(rng_row, lengths_transposed)


###
# a custom encoder function (in case we cant get tensorflows to work)

def encoder(inputs, lengths, name, num_units, reverse=False, swap=False):
    with tf.variable_scope(name):
        weight_initializer = tf.truncated_normal_initializer(stddev=0.1)
        input_units = inputs.get_shape()[2]
        W_z = tf.get_variable('W_z',
                              shape=[input_units+num_units, num_units],
                              initializer=weight_initializer)
        W_r = tf.get_variable('W_r',
                              shape=[input_units+num_units, num_units],
                              initializer=weight_initializer)
        W_h = tf.get_variable('W_h',
                              shape=[input_units+num_units, num_units],
                              initializer=weight_initializer)
        b_z = tf.get_variable('b_z',
                              shape=[num_units],
                              initializer=tf.constant_initializer(1.0))
        b_r = tf.get_variable('b_r',
                              shape=[num_units],
                              initializer=tf.constant_initializer(1.0))
        b_h = tf.get_variable('b_h',
                              shape=[num_units],
                              initializer=tf.constant_initializer())

        max_sequence_length = tf.reduce_max(lengths)
        min_sequence_length = tf.reduce_min(lengths)

        time = tf.constant(0)

        state_shape = tf.concat(0, [tf.expand_dims(tf.shape(lengths)[0], 0),
                                    tf.expand_dims(tf.constant(num_units), 0)])
        # state_shape = tf.Print(state_shape, [state_shape])
        state = tf.zeros(state_shape, dtype=tf.float32)

        if reverse:
            inputs = tf.reverse(inputs, dims=[False, True, False])
        inputs = tf.transpose(inputs, perm=[1, 0, 2])
        input_ta = tensor_array_ops.TensorArray(tf.float32, size=1, dynamic_size=True)
        input_ta = input_ta.unpack(inputs)

        output_ta = tensor_array_ops.TensorArray(tf.float32, size=1, dynamic_size=True)

        def encoder_cond(time, state, output_ta_t):
            return tf.less(time, max_sequence_length)

        def encoder_body(time, old_state, output_ta_t):
            x_t = input_ta.read(time)

            con = tf.concat(1, [x_t, old_state])
            z = tf.sigmoid(tf.matmul(con, W_z) + b_z)
            r = tf.sigmoid(tf.matmul(con, W_r) + b_r)
            con = tf.concat(1, [x_t, r*old_state])
            h = tf.tanh(tf.matmul(con, W_h) + b_h)
            new_state = (1-z)*h + z*old_state

            output_ta_t = output_ta_t.write(time, new_state)

            def updateall():
                return new_state

            def updatesome():
                if reverse:
                    return tf.select(
                        tf.greater_equal(time, max_sequence_length-lengths),
                        new_state,
                        old_state)
                else:
                    return tf.select(tf.less(time, lengths), new_state, old_state)

            if reverse:
                state = tf.cond(
                    tf.greater_equal(time, max_sequence_length-min_sequence_length),
                    updateall,
                    updatesome)
            else:
                state = tf.cond(tf.less(time, min_sequence_length), updateall, updatesome)

            return (time + 1, state, output_ta_t)

        loop_vars = [time, state, output_ta]

        time, state, output_ta = tf.while_loop(encoder_cond, encoder_body, loop_vars, swap_memory=swap)

        enc_state = state
        enc_out = tf.transpose(output_ta.pack(), perm=[1, 0, 2])

        if reverse:
            enc_out = tf.reverse(enc_out, dims=[False, True, False])

        enc_out.set_shape([None, None, num_units])

        return enc_state, enc_out


###
# a custom decoder function

def decoder(initial_state, target_input, target_len, num_units,
            embeddings, W_out, b_out,
            W_z_x_init = tf.truncated_normal_initializer(stddev=0.1),
            W_z_h_init = tf.truncated_normal_initializer(stddev=0.1),
            W_r_x_init = tf.truncated_normal_initializer(stddev=0.1),
            W_r_h_init = tf.truncated_normal_initializer(stddev=0.1),
            W_h_x_init = tf.truncated_normal_initializer(stddev=0.1),
            W_h_h_init = tf.truncated_normal_initializer(stddev=0.1),
            b_z_init = tf.constant_initializer(0.0),
            b_r_init = tf.constant_initializer(0.0),
            b_h_init = tf.constant_initializer(0.0),
            name='decoder', swap=False):
    """decoder
        TODO
    """


    with tf.variable_scope(name):
        # we need the max seq len to optimize our RNN computation later on
        max_sequence_length = tf.reduce_max(target_len)
        # target_dims is just the embedding size
        target_dims = target_input.get_shape()[2]
        # set up weights for the GRU gates
        var = tf.get_variable # for ease of use
        # target_dims + num_units is because we stack embeddings and prev. hidden state to
        # optimize speed
        W_z_x = var('W_z_x', shape=[target_dims, num_units], initializer=W_z_x_init)
        W_z_h = var('W_z_h', shape=[num_units, num_units], initializer=W_z_h_init)
        b_z = var('b_z', shape=[num_units], initializer=b_z_init)
        W_r_x = var('W_r_x', shape=[target_dims, num_units], initializer=W_r_x_init)
        W_r_h = var('W_r_h', shape=[num_units, num_units], initializer=W_r_h_init)
        b_r = var('b_r', shape=[num_units], initializer=b_r_init)
        W_h_x = var('W_h_x', shape=[target_dims, num_units], initializer=W_h_x_init)
        W_h_h = var('W_h_h', shape=[num_units, num_units], initializer=W_h_h_init)
        b_h = var('b_h', shape=[num_units], initializer=b_h_init)

        # make inputs time-major
        inputs = tf.transpose(target_input, perm=[1, 0, 2])
        # make tensor array for inputs, these are dynamic and used in the while-loop
        # these are not in the api documentation yet, you will have to look at github.com/tensorflow
        input_ta = tensor_array_ops.TensorArray(tf.float32, size=1, dynamic_size=True)
        input_ta = input_ta.unpack(inputs)

        # function to the while-loop, for early stopping
        def decoder_cond(time, state, output_ta_t):
            return tf.less(time, max_sequence_length)

        # the body_builder is just a wrapper to parse feedback
        def decoder_body_builder(feedback=False):
            # the decoder body, this is where the RNN magic happens!
            def decoder_body(time, old_state, output_ta_t):
                # when validating we need previous prediction, handle in feedback
                if feedback:
                    def from_previous():
                        prev_1 = tf.matmul(old_state, W_out) + b_out
                        return tf.gather(embeddings, tf.argmax(prev_1, 1))
                    x_t = tf.cond(tf.greater(time, 0), from_previous, lambda: input_ta.read(0))
                else:
                    # else we just read the next timestep
                    x_t = input_ta.read(time)

                # calculate the GRU
                z = tf.sigmoid(tf.matmul(x_t, W_z_x) + tf.matmul(old_state, W_z_h) + b_z) # update gate
                r = tf.sigmoid(tf.matmul(x_t, W_r_x) + tf.matmul(old_state, W_r_h) + b_r) # reset gate
                h = tf.tanh(tf.matmul(x_t, W_h_x) + tf.matmul(r*old_state, W_h_h) + b_h) # proposed new state
                new_state = (1-z)*h + z*old_state # new state

                # writing output
                output_ta_t = output_ta_t.write(time, new_state)

                # return in "input-to-next-step" style
                return (time + 1, new_state, output_ta_t)
            return decoder_body
        # set up variables to loop with
        output_ta = tensor_array_ops.TensorArray(tf.float32, size=1, dynamic_size=True, infer_shape=False)
        time = tf.constant(0)
        loop_vars = [time, initial_state, output_ta]

        # run the while-loop for training
        _, state, output_ta = tf.while_loop(decoder_cond,
                                            decoder_body_builder(),
                                            loop_vars,
                                            swap_memory=swap)
        # run the while-loop for validation
        _, valid_state, valid_output_ta = tf.while_loop(decoder_cond,
                                                        decoder_body_builder(feedback=True),
                                                        loop_vars,
                                                        swap_memory=swap)
        # returning to batch major
        dec_out = tf.transpose(output_ta.pack(), perm=[1, 0, 2])
        valid_dec_out = tf.transpose(valid_output_ta.pack(), perm=[1, 0, 2])
        return dec_out, valid_dec_out
