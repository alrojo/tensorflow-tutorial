import numpy as np
import theano
import theano.tensor as T
from lasagne import nonlinearities
from lasagne import init
from lasagne.utils import unroll_scan
from lasagne.layers import MergeLayer
from lasagne.layers.base import Layer
from lasagne.layers import helper

import numpy as np
import theano
import theano.tensor as T
import lasagne.init as init
import lasagne.nonlinearities as nonlinearities

from lasagne.layers import Layer
import lasagne


# LSTMAttentionDecodeLayer
# Model: Encoder -> Decoder    Decoder-LSTM: ... hid_t-1 ->  hid_t  -> h_t+1 ....
#                         attention_network         |         |        |
#          weighted encoder hidden(output)      whid_t-1 -> whid_t -> wh_t+1


# LSTMAttentionDecodeFeedBackLayer
# Model: Encoder -> Decoder    Decoder-LSTM: ... hid_dec_t-1-> hid_dec_t-> hid_dec_t+1 ....
#                                                   |     /^  |    /^ |     /^
#                         attention_network         |   /     |  /    |   /
#                                                   | /       |/      | /
#          weighted encoder hidden(output)      whid_enc_t-1 -> whid_enc__t -> wh_t+1
#
# This model also allows for adden "pre-steps" to the decoder where the model can
# "comprehend the input data". basically this is just adding extra steps to the
# decoder before producing the targets
#
#


class LSTMAttentionDecodeLayer(MergeLayer):
    r"""A long short-term memory (LSTM) layer.

    Includes optional "peephole connections" and a forget gate.  Based on the
    definition in [1]_, which is the current common definition.  The output is
    computed by

    .. math ::

        i_t &= \sigma_i(W_{xi}x_t + W_{hi}h_{t-1}
               + w_{ci}\odot c_{t-1} + b_i)\\
        f_t &= \sigma_f(W_{xf}x_t + W_{hf}h_{t-1}
               + w_{cf}\odot c_{t-1} + b_f)\\
        c_t &= f_t \odot c_{t - 1}
               + i_t\sigma_c(W_{xc}x_t + W_{hc} h_{t-1} + b_c)\\
        o_t &= \sigma_o(W_{xo}x_t + W_{ho}h_{t-1} + w_{co}\odot c_t + b_o)\\
        h_t &= o_t \odot \sigma_h(c_t)

    Parameters
    ----------
    incoming : a :class:`lasagne.layers.Layer` instance or a tuple
        The layer feeding into this layer, or the expected input shape.
    num_units : int
        Number of hidden/cell units in the layer.
    W_in_to_ingate : Theano shared variable, numpy array or callable
        Initializer for input-to-input gate weight matrix (:math:`W_{xi}`).
    W_hid_to_ingate : Theano shared variable, numpy array or callable
        Initializer for hidden-to-input gate weight matrix (:math:`W_{hi}`).
    W_cell_to_ingate : Theano shared variable, numpy array or callable
        Initializer for cell-to-input gate weight vector (:math:`w_{ci}`).
    b_ingate : Theano shared variable, numpy array or callable
        Initializer for input gate bias vector (:math:`b_i`).
    nonlinearity_ingate : callable or None
        The nonlinearity that is applied to the input gate activation
        (:math:`\sigma_i`). If None is provided, no nonlinearity will be
        applied.
    W_in_to_forgetgate : Theano shared variable, numpy array or callable
        Initializer for input-to-forget gate weight matrix (:math:`W_{xf}`).
    W_hid_to_forgetgate : Theano shared variable, numpy array or callable
        Initializer for hidden-to-forget gate weight matrix (:math:`W_{hf}`).
    W_cell_to_forgetgate : Theano shared variable, numpy array or callable
        Initializer for cell-to-forget gate weight vector (:math:`w_{cf}`).
    b_forgetgate : Theano shared variable, numpy array or callable
        Initializer for forget gate bias vector (:math:`b_f`).
    nonlinearity_forgetgate : callable or None
        The nonlinearity that is applied to the forget gate activation
        (:math:`\sigma_f`). If None is provided, no nonlinearity will be
        applied.
    W_in_to_cell : Theano shared variable, numpy array or callable
        Initializer for input-to-cell weight matrix (:math:`W_{ic}`).
    W_hid_to_cell : Theano shared variable, numpy array or callable
        Initializer for hidden-to-cell weight matrix (:math:`W_{hc}`).
    b_cell : Theano shared variable, numpy array or callable
        Initializer for cell bias vector (:math:`b_c`).
    nonlinearity_cell : callable or None
        The nonlinearity that is applied to the cell activation
        (;math:`\sigma_c`). If None is provided, no nonlinearity will be
        applied.
    W_in_to_outgate : Theano shared variable, numpy array or callable
        Initializer for input-to-output gate weight matrix (:math:`W_{io}`).
    W_hid_to_outgate : Theano shared variable, numpy array or callable
        Initializer for hidden-to-output gate weight matrix (:math:`W_{ho}`).
    W_cell_to_outgate : Theano shared variable, numpy array or callable
        Initializer for cell-to-output gate weight vector (:math:`w_{co}`).
    b_outgate : Theano shared variable, numpy array or callable
        Initializer for hidden-to-input gate weight matrix (:math:`b_o`).
    nonlinearity_outgate : callable or None
        The nonlinearity that is applied to the output gate activation
        (:math:`\sigma_o`). If None is provided, no nonlinearity will be
        applied.
    nonlinearity_out : callable or None
        The nonlinearity that is applied to the output (:math:`\sigma_h`). If
        None is provided, no nonlinearity will be applied.
    cell_init : callable, np.ndarray, theano.shared or TensorVariable
        Passing in a TensorVariable allows the user to specify
        the value of `cell_init` (:math:`c_0`). In this mode `learn_init` is
        ignored for the cell state.
    hid_init : callable, np.ndarray, theano.shared or TensorVariable
        Passing in a TensorVariable allows the user to specify
        the value of `hid_init` (:math:`h_0`). In this mode `learn_init` is
        ignored for the hidden state.
    backwards : bool
        If True, process the sequence backwards and then reverse the
        output again such that the output from the layer is always
        from :math:`x_1` to :math:`x_n`.
    learn_init : bool
        If True, initial hidden values are learned. If `hid_init` or
        `cell_init` are TensorVariables then the TensorVariable is used and
        `learn_init` is ignored for that initial state.
    peepholes : bool
        If True, the LSTM uses peephole connections.
        When False, `W_cell_to_ingate`, `W_cell_to_forgetgate` and
        `W_cell_to_outgate` are ignored.
    gradient_steps : int
        Number of timesteps to include in the backpropagated gradient.
        If -1, backpropagate through the entire sequence.
    grad_clipping: False or float
        If a float is provided, the gradient messages are clipped during the
        backward pass.  If False, the gradients will not be clipped.  See [1]_
        (p. 6) for further explanation.
    unroll_scan : bool
        If True the recursion is unrolled instead of using scan. For some
        graphs this gives a significant speed up but it might also consume
        more memory. When `unroll_scan` is true then the `gradient_steps`
        setting is ignored.
    precompute_input : bool
        If True, precompute input_to_hid before iterating through
        the sequence. This can result in a speedup at the expense of
        an increase in memory usage.

    References
    ----------
    .. [1] Graves, Alex: "Generating sequences with recurrent neural networks."
           arXiv preprint arXiv:1308.0850 (2013).
    """
    def __init__(self, incoming,
                 num_units,
                 aln_num_units,
                 n_decodesteps,
                 W_align=init.Normal(0.1),
                 U_align=init.Normal(0.1),
                 v_align=init.Normal(0.1),
                 nonlinearity_align=nonlinearities.tanh,
                 W_hid_to_ingate=init.Normal(0.1),
                 W_cell_to_ingate=init.Normal(0.1),
                 b_ingate=init.Constant(0.),
                 nonlinearity_ingate=nonlinearities.sigmoid,
                 #W_in_to_forgetgate=init.Normal(0.1),
                 W_hid_to_forgetgate=init.Normal(0.1),
                 W_cell_to_forgetgate=init.Normal(0.1),
                 b_forgetgate=init.Constant(0.),
                 nonlinearity_forgetgate=nonlinearities.sigmoid,
                 #W_in_to_cell=init.Normal(0.1),
                 W_hid_to_cell=init.Normal(0.1),
                 b_cell=init.Constant(0.),
                 nonlinearity_cell=nonlinearities.tanh,
                 #W_in_to_outgate=init.Normal(0.1),
                 W_hid_to_outgate=init.Normal(0.1),
                 W_cell_to_outgate=init.Normal(0.1),
                 b_outgate=init.Constant(0.),
                 nonlinearity_outgate=nonlinearities.sigmoid,
                 nonlinearity_out=nonlinearities.tanh,
                 cell_init=init.Constant(0.),
                 hid_init=init.Constant(0.),
                 backwards=False,
                 learn_init=False,
                 peepholes=True,
                 gradient_steps=-1,
                 grad_clipping=False,
                 unroll_scan=False,
                 mask_input=None,
                 #precompute_input=True,
                 **kwargs):

        # Initialize parent layer
        # This layer inherits from a MergeLayer, because it can have two
        # inputs - the layer input, and the mask.  We will just provide the
        # layer input as incomings, unless a mask input was provided.
        incomings = [incoming]
        if mask_input is not None:
            incomings.append(mask_input)
        super(LSTMAttentionDecodeLayer, self).__init__(incomings, **kwargs)

        # For any of the nonlinearities, if None is supplied, use identity
        if nonlinearity_ingate is None:
            self.nonlinearity_ingate = nonlinearities.identity
        else:
            self.nonlinearity_ingate = nonlinearity_ingate

        if nonlinearity_forgetgate is None:
            self.nonlinearity_forgetgate = nonlinearities.identity
        else:
            self.nonlinearity_forgetgate = nonlinearity_forgetgate

        if nonlinearity_cell is None:
            self.nonlinearity_cell = nonlinearities.identity
        else:
            self.nonlinearity_cell = nonlinearity_cell

        if nonlinearity_outgate is None:
            self.nonlinearity_outgate = nonlinearities.identity
        else:
            self.nonlinearity_outgate = nonlinearity_outgate

        if nonlinearity_out is None:
            self.nonlinearity_out = nonlinearities.identity
        else:
            self.nonlinearity_out = nonlinearity_out

        self.learn_init = learn_init
        self.num_units = num_units
        self.backwards = backwards
        self.peepholes = peepholes
        self.gradient_steps = gradient_steps
        self.grad_clipping = grad_clipping
        self.unroll_scan = unroll_scan
        self.n_decodesteps = n_decodesteps
        self.aln_num_units = aln_num_units
        self.nonlinearity_align = nonlinearity_align

        # Retrieve the dimensionality of the incoming layer
        input_shape = self.input_shapes[0]
        if unroll_scan and input_shape[1] is None:
            raise ValueError("Input sequence length cannot be specified as "
                             "None when unroll_scan is True")

        num_inputs = np.prod(self.input_shape[2:])

        # Initialize parameters using the supplied args
        #self.W_in_to_ingate = self.add_param(
        #    W_in_to_ingate, (num_inputs, num_units), name="W_in_to_ingate")

        self.W_hid_to_ingate = self.add_param(
            W_hid_to_ingate, (num_units, num_units), name="W_hid_to_ingate")

        self.b_ingate = self.add_param(
            b_ingate, (num_units,), name="b_ingate", regularizable=False)

        #self.W_in_to_forgetgate = self.add_param(
        #    W_in_to_forgetgate, (num_inputs, num_units),
        #    name="W_in_to_forgetgate")

        self.W_hid_to_forgetgate = self.add_param(
            W_hid_to_forgetgate, (num_units, num_units),
            name="W_hid_to_forgetgate")

        self.b_forgetgate = self.add_param(
            b_forgetgate, (num_units,), name="b_forgetgate",
            regularizable=False)

        #self.W_in_to_cell = self.add_param(
        #    W_in_to_cell, (num_inputs, num_units), name="W_in_to_cell")

        self.W_hid_to_cell = self.add_param(
            W_hid_to_cell, (num_units, num_units), name="W_hid_to_cell")

        self.b_cell = self.add_param(
            b_cell, (num_units,), name="b_cell", regularizable=False)

        #self.W_in_to_outgate = self.add_param(
        #    W_in_to_outgate, (num_inputs, num_units), name="W_in_to_outgate")

        self.W_hid_to_outgate = self.add_param(
            W_hid_to_outgate, (num_units, num_units), name="W_hid_to_outgate")

        self.b_outgate = self.add_param(
            b_outgate, (num_units,), name="b_outgate", regularizable=False)

        # Stack input weight matrices into a (num_inputs, 4*num_units)
        # matrix, which speeds up computation
        #self.W_in_stacked = T.concatenate(
        #    [self.W_in_to_ingate, self.W_in_to_forgetgate,
        #     self.W_in_to_cell, self.W_in_to_outgate], axis=1)

        # Same for hidden weight matrices
        self.W_hid_stacked = T.concatenate(
            [self.W_hid_to_ingate, self.W_hid_to_forgetgate,
             self.W_hid_to_cell, self.W_hid_to_outgate], axis=1)

        # Stack biases into a (4*num_units) vector
        self.b_stacked = T.concatenate(
            [self.b_ingate, self.b_forgetgate,
             self.b_cell, self.b_outgate], axis=0)

        # If peephole (cell to gate) connections were enabled, initialize
        # peephole connections.  These are elementwise products with the cell
        # state, so they are represented as vectors.
        if self.peepholes:
            self.W_cell_to_ingate = self.add_param(
                W_cell_to_ingate, (num_units, ), name="W_cell_to_ingate")

            self.W_cell_to_forgetgate = self.add_param(
                W_cell_to_forgetgate, (num_units, ),
                name="W_cell_to_forgetgate")

            self.W_cell_to_outgate = self.add_param(
                W_cell_to_outgate, (num_units, ), name="W_cell_to_outgate")

        self.W_align = self.add_param(W_align, (num_units, self.aln_num_units),
                                   name="AlignSeqOutputLayer: (aln) W_a")
        self.U_align = self.add_param(U_align, (num_inputs, self.aln_num_units),
                           name="AlignSeqOutputLayer: (aln) U_a")
        self.v_align = self.add_param(v_align, (self.aln_num_units, 1),
                                 name="AlignSeqOutputLayer: v_a")


        # Setup initial values for the cell and the hidden units
        if isinstance(cell_init, T.TensorVariable):
            if cell_init.ndim != 2:
                raise ValueError(
                    "When cell_init is provided as a TensorVariable, it should"
                    " have 2 dimensions and have shape (num_batch, num_units)")
            self.cell_init = cell_init
        else:
            self.cell_init = self.add_param(
                cell_init, (1, num_units), name="cell_init",
                trainable=learn_init, regularizable=False)

        if isinstance(hid_init, T.TensorVariable):
            if hid_init.ndim != 2:
                raise ValueError(
                    "When hid_init is provided as a TensorVariable, it should "
                    "have 2 dimensions and have shape (num_batch, num_units)")
            self.hid_init = hid_init
        else:
            self.hid_init = self.add_param(
                hid_init, (1, self.num_units), name="hid_init",
                trainable=learn_init, regularizable=False)

    def get_output_shape_for(self, input_shapes):
        input_shape = input_shapes[0]
        return input_shape[0], None, self.num_units

    def get_output_for(self, inputs, **kwargs):
        """
        Compute this layer's output function given a symbolic input variable

        Parameters
        ----------
        input : theano.TensorType
            Symbolic input variable.
        mask : theano.TensorType
            Theano variable denoting whether each time step in each
            sequence in the batch is part of the sequence or not.  If ``None``,
            then it is assumed that all sequences are of the same length.  If
            not all sequences are of the same length, then it must be
            supplied as a matrix of shape ``(n_batch, n_time_steps)`` where
            ``mask[i, j] = 1`` when ``j <= (length of sequence i)`` and
            ``mask[i, j] = 0`` when ``j > (length of sequence i)``.

        Returns
        -------
        layer_output : theano.TensorType
            Symblic output variable.
        """
        input = inputs[0]
        # Retrieve the mask when it is supplied
        mask = inputs[1] if len(inputs) > 1 else None

        # Treat all dimensions after the second as flattened feature dimensions
        # Retrieve the layer input
        if input.ndim > 3:
            input = input.reshape((input.shape[0], input.shape[1],
                                   T.prod(input.shape[2:])))
        num_batch = input.shape[0]
        encode_seqlen = input.shape[1]

        # At each call to scan, input_n will be (n_time_steps, 4*num_units).
        # We define a slicing function that extract the input to each LSTM gate
        def slice_w(x, n):
            return x[:, n*self.num_units:(n+1)*self.num_units]

        # Create single recurrent computation step function
        # input_n is the n'th vector of the input
        def step(cell_previous, hid_previous, a_prev,
                 hUa, W_align, v_align,
                 W_hid_stacked, W_cell_to_ingate, W_cell_to_forgetgate,
                 W_cell_to_outgate, b_stacked):

            # Calculate gates pre-activations and slice
            gates = T.dot(hid_previous, W_hid_stacked) + b_stacked

            # Clip gradients
            if self.grad_clipping is not False:
                gates = theano.gradient.grad_clip(
                    gates, -self.grad_clipping, self.grad_clipping)

            # Extract the pre-activation gate values
            ingate = slice_w(gates, 0)
            forgetgate = slice_w(gates, 1)
            cell_input = slice_w(gates, 2)
            outgate = slice_w(gates, 3)

            if self.peepholes:
                # Compute peephole connections
                ingate += cell_previous*W_cell_to_ingate
                forgetgate += cell_previous*W_cell_to_forgetgate

            # Apply nonlinearities
            ingate = self.nonlinearity_ingate(ingate)
            forgetgate = self.nonlinearity_forgetgate(forgetgate)
            cell_input = self.nonlinearity_cell(cell_input)
            outgate = self.nonlinearity_outgate(outgate)

            # Compute new cell value
            cell = forgetgate*cell_previous + ingate*cell_input

            if self.peepholes:
                outgate += cell*W_cell_to_outgate

            # W_align:  (num_units, aln_num_units)
            # U_align:  (num_feats, aln_num_units)
            # v_align:  (aln_num_units, 1)
            # hUa:      (BS, Seqlen, aln_num_units)
            # hid:      (BS, num_units_dec)
            # input:    (BS, Seqlen, num_inputs)

            # Compute new hidden unit activation
            hid = outgate*self.nonlinearity_out(cell)

            #compute (unormalized) attetion vector
            sWa = T.dot(hid, W_align)       # (BS, aln_num_units)
            sWa = sWa.dimshuffle(0, 'x', 1)   # (BS, 1, aln_num_units)
            tanh_sWahUa = self.nonlinearity_align(sWa + hUa)
                                            # (BS, seqlen, num_units_aln)

            # CALCULATE WEIGHT FOR EACH HIDDEN STATE VECTOR
            a = T.dot(tanh_sWahUa, v_align)  # (BS, Seqlen, 1)
            a = T.reshape(a, (a.shape[0], a.shape[1]))
            #                                # (BS, Seqlen)
            # # ->(BS, seq_len)
            #a = a.squeeze()
            #a = a*a
            #a = a*mask - (1-mask)*10000 #this line does not work
            #a = T.reshape(a, (input.shape[0], input.shape[1]))

            #alpha = T.nnet.softmax(a)
            #alpha = T.reshape(alpha, (input.shape[0], input.shape[1]))

            #
            # # create alpha in dim (batch_size, seq_len, 1)

            #
            #weighted_hidden = input * alpha.dimshuffle(0, 1, 'x')
            #weighted_hidden = T.sum(weighted_hidden, axis=1)  #sum seqlen out

            return [cell, hid, a]

        sequences = []
        step_fun = step

        ones = T.ones((num_batch, 1))
        if isinstance(self.cell_init, T.TensorVariable):
            cell_init = self.cell_init
        else:
            # Dot against a 1s vector to repeat to shape (num_batch, num_units)
            cell_init = T.dot(ones, self.cell_init)

        if isinstance(self.hid_init, T.TensorVariable):
            hid_init = self.hid_init
        else:
            # Dot against a 1s vector to repeat to shape (num_batch, num_units)
            hid_init = T.dot(ones, self.hid_init)

        #weighted_hidden_init = T.zeros((num_batch, input.shape[2]))
        alpha_init = T.zeros((num_batch, encode_seqlen))

        # The hidden-to-hidden weight matrix is always used in step

        hUa = T.dot(input, self.U_align)   # (num_batch, seq_len, num_units_aln)

        non_seqs = [hUa, self.W_align, self.v_align,
                    self.W_hid_stacked]
        # The "peephole" weight matrices are only used when self.peepholes=True
        if self.peepholes:
            non_seqs += [self.W_cell_to_ingate,
                         self.W_cell_to_forgetgate,
                         self.W_cell_to_outgate]
        # theano.scan only allows for positional arguments, so when
        # self.peepholes is False, we need to supply fake placeholder arguments
        # for the three peephole matrices.
        else:
            non_seqs += [(), (), ()]
        # When we aren't precomputing the input outside of scan, we need to
        # provide the input weights and biases to the step function
        non_seqs += [self.b_stacked]

        if self.unroll_scan:
            # Explicitly unroll the recurrence instead of using scan
            cell_out, hid_out, a_out = unroll_scan(
                fn=step_fun,
                sequences=sequences,
                outputs_info=[cell_init, hid_init, alpha_init],
                go_backwards=self.backwards,
                non_sequences=non_seqs,
                n_steps=self.n_decodesteps)
        else:
            # Scan op iterates over first dimension of input and repeatedly
            # applies the step function
            cell_out, hid_out, a_out = theano.scan(
                fn=step_fun,
                sequences=sequences,
                outputs_info=[cell_init, hid_init, alpha_init],
                go_backwards=self.backwards,
                truncate_gradient=self.gradient_steps,
                non_sequences=non_seqs,
                n_steps=self.n_decodesteps,
                strict=True)[0]

        # dimshuffle back to (n_batch, n_time_steps, n_features))

        #a_out - (n_decodesteps, bs, seqlen)
        #hid_out -   (n_decode_steps, bs, num_units)


        # mask:  (BS, encode_seqlen
        # a_out; (n_decodesteps, BS, encode_seqlen)
        cell_out = cell_out.dimshuffle(1, 0, 2)
        mask = mask.dimshuffle(0, 'x', 1)
        a_out = a_out.dimshuffle(1, 0, 2)  # (BS, n_decodesteps, encode_seqlen)

        # set masked positions to large negative value
        a_out = a_out*mask - (1-mask)*10000

        # normalize over encode_seqlen (->large negative values = 0)
        a_out = T.reshape(a_out, (num_batch*self.n_decodesteps, encode_seqlen))
        alpha = T.nnet.softmax(a_out)
        alpha = T.reshape(alpha, (num_batch, self.n_decodesteps, encode_seqlen))

        # (BS, encode_seqlen, num_units) -> (BS, num_units, 1 encode_seqlen,)
        input = input.dimshuffle(0, 2, 'x',  1)
        # (BS, n_decodesteps, encode_seqlen) -> (BS, '1', n_decodesteps, encode_seqlen)
        alpha = alpha.dimshuffle(0, 'x', 1, 2)
        weighted_hidden_out = input*alpha

        weighted_hidden_out = T.sum(weighted_hidden_out, axis=3)
        # (BS, n_decodesteps, num_encode_units)

        # if scan is backward reverse the output
        if self.backwards:
            hid_out = hid_out[:, ::-1]
            cell_out = cell_out[:, ::-1]
            weighted_hidden_out = weighted_hidden_out[:, ::-1]
            alpha = alpha[:, ::-1]

        self.hid_out = hid_out
        self.cell_out = cell_out
        self.weighted_hidden_out = weighted_hidden_out
        self.alpha = alpha

        return self.weighted_hidden_out


class LSTMAttentionDecodeFeedbackLayer(MergeLayer):
    r"""A long short-term memory (LSTM) layer.

    Includes optional "peephole connections" and a forget gate.  Based on the
    definition in [1]_, which is the current common definition.  The output is
    computed by

    .. math ::

        i_t &= \sigma_i(W_{xi}x_t + W_{hi}h_{t-1}
               + w_{ci}\odot c_{t-1} + b_i)\\
        f_t &= \sigma_f(W_{xf}x_t + W_{hf}h_{t-1}
               + w_{cf}\odot c_{t-1} + b_f)\\
        c_t &= f_t \odot c_{t - 1}
               + i_t\sigma_c(W_{xc}x_t + W_{hc} h_{t-1} + b_c)\\
        o_t &= \sigma_o(W_{xo}x_t + W_{ho}h_{t-1} + w_{co}\odot c_t + b_o)\\
        h_t &= o_t \odot \sigma_h(c_t)

    Parameters
    ----------
    incoming : a :class:`lasagne.layers.Layer` instance or a tuple
        The layer feeding into this layer, or the expected input shape.
    num_units : int
        Number of hidden/cell units in the layer.
    W_in_to_ingate : Theano shared variable, numpy array or callable
        Initializer for input-to-input gate weight matrix (:math:`W_{xi}`).
    W_hid_to_ingate : Theano shared variable, numpy array or callable
        Initializer for hidden-to-input gate weight matrix (:math:`W_{hi}`).
    W_cell_to_ingate : Theano shared variable, numpy array or callable
        Initializer for cell-to-input gate weight vector (:math:`w_{ci}`).
    b_ingate : Theano shared variable, numpy array or callable
        Initializer for input gate bias vector (:math:`b_i`).
    nonlinearity_ingate : callable or None
        The nonlinearity that is applied to the input gate activation
        (:math:`\sigma_i`). If None is provided, no nonlinearity will be
        applied.
    W_in_to_forgetgate : Theano shared variable, numpy array or callable
        Initializer for input-to-forget gate weight matrix (:math:`W_{xf}`).
    W_hid_to_forgetgate : Theano shared variable, numpy array or callable
        Initializer for hidden-to-forget gate weight matrix (:math:`W_{hf}`).
    W_cell_to_forgetgate : Theano shared variable, numpy array or callable
        Initializer for cell-to-forget gate weight vector (:math:`w_{cf}`).
    b_forgetgate : Theano shared variable, numpy array or callable
        Initializer for forget gate bias vector (:math:`b_f`).
    nonlinearity_forgetgate : callable or None
        The nonlinearity that is applied to the forget gate activation
        (:math:`\sigma_f`). If None is provided, no nonlinearity will be
        applied.
    W_in_to_cell : Theano shared variable, numpy array or callable
        Initializer for input-to-cell weight matrix (:math:`W_{ic}`).
    W_hid_to_cell : Theano shared variable, numpy array or callable
        Initializer for hidden-to-cell weight matrix (:math:`W_{hc}`).
    b_cell : Theano shared variable, numpy array or callable
        Initializer for cell bias vector (:math:`b_c`).
    nonlinearity_cell : callable or None
        The nonlinearity that is applied to the cell activation
        (;math:`\sigma_c`). If None is provided, no nonlinearity will be
        applied.
    W_in_to_outgate : Theano shared variable, numpy array or callable
        Initializer for input-to-output gate weight matrix (:math:`W_{io}`).
    W_hid_to_outgate : Theano shared variable, numpy array or callable
        Initializer for hidden-to-output gate weight matrix (:math:`W_{ho}`).
    W_cell_to_outgate : Theano shared variable, numpy array or callable
        Initializer for cell-to-output gate weight vector (:math:`w_{co}`).
    b_outgate : Theano shared variable, numpy array or callable
        Initializer for hidden-to-input gate weight matrix (:math:`b_o`).
    nonlinearity_outgate : callable or None
        The nonlinearity that is applied to the output gate activation
        (:math:`\sigma_o`). If None is provided, no nonlinearity will be
        applied.
    nonlinearity_out : callable or None
        The nonlinearity that is applied to the output (:math:`\sigma_h`). If
        None is provided, no nonlinearity will be applied.
    cell_init : callable, np.ndarray, theano.shared or TensorVariable
        Passing in a TensorVariable allows the user to specify
        the value of `cell_init` (:math:`c_0`). In this mode `learn_init` is
        ignored for the cell state.
    hid_init : callable, np.ndarray, theano.shared or TensorVariable
        Passing in a TensorVariable allows the user to specify
        the value of `hid_init` (:math:`h_0`). In this mode `learn_init` is
        ignored for the hidden state.
    backwards : bool
        If True, process the sequence backwards and then reverse the
        output again such that the output from the layer is always
        from :math:`x_1` to :math:`x_n`.
    learn_init : bool
        If True, initial hidden values are learned. If `hid_init` or
        `cell_init` are TensorVariables then the TensorVariable is used and
        `learn_init` is ignored for that initial state.
    peepholes : bool
        If True, the LSTM uses peephole connections.
        When False, `W_cell_to_ingate`, `W_cell_to_forgetgate` and
        `W_cell_to_outgate` are ignored.
    gradient_steps : int
        Number of timesteps to include in the backpropagated gradient.
        If -1, backpropagate through the entire sequence.
    grad_clipping: False or float
        If a float is provided, the gradient messages are clipped during the
        backward pass.  If False, the gradients will not be clipped.  See [1]_
        (p. 6) for further explanation.
    unroll_scan : bool
        If True the recursion is unrolled instead of using scan. For some
        graphs this gives a significant speed up but it might also consume
        more memory. When `unroll_scan` is true then the `gradient_steps`
        setting is ignored.
    precompute_input : bool
        If True, precompute input_to_hid before iterating through
        the sequence. This can result in a speedup at the expense of
        an increase in memory usage.

    References
    ----------
    .. [1] Graves, Alex: "Generating sequences with recurrent neural networks."
           arXiv preprint arXiv:1308.0850 (2013).
    """
    def __init__(self, incoming,
                 num_units,
                 aln_num_units,
                 n_decodesteps,
                 W_align=init.Normal(0.1),
                 U_align=init.Normal(0.1),
                 v_align=init.Normal(0.1),
                 U_conv_align=init.Normal(0.1),
                 nonlinearity_align=nonlinearities.tanh,
                 W_hid_to_ingate=init.Normal(0.1),
                 W_cell_to_ingate=init.Normal(0.1),
                 b_ingate=init.Constant(0.),
                 nonlinearity_ingate=nonlinearities.sigmoid,
                 #W_in_to_forgetgate=init.Normal(0.1),
                 W_hid_to_forgetgate=init.Normal(0.1),
                 W_cell_to_forgetgate=init.Normal(0.1),
                 b_forgetgate=init.Constant(0.),
                 nonlinearity_forgetgate=nonlinearities.sigmoid,
                 #W_in_to_cell=init.Normal(0.1),
                 W_hid_to_cell=init.Normal(0.1),
                 b_cell=init.Constant(0.),
                 nonlinearity_cell=nonlinearities.tanh,
                 #W_in_to_outgate=init.Normal(0.1),
                 W_hid_to_outgate=init.Normal(0.1),
                 W_cell_to_outgate=init.Normal(0.1),
                 b_outgate=init.Constant(0.),
                 nonlinearity_outgate=nonlinearities.sigmoid,
                 nonlinearity_out=nonlinearities.tanh,
                 cell_init=init.Constant(0.),
                 hid_init=init.Constant(0.),
                 backwards=False,
                 learn_init=False,
                 peepholes=True,
                 gradient_steps=-1,
                 grad_clipping=False,
                 unroll_scan=False,
                 attention_softmax_function=T.nnet.softmax,
                 #precompute_input=True,
                 decode_pre_steps=0,
                 return_decodehid=False,
                 mask_input=None,
                 **kwargs):

        # Initialize parent layer
        incomings = [incoming]
        if mask_input is not None:
            incomings.append(mask_input)
        super(LSTMAttentionDecodeFeedbackLayer, self).__init__(
            incomings, **kwargs)

        # For any of the nonlinearities, if None is supplied, use identity
        if nonlinearity_ingate is None:
            self.nonlinearity_ingate = nonlinearities.identity
        else:
            self.nonlinearity_ingate = nonlinearity_ingate

        if nonlinearity_forgetgate is None:
            self.nonlinearity_forgetgate = nonlinearities.identity
        else:
            self.nonlinearity_forgetgate = nonlinearity_forgetgate

        if nonlinearity_cell is None:
            self.nonlinearity_cell = nonlinearities.identity
        else:
            self.nonlinearity_cell = nonlinearity_cell

        if nonlinearity_outgate is None:
            self.nonlinearity_outgate = nonlinearities.identity
        else:
            self.nonlinearity_outgate = nonlinearity_outgate

        if nonlinearity_out is None:
            self.nonlinearity_out = nonlinearities.identity
        else:
            self.nonlinearity_out = nonlinearity_out

        self.attention_softmax_function = attention_softmax_function

        self.learn_init = learn_init
        self.num_units = num_units
        self.backwards = backwards
        self.peepholes = peepholes
        self.gradient_steps = gradient_steps
        self.grad_clipping = grad_clipping
        self.unroll_scan = unroll_scan
        self.n_decodesteps = n_decodesteps
        self.aln_num_units = aln_num_units
        self.nonlinearity_align = nonlinearity_align
        self.decode_pre_steps = decode_pre_steps
        self.return_decodehid = return_decodehid

        input_shape = self.input_shapes[0]
        if unroll_scan and input_shape[1] is None:
            raise ValueError("Input sequence length cannot be specified as "
                             "None when unroll_scan is True")

        num_inputs = np.prod(input_shape[2:])
        self.num_inputs = num_inputs
        # Initialize parameters using the supplied args
        #self.W_in_to_ingate = self.add_param(
        #    W_in_to_ingate, (num_inputs, num_units), name="W_in_to_ingate")

        self.W_hid_to_ingate = self.add_param(
            W_hid_to_ingate, (num_units, num_units), name="W_hid_to_ingate")

        self.b_ingate = self.add_param(
            b_ingate, (num_units,), name="b_ingate", regularizable=False)

        #self.W_in_to_forgetgate = self.add_param(
        #    W_in_to_forgetgate, (num_inputs, num_units),
        #    name="W_in_to_forgetgate")

        self.W_hid_to_forgetgate = self.add_param(
            W_hid_to_forgetgate, (num_units, num_units),
            name="W_hid_to_forgetgate")

        self.b_forgetgate = self.add_param(
            b_forgetgate, (num_units,), name="b_forgetgate",
            regularizable=False)

        #self.W_in_to_cell = self.add_param(
        #    W_in_to_cell, (num_inputs, num_units), name="W_in_to_cell")

        self.W_hid_to_cell = self.add_param(
            W_hid_to_cell, (num_units, num_units), name="W_hid_to_cell")

        self.b_cell = self.add_param(
            b_cell, (num_units,), name="b_cell", regularizable=False)

        #self.W_in_to_outgate = self.add_param(
        #    W_in_to_outgate, (num_inputs, num_units), name="W_in_to_outgate")

        self.W_hid_to_outgate = self.add_param(
            W_hid_to_outgate, (num_units, num_units), name="W_hid_to_outgate")

        self.b_outgate = self.add_param(
            b_outgate, (num_units,), name="b_outgate", regularizable=False)


        self.W_weightedhid_to_ingate = self.add_param(
            W_hid_to_ingate, (num_inputs, num_units), name="W_weightedhid_to_ingate")

        self.W_weightedhid_to_forgetgate = self.add_param(
            W_hid_to_forgetgate, (num_inputs, num_units),
            name="W_weightedhid_to_forgetgate")

        self.W_weightedhid_to_cell = self.add_param(
            W_hid_to_cell, (num_inputs, num_units), name="W_weightedhid_to_cell")

        self.W_weightedhid_to_outgate = self.add_param(
            W_hid_to_outgate, (num_inputs, num_units), name="W_weightedhid_to_outgate")




        # Stack input weight matrices into a (num_inputs, 4*num_units)
        # matrix, which speeds up computation
        #self.W_in_stacked = T.concatenate(
        #    [self.W_in_to_ingate, self.W_in_to_forgetgate,
        #     self.W_in_to_cell, self.W_in_to_outgate], axis=1)

        # Same for hidden weight matrices
        self.W_hid_stacked = T.concatenate(
            [self.W_hid_to_ingate, self.W_hid_to_forgetgate,
             self.W_hid_to_cell, self.W_hid_to_outgate], axis=1)

        self.W_weightedhid_stacked = T.concatenate(
            [self.W_weightedhid_to_ingate, self.W_weightedhid_to_forgetgate,
             self.W_weightedhid_to_cell, self.W_weightedhid_to_outgate], axis=1)

        # Stack biases into a (4*num_units) vector
        self.b_stacked = T.concatenate(
            [self.b_ingate, self.b_forgetgate,
             self.b_cell, self.b_outgate], axis=0)

        # If peephole (cell to gate) connections were enabled, initialize
        # peephole connections.  These are elementwise products with the cell
        # state, so they are represented as vectors.
        if self.peepholes:
            self.W_cell_to_ingate = self.add_param(
                W_cell_to_ingate, (num_units, ), name="W_cell_to_ingate")

            self.W_cell_to_forgetgate = self.add_param(
                W_cell_to_forgetgate, (num_units, ),
                name="W_cell_to_forgetgate")

            self.W_cell_to_outgate = self.add_param(
                W_cell_to_outgate, (num_units, ), name="W_cell_to_outgate")

        self.W_align = self.add_param(W_align, (num_units, self.aln_num_units),
                                   name="AlignSeqOutputLayer: (aln) W_a")
        self.U_align = self.add_param(U_align, (num_inputs, self.aln_num_units),
                           name="AlignSeqOutputLayer: (aln) U_a")
        self.v_align = self.add_param(v_align, (self.aln_num_units, 1),
                                 name="AlignSeqOutputLayer: v_a")


        # Setup initial values for the cell and the hidden units
        if isinstance(cell_init, T.TensorVariable):
            if cell_init.ndim != 2:
                raise ValueError(
                    "When cell_init is provided as a TensorVariable, it should"
                    " have 2 dimensions and have shape (num_batch, num_units)")
            self.cell_init = cell_init
        else:
            self.cell_init = self.add_param(
                cell_init, (1, num_units), name="cell_init",
                trainable=learn_init, regularizable=False)

        if isinstance(hid_init, T.TensorVariable):
            if hid_init.ndim != 2:
                raise ValueError(
                    "When hid_init is provided as a TensorVariable, it should "
                    "have 2 dimensions and have shape (num_batch, num_units)")
            self.hid_init = hid_init
        else:
            self.hid_init = self.add_param(
                hid_init, (1, self.num_units), name="hid_init",
                trainable=learn_init, regularizable=False)

    def get_output_shape_for(self, input_shapes):
        input_shape = input_shapes[0]
        return input_shape[0], None, self.num_units

    def get_params(self, **tags):
        # Get all parameters from this layer, the master layer
        params = super(LSTMAttentionDecodeFeedbackLayer, self).get_params(**tags)
        # Combine with all parameters from the child layers
        return params

    def get_output_for(self, inputs, **kwargs):
        """
        Compute this layer's output function given a symbolic input variable

        Parameters
        ----------
        input : theano.TensorType
            Symbolic input variable.
        mask : theano.TensorType
            Theano variable denoting whether each time step in each
            sequence in the batch is part of the sequence or not.  If ``None``,
            then it is assumed that all sequences are of the same length.  If
            not all sequences are of the same length, then it must be
            supplied as a matrix of shape ``(n_batch, n_time_steps)`` where
            ``mask[i, j] = 1`` when ``j <= (length of sequence i)`` and
            ``mask[i, j] = 0`` when ``j > (length of sequence i)``.

        Returns
        -------
        layer_output : theano.TensorType
            Symblic output variable.
        """
        input = inputs[0]
        # Retrieve the mask when it is supplied
        mask = inputs[1] if len(inputs) > 1 else None

        # Treat all dimensions after the second as flattened feature dimensions
        if input.ndim > 3:
            input = input.reshape((input.shape[0], input.shape[1],
                                   T.prod(input.shape[2:])))
        num_batch = input.shape[0]
        encode_seqlen = input.shape[1]

        if mask is None:
            mask = T.ones((num_batch, encode_seqlen),dtype='float32')
        # At each call to scan, input_n will be (n_time_steps, 4*num_units).
        # We define a slicing function that extract the input to each LSTM gate
        def slice_w(x, n):
            return x[:, n*self.num_units:(n+1)*self.num_units]

        # Create single recurrent computation step function
        # input_n is the n'th vector of the input
        def step(cell_previous, hid_previous, alpha_prev, weighted_hidden_prev,
                 input, mask, hUa, W_align, v_align,
                 W_hid_stacked, W_weightedhid_stacked, W_cell_to_ingate,
                 W_cell_to_forgetgate, W_cell_to_outgate,
                 b_stacked, *args):

            #compute (unormalized) attetion vector
            sWa = T.dot(hid_previous, W_align)       # (BS, aln_num_units)
            sWa = sWa.dimshuffle(0, 'x', 1)   # (BS, 1, aln_num_units)
            align_act = sWa + hUa
            tanh_sWahUa = self.nonlinearity_align(align_act)
                                            # (BS, seqlen, num_units_aln)

            # CALCULATE WEIGHT FOR EACH HIDDEN STATE VECTOR
            a = T.dot(tanh_sWahUa, v_align)  # (BS, Seqlen, 1)
            a = T.reshape(a, (a.shape[0], a.shape[1]))
            #                                # (BS, Seqlen)
            # # ->(BS, seq_len)

            a = a*mask - (1-mask)*10000

            alpha = self.attention_softmax_function(a)
            #alpha = T.reshape(alpha, (input.shape[0], input.shape[1]))

            # input: (BS, Seqlen, num_units)
            weighted_hidden = input * alpha.dimshuffle(0, 1, 'x')
            weighted_hidden = T.sum(weighted_hidden, axis=1)  #sum seqlen out


            # Calculate gates pre-activations and slice

            # (BS, dec_hid) x (dec_hid, dec_hid)
            gates = T.dot(hid_previous, W_hid_stacked) + b_stacked
            # (BS, enc_hid) x (enc_hid, dec_hid)
            gates += T.dot(weighted_hidden, W_weightedhid_stacked)

            # Clip gradients
            if self.grad_clipping is not False:
                gates = theano.gradient.grad_clip(
                    gates, -self.grad_clipping, self.grad_clipping)

            # Extract the pre-activation gate values
            ingate = slice_w(gates, 0)
            forgetgate = slice_w(gates, 1)
            cell_input = slice_w(gates, 2)
            outgate = slice_w(gates, 3)

            if self.peepholes:
                # Compute peephole connections
                ingate += cell_previous*W_cell_to_ingate
                forgetgate += cell_previous*W_cell_to_forgetgate

            # Apply nonlinearities
            ingate = self.nonlinearity_ingate(ingate)
            forgetgate = self.nonlinearity_forgetgate(forgetgate)
            cell_input = self.nonlinearity_cell(cell_input)
            outgate = self.nonlinearity_outgate(outgate)

            # Compute new cell value
            cell = forgetgate*cell_previous + ingate*cell_input

            if self.peepholes:
                outgate += cell*W_cell_to_outgate

            # W_align:  (num_units, aln_num_units)
            # U_align:  (num_feats, aln_num_units)
            # v_align:  (aln_num_units, 1)
            # hUa:      (BS, Seqlen, aln_num_units)
            # hid:      (BS, num_units_dec)
            # input:    (BS, Seqlen, num_inputs)

            # Compute new hidden unit activation
            hid = outgate*self.nonlinearity_out(cell)

            return [cell, hid, alpha, weighted_hidden]

        sequences = []
        step_fun = step

        ones = T.ones((num_batch, 1))
        if isinstance(self.cell_init, T.TensorVariable):
            cell_init = self.cell_init
        else:
            # Dot against a 1s vector to repeat to shape (num_batch, num_units)
            cell_init = T.dot(ones, self.cell_init)

        if isinstance(self.hid_init, T.TensorVariable):
            hid_init = self.hid_init
        else:
            # Dot against a 1s vector to repeat to shape (num_batch, num_units)
            hid_init = T.dot(ones, self.hid_init)

        #weighted_hidden_init = T.zeros((num_batch, input.shape[2]))
        alpha_init = T.zeros((num_batch, encode_seqlen))

        weighted_hidden_init = T.zeros((num_batch, self.num_inputs))

        # The hidden-to-hidden weight matrix is always used in step

        hUa = T.dot(input, self.U_align)   # (num_batch, seq_len, num_units_aln)

        non_seqs = [input, mask, hUa, self.W_align, self.v_align,
                    self.W_hid_stacked, self.W_weightedhid_stacked]
        # The "peephole" weight matrices are only used when self.peepholes=True
        if self.peepholes:
            non_seqs += [self.W_cell_to_ingate,
                         self.W_cell_to_forgetgate,
                         self.W_cell_to_outgate]
        # theano.scan only allows for positional arguments, so when
        # self.peepholes is False, we need to supply fake placeholder arguments
        # for the three peephole matrices.
        else:
            non_seqs += [(), (), ()]
        # When we aren't precomputing the input outside of scan, we need to
        # provide the input weights and biases to the step function

        non_seqs += [self.b_stacked]

        if self.unroll_scan:
            # Explicitly unroll the recurrence instead of using scan
            cell_out, hid_out, alpha_out, weighted_hidden_out = unroll_scan(
                fn=step_fun,
                sequences=sequences,
                outputs_info=[cell_init, hid_init, alpha_init, weighted_hidden_init],
                go_backwards=self.backwards,
                non_sequences=non_seqs,
                n_steps=self.n_decodesteps + self.decode_pre_steps)
        else:
            # Scan op iterates over first dimension of input and repeatedly
            # applies the step function
            cell_out, hid_out, alpha_out, weighted_hidden_out = theano.scan(
                fn=step_fun,
                sequences=sequences,
                outputs_info=[cell_init, hid_init, alpha_init, weighted_hidden_init],
                go_backwards=self.backwards,
                truncate_gradient=self.gradient_steps,
                non_sequences=non_seqs,
                n_steps=self.n_decodesteps + self.decode_pre_steps,
                strict=True)[0]

        # dimshuffle back to (n_batch, n_time_steps, n_features))

        #a_out - (n_decodesteps, bs, seqlen)
        #hid_out -   (n_decode_steps, bs, num_units)


        # mask:  (BS, encode_seqlen
        # a_out; (n_decodesteps, BS, encode_seqlen)
        cell_out = cell_out.dimshuffle(1, 0, 2)
        hid_out = hid_out.dimshuffle(1, 0, 2)  # (BS, n_decodesteps, encode_seqlen)
        mask = mask.dimshuffle(0, 'x', 1)
        alpha_out = alpha_out.dimshuffle(1, 0, 2)  # (BS, n_decodesteps, encode_seqlen)

        weighted_hidden_out = weighted_hidden_out.dimshuffle(1, 0, 2)

        # if scan is backward reverse the output
        if self.backwards:
            hid_out = hid_out[:, ::-1]
            cell_out = cell_out[:, ::-1]
            weighted_hidden_out = weighted_hidden_out[:, ::-1]
            alpha_out = alpha_out[:, ::-1]

        if self.decode_pre_steps > 0:
            hid_out = hid_out[:, self.decode_pre_steps:]
            cell_out = hid_out[:, self.decode_pre_steps:]
            weighted_hidden_out = weighted_hidden_out[:, self.decode_pre_steps:]
            alpha_out = hid_out[:, self.decode_pre_steps:]

        self.hid_out = hid_out
        self.cell_out = cell_out
        self.weighted_hidden_out = weighted_hidden_out
        self.alpha = alpha_out

        if self.return_decodehid:
            return hid_out
        else:
            return weighted_hidden_out


