from __future__ import print_function
import numpy as np

target_to_text = {
    '0':'zero',
    '1':'one',
    '2':'two',
    '3':'three',
    '4':'four',
    '5':'five',
    '6':'six',
    '7':'seven',
    '8':'eight',
    '9':'nine',
}

stop_character = start_character = '#'

input_characters = " ".join(target_to_text.values())
valid_characters = ['0', '1', '2', '3',  '4',  '5',  '6',  '7',  '8',  '9',  '#'] + \
              list(set(input_characters))

def print_valid_characters():
    l = ''
    for i,c in enumerate(valid_characters):
        l += "\'%s\'=%i,\t" % (c,i)
    print("Number of valid characters:", len(valid_characters))
    print(l)

ninput_chars = len(valid_characters)
def get_batch(batch_size=100, min_digits = 3, max_digits=3):
    '''
    Generates random sequences of integers and translates them to text i.e. 1->'one'.
    :param batch_size: number of samples to return
    :param min_digits: minimum length of target
    :param max_digits: maximum length of target
    '''
    text_inputs = []
    int_inputs = []
    text_targets_in = []
    text_targets_out = []
    int_targets_in = []
    int_targets_out = []
    for i in range(batch_size):
        #convert integer into a list of digits
        tar_len = np.random.randint(min_digits,max_digits+1)
        text_target = inp_str = "".join(map(str,np.random.randint(0,10,tar_len)))
        text_target_in = start_character + text_target
        text_target_out = text_target + stop_character

        #generate the targets as a list of intergers
        int_target_in = map(lambda c: valid_characters.index(c), text_target_in)
        int_target_out = map(lambda c: valid_characters.index(c), text_target_out)

        #generate the text input
        text_input = " ".join(map(lambda k: target_to_text[k], inp_str))
        #generate the inputs as a list of intergers
        int_input = map(lambda c: valid_characters.index(c), text_input)

        text_inputs.append(text_input)
        int_inputs.append(int_input)
        text_targets_in.append(text_target_in)
        text_targets_out.append(text_target_out)
        int_targets_in.append(int_target_in)
        int_targets_out.append(int_target_out)

    #create the input matrix, mask and seq_len - note that we zero pad the shorter sequences.
    max_input_len = max(map(len, int_inputs))
    inputs = np.zeros((batch_size, max_input_len))
#    input_masks = np.zeros((batch_size,max_input_len))
    for (i,inp) in enumerate(int_inputs):
        cur_len = len(inp)
        inputs[i,:cur_len] = inp
#        input_masks[i,:cur_len] = 1
    inputs_seqlen = np.asarray(map(len, int_inputs))

    max_target_in_len = max(map(len, int_targets_in))
    targets_in = np.zeros((batch_size, max_target_in_len))
    targets_mask = np.zeros((batch_size, max_target_in_len))
    for (i, tar) in enumerate(int_targets_in):
        cur_len = len(tar)
        targets_in[i, :cur_len] = tar
    targets_seqlen = np.asarray(map(len, int_targets_in))

    max_target_out_len = max(map(len, int_targets_out))
    targets_out = np.zeros((batch_size, max_target_in_len))
    for (i,tar) in enumerate(int_targets_out):
        cur_len = len(tar)
        targets_out[i,:cur_len] = tar
        targets_mask[i,:cur_len] = 1

    return inputs.astype('int32'), \
           inputs_seqlen.astype('int32'), \
           targets_in.astype('int32'), \
           targets_out.astype('int32'), \
           targets_seqlen.astype('int32'), \
           targets_mask.astype('float32'), \
           text_inputs, \
           text_targets_in, \
           text_targets_out

if __name__ == '__main__':
    batch_size = 3
    inputs, inputs_seqlen, targets_in, targets_out, targets_seqlen, targets_mask, \
    text_inputs, text_targets_in, text_targets_out = \
        get_batch(batch_size=batch_size, max_digits=2, min_digits=1)

    print("input types:", inputs.dtype, inputs_seqlen.dtype, targets_in.dtype, targets_out.dtype, targets_seqlen.dtype)
    print(print_valid_characters())
    print("Stop/start character = #")

    for i in range(batch_size):
        print("\nSAMPLE",i)
        print("TEXT INPUTS:\t\t\t", text_inputs[i])
        print("TEXT TARGETS INPUT:\t\t", text_targets_in[i])
        print("TEXT TARGETS OUTPUT:\t\t", text_targets_out[i])
        print("ENCODED INPUTS:\t\t\t", inputs[i])
        print("INPUTS SEQUENCE LENGTH:\t\t", inputs_seqlen[i])
        print("ENCODED TARGETS INPUT:\t\t", targets_in[i])
        print("ENCODED TARGETS OUTPUT:\t\t", targets_out[i])
        print("TARGETS SEQUENCE LENGTH:\t", targets_seqlen[i])
        print("TARGETS MASK:\t\t\t", targets_mask[i])