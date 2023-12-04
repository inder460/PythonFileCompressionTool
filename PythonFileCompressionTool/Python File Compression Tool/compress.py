
from __future__ import annotations

import time

from huffman import HuffmanTree
from utils import *


# ====================
# Functions for compression


def build_frequency_dict(text: bytes) -> dict[int, int]:
    """ Return a dictionary which maps each of the bytes in <text> to its
    frequency.

    >>> d = build_frequency_dict(bytes([65, 66, 67, 66]))
    >>> d == {65: 1, 66: 2, 67: 1}
    True
    >>> d = build_frequency_dict(bytes([65, 66, 67, 66, 65, 68, 68, 65]))
    >>> d == {65: 3, 66: 2, 67: 1, 68: 2}
    True
    >>> d = build_frequency_dict(bytes([]))
    >>> d == {}
    True
    """
    # https://brilliant.org/wiki/huffman-encoding/#:~:
    # text=Huffman%20coding%20uses%20a%20greedy,string%20representing%20any%20
    # other%20symbol.

    # declare empty dictionary to return what is asked
    byte_dict = {}
    # counter for while loop
    i = 0
    # loop over the length of text
    while i < len(text):
        # get the byte at i
        byte = text[i]
        # add the byte in the dictionary and initialize at 1
        if byte not in byte_dict:
            byte_dict[byte] = 1
        # else add 1 to the counter of the existing byte
        else:
            byte_dict[byte] += 1
        # add 1 to index
        i += 1
    # return dictionary
    return byte_dict


def build_huffman_tree(freq_dict: dict[int, int]) -> HuffmanTree:
    """ Return the Huffman tree corresponding to the frequency dictionary
    <freq_dict>.

    Precondition: freq_dict is not empty.

    >>> freq = {2: 6, 3: 4}
    >>> t = build_huffman_tree(freq)
    >>> result = HuffmanTree(None, HuffmanTree(3), HuffmanTree(2))
    >>> t == result
    True
    >>> freq = {2: 6, 3: 4, 7: 5}
    >>> t = build_huffman_tree(freq)
    >>> result = HuffmanTree(None, HuffmanTree(2), \
                             HuffmanTree(None, HuffmanTree(3), HuffmanTree(7)))
    >>> t == result
    True
    >>> import random
    >>> symbol = random.randint(0,255)
    >>> freq = {symbol: 6}
    >>> t = build_huffman_tree(freq)
    >>> any_valid_byte_other_than_symbol = (symbol + 1) % 256
    >>> dummy_tree = HuffmanTree(any_valid_byte_other_than_symbol)
    >>> result = HuffmanTree(None, HuffmanTree(symbol), dummy_tree)
    >>> t.left == result.left or t.right == result.left
    True
    """
    # make list of huffman trees for evert symbol in dict
    trees = [HuffmanTree(symbol) for symbol in freq_dict]
    # sort tress in list based on frequencies of the symbols
    # using the dict
    trees.sort(key=lambda tree: freq_dict[tree.symbol])
    # https://realpython.com/python-lambda/

    # combine two trees with the lowest frequencies repeatedly
    # until only one tree left
    while len(trees) > 1:
        left, right = trees.pop(0), trees.pop(0)
        freq_sum = freq_dict[left.symbol] + freq_dict[right.symbol]
        parent = HuffmanTree(None, left, right)
        trees.append(parent)
        freq_dict[parent.symbol] = freq_sum
        trees.sort(key=lambda tree: freq_dict[tree.symbol])
        # https://realpython.com/python-lambda/

    # return root node of the only tree in list
    return trees[0]


def get_codes(tree: HuffmanTree) -> dict[int, str]:
    """ Return a dictionary which maps symbols from the Huffman tree <tree>
    to codes.

    >>> tree = HuffmanTree(None, HuffmanTree(3), HuffmanTree(2))
    >>> d = get_codes(tree)
    >>> d == {3: "0", 2: "1"}
    True
    >>> tree = HuffmanTree(None, HuffmanTree(12), HuffmanTree(34))
    >>> d = get_codes(tree)
    >>> d == {12: "0", 34: "1"}
    True
    """
    # declare dictionary for returning
    huffman_codes = {}
    # initialize a stack with symbol and empty string
    stack = [(tree, "")]
    # go through every item in the tree
    while stack:
        # remove the symbol and code
        symbol, code = stack.pop()
        # if the symbol is a leaf in the tree
        # add the symbol and code to huffman_codes
        if symbol.is_leaf():
            huffman_codes[symbol.symbol] = code
        # if the symbol is not a leaf put its children
        # onto the stack with codes
        # left children get 0 and right get 1
        else:
            if symbol.right:
                stack.append((symbol.right, code + "1"))
            if symbol.left:
                stack.append((symbol.left, code + "0"))
    # return dictionary
    return huffman_codes


def number_nodes(tree: HuffmanTree) -> None:
    """ Number internal nodes in <tree> according to postorder traversal. The
    numbering starts at 0.

    >>> left = HuffmanTree(None, HuffmanTree(3), HuffmanTree(2))
    >>> right = HuffmanTree(None, HuffmanTree(9), HuffmanTree(10))
    >>> tree = HuffmanTree(None, left, right)
    >>> number_nodes(tree)
    >>> tree.left.number
    0
    >>> tree.right.number
    1
    >>> tree.number
    2
    """
    # declare some variables
    stack = [tree]
    parent = None
    count = 0
    # loop through the whole tree
    while stack:
        node = stack[-1]

        # check if node is parents or left or right child of parent
        if not parent or parent.left == node or parent.right == node:
            # if the node has left child, traverse left
            if node.left:
                stack.append(node.left)
            # if the node has right child, traverse right
            elif node.right:
                stack.append(node.right)

        # if parent the left child and node has a right child, traverse right
        elif node.left == parent:
            if node.right:
                stack.append(node.right)

        # else the node has both left and right children
        else:
            if node.left and node.right:
                # assign a number to the node
                node.number = count
                # add 1 to count
                count += 1
            # pop from stack since it has been accounted for
            stack.pop()

        # Update the parent node to the current node
        parent = node


def avg_length(tree: HuffmanTree, freq_dict: dict[int, int]) -> float:
    """ Return the average number of bits required per symbol, to compress the
    text made of the symbols and frequencies in <freq_dict>, using the Huffman
    tree <tree>.

    The average number of bits = the weighted sum of the length of each symbol
    (where the weights are given by the symbol's frequencies), divided by the
    total of all symbol frequencies.

    >>> freq = {3: 2, 2: 7, 9: 1}
    >>> left = HuffmanTree(None, HuffmanTree(3), HuffmanTree(2))
    >>> right = HuffmanTree(9)
    >>> tree = HuffmanTree(None, left, right)
    >>> avg_length(tree, freq)  # (2*2 + 7*2 + 1*1) / (2 + 7 + 1)
    1.9
    """
    # call get_codes and get the codes for given tree
    tree_codes = get_codes(tree)
    # loop through tree_codes and multiply each frequency with the
    # length of each symbol
    symbols_length = [len(tree_codes[avg_time]) * freq_dict[avg_time]
                      for avg_time in tree_codes]
    # get the sum of both symbols_length and sum of the frequency
    # and return the answer and make sure it is a float
    return float(sum(symbols_length) / sum(freq_dict.values()))


def compress_bytes(text: bytes, codes: dict[int, str]) -> bytes:
    """ Return the compressed form of <text>, using the mapping from <codes>
    for each symbol.

    >>> d = {0: "0", 1: "10", 2: "11"}
    >>> text = bytes([1, 2, 1, 0])
    >>> result = compress_bytes(text, d)
    >>> result == bytes([184])
    True
    >>> [byte_to_bits(byte) for byte in result]
    ['10111000']
    >>> text = bytes([1, 2, 1, 0, 2])
    >>> result = compress_bytes(text, d)
    >>> [byte_to_bits(byte) for byte in result]
    ['10111001', '10000000']
    """
    # declare list to store compressed bytes
    compressed_bytes = []

    # initialize current byte and bit index
    current_byte = 0
    current_bit_index = 0

    # loop through every symbol in input
    for symbol in text:
        # get code for symbol
        code = codes[symbol]

        # loop through every bit
        for bit in code:
            # if bit is 1
            if bit == '1':
                # make corresponding bit in current byte to 1
                current_byte |= 1 << (7 - current_bit_index)

            # add the current bit index
            current_bit_index += 1

            # if byte is full
            if current_bit_index == 8:
                # append to list of compressed bytes
                compressed_bytes.append(current_byte)
                # reset the byte and index to 0
                current_byte = 0
                current_bit_index = 0

    # if final byte not full
    if current_bit_index > 0:
        # append to list of compressed bytes
        compressed_bytes.append(current_byte)

    # return compressed bytes but convert to bytes since that is return type
    return bytes(compressed_bytes)


def tree_to_bytes(tree: HuffmanTree) -> bytes:
    """ Return a bytes representation of the Huffman tree <tree>.
    The representation should be based on the postorder traversal of the tree's
    internal nodes, starting from 0.

    Precondition: <tree> has its nodes numbered.

    >>> tree = HuffmanTree(None, HuffmanTree(3, None, None), \
    HuffmanTree(2, None, None))
    >>> number_nodes(tree)
    >>> list(tree_to_bytes(tree))
    [0, 3, 0, 2]
    >>> left = HuffmanTree(None, HuffmanTree(3, None, None), \
    HuffmanTree(2, None, None))
    >>> right = HuffmanTree(5)
    >>> tree = HuffmanTree(None, left, right)
    >>> number_nodes(tree)
    >>> list(tree_to_bytes(tree))
    [0, 3, 0, 2, 1, 0, 0, 5]
    >>> tree = build_huffman_tree(build_frequency_dict(b"helloworld"))
    >>> number_nodes(tree)
    >>> list(tree_to_bytes(tree))\
            #doctest: +NORMALIZE_WHITESPACE
    [0, 104, 0, 101, 0, 119, 0, 114, 1, 0, 1, 1, 0, 100, 0, 111, 0, 108,\
    1, 3, 1, 2, 1, 4]
    """
    # make list for internal nodes
    inter_nodes = []
    # call helper function below to get internal nodes
    internal_nodes_list(tree, inter_nodes)
    # make list for result
    result = []
    # initialize i = 0 for while loop and index for inter_nodes list
    i = 0
    # go through the internal nodes
    while i < len(inter_nodes):
        # set the tree to ith element in inter_nodes list
        tree = inter_nodes[i]
        # if left node of left subtree is a leaf
        if tree.left.left is None and tree.left.right is None:
            # add 0 and symbol to result
            result.extend([0, tree.left.symbol])
        # else the left node of left subtree is not a leaf
        else:
            # add 1 and number to result
            result.extend([1, tree.left.number])
        # if right node of right subtree is a leaf
        if tree.right.left is None and tree.right.right is None:
            # add 0 and symbol to result
            result.extend([0, tree.right.symbol])
        # else the right node of right subtree is not a leaf
        else:
            # add 1 and number to result
            result.extend([1, tree.right.number])
        # add 1 to i to keep up with while loop
        i += 1

    # convert result list to bytes and return
    return bytes(result)


def internal_nodes_list(tree: HuffmanTree, inter_nodes: list) -> None:
    """
    Function to return list of internal nodes.
    Helper function for tress_to_bytes.
    Side Notes/Things To Ask:
    I tried doing these if statements in the function without the helper,
    but I ran into errors, and I am not sure as to why. Maybe since it is
    recursive so the for loop I was trying did not work?
    """
    # if tree.left is a leaf
    if tree.left is not None:
        # recurse through left subtree
        internal_nodes_list(tree.left, inter_nodes)
    # if tree.right is a leaf
    if tree.right is not None:
        # recurse through right subtree
        internal_nodes_list(tree.right, inter_nodes)
    # if the node is an internal node
    if tree.left is not None and tree.right is not None:
        # append it to the list
        inter_nodes.append(tree)


def compress_file(in_file: str, out_file: str) -> None:
    """ Compress contents of the file <in_file> and store results in <out_file>.
    Both <in_file> and <out_file> are string objects representing the names of
    the input and output files.

    Precondition: The contents of the file <in_file> are not empty.
    """
    with open(in_file, "rb") as f1:
        text = f1.read()
    freq = build_frequency_dict(text)
    tree = build_huffman_tree(freq)
    codes = get_codes(tree)
    number_nodes(tree)
    print("Bits per symbol:", avg_length(tree, freq))
    result = (tree.num_nodes_to_bytes() + tree_to_bytes(tree)
              + int32_to_bytes(len(text)))
    result += compress_bytes(text, codes)
    with open(out_file, "wb") as f2:
        f2.write(result)


# ====================
# Functions for decompression

def generate_tree_general(node_lst: list[ReadNode],
                          root_index: int) -> HuffmanTree:
    """ Return the Huffman tree corresponding to node_lst[root_index].
    The function assumes nothing about the order of the tree nodes in the list.

    >>> lst = [ReadNode(0, 5, 0, 7), ReadNode(0, 10, 0, 12), \
    ReadNode(1, 1, 1, 0)]
    >>> generate_tree_general(lst, 2)
    HuffmanTree(None, HuffmanTree(None, HuffmanTree(10, None, None), \
HuffmanTree(12, None, None)), \
HuffmanTree(None, HuffmanTree(5, None, None), HuffmanTree(7, None, None)))
    """
    # variable to get root node from given node list
    root_node = node_lst[root_index]
    # create right subtree
    if root_node.r_type == 0:
        right = HuffmanTree(root_node.r_data)
    # use recursion if root_node.r_type != 0 to create right subtree
    else:
        right = generate_tree_general(node_lst, root_node.r_data)
    # now do same stuff but for left subtree
    # create left subtree
    if root_node.l_type == 0:
        left = HuffmanTree(root_node.l_data)
    # use recursion if root_node.l_type != 0 to create left subtree
    else:
        left = generate_tree_general(node_lst, root_node.l_data)

    # create new_tree to form new HuffmanTree and return it
    new_tree = HuffmanTree(None, left, right)
    return new_tree


def generate_tree_postorder(node_lst: list[ReadNode],
                            root_index: int) -> HuffmanTree:
    """ Return the Huffman tree corresponding to node_lst[root_index].
    The function assumes that the list represents a tree in postorder.

    >>> lst = [ReadNode(0, 5, 0, 7), ReadNode(0, 10, 0, 12), \
    ReadNode(1, 0, 1, 0)]
    >>> generate_tree_postorder(lst, 2)
    HuffmanTree(None, HuffmanTree(None, HuffmanTree(5, None, None), \
HuffmanTree(7, None, None)), \
HuffmanTree(None, HuffmanTree(10, None, None), HuffmanTree(12, None, None)))
    """
    # make new Huffman Tree
    root = HuffmanTree(None, None)
    # get the length of node_lst
    length = len(node_lst)

    # find offset for right child depending on r_type and r_data
    if node_lst[root_index].r_type == 0:
        right_offset = node_lst[root_index].r_data
    else:
        right_offset = node_lst[root_index].r_data - (length - 1)
    # find offset for left child depending on l_type and l_data
    if node_lst[root_index].l_type == 0:
        left_offset = node_lst[root_index].l_data
    else:
        left_offset = node_lst[root_index].l_data - length

    # make right subtree of current node
    root.right = generate_tree_general(node_lst, right_offset)
    # make left subtree of current node
    root.left = generate_tree_general(node_lst, left_offset)

    # return Huffman Tree
    return root


def decompress_bytes(tree: HuffmanTree, text: bytes, size: int) -> bytes:
    """ Use Huffman tree <tree> to decompress <size> bytes from <text>.

    >>> tree = build_huffman_tree(build_frequency_dict(b'helloworld'))
    >>> number_nodes(tree)
    >>> decompress_bytes(tree, \
             compress_bytes(b'helloworld', get_codes(tree)), len(b'helloworld'))
    b'helloworld'
    """
    # get codes for everything in tree
    codes = get_codes(tree)
    # make inverse mapping of codes
    inv_codes = {v: k for k, v in codes.items()}
    # convert compressed byte to bit
    bits = ''
    for char in list(text):
        bits += byte_to_bits(char)
    # make list for uncompressed item
    uncompressed = []
    # string variable told hold a bit in bits
    txt = ''
    # go through all the bits
    for bit in bits:
        # concatenate txt and bit
        txt += bit
        # if txt is found in inv_codes
        if txt in inv_codes:
            # append it to uncompressed list
            uncompressed.append(inv_codes[txt])
            # reset txt variable
            txt = ''
            if len(uncompressed) == size:
                break
    # convert uncompressed list to bytes and return
    return bytes(uncompressed)


def decompress_file(in_file: str, out_file: str) -> None:
    """ Decompress contents of <in_file> and store results in <out_file>.
    Both <in_file> and <out_file> are string objects representing the names of
    the input and output files.

    Precondition: The contents of the file <in_file> are not empty.
    """
    with open(in_file, "rb") as f:
        num_nodes = f.read(1)[0]
        buf = f.read(num_nodes * 4)
        node_lst = bytes_to_nodes(buf)
        # use generate_tree_general or generate_tree_postorder here
        tree = generate_tree_general(node_lst, num_nodes - 1)
        size = bytes_to_int(f.read(4))
        with open(out_file, "wb") as g:
            text = f.read()
            g.write(decompress_bytes(tree, text, size))


# ====================
# Other functions

def improve_tree(tree: HuffmanTree, freq_dict: dict[int, int]) -> None:
    """ Improve the tree <tree> as much as possible, without changing its shape,
    by swapping nodes. The improvements are with respect to the dictionary of
    symbol frequencies <freq_dict>.

    >>> left = HuffmanTree(None, HuffmanTree(99, None, None), \
    HuffmanTree(100, None, None))
    >>> right = HuffmanTree(None, HuffmanTree(101, None, None), \
    HuffmanTree(None, HuffmanTree(97, None, None), HuffmanTree(98, None, None)))
    >>> tree = HuffmanTree(None, left, right)
    >>> freq = {97: 26, 98: 23, 99: 20, 100: 16, 101: 15}
    >>> avg_length(tree, freq)
    2.49
    >>> improve_tree(tree, freq)
    >>> avg_length(tree, freq)
    2.31
    """
    # made new list for tree variable given
    new_lst = [tree]
    # sort symbols by frequency and put it into lst
    lst = sorted(freq_dict, key=freq_dict.get)
    # counter for lst index(used in loop)
    counter = 0
    # go over everything inside new_lst
    while len(new_lst) != 0:
        # get last node from new_lst
        NODE = new_lst.pop()
        # if it is not a leaf
        if not NODE.is_leaf():
            # append left child
            if NODE.left is not None:
                new_lst.append(NODE.left)
            # append right child
            if NODE.right is not None:
                new_lst.append(NODE.right)
        # if it is a leaf
        else:
            # get the symbol from lst
            NODE.symbol = lst[counter]
            # add 1 to index counter
            counter += 1


if __name__ == "__main__":
    import doctest

    doctest.testmod()

    import python_ta

    python_ta.check_all(config={
        'allowed-io': ['compress_file', 'decompress_file'],
        'allowed-import-modules': [
            'python_ta', 'doctest', 'typing', '__future__',
            'time', 'utils', 'huffman', 'random'
        ],
        'disable': ['W0401']
    })

    mode = input(
        "Press c to compress, d to decompress, or other key to exit: ")
    if mode == "c":
        fname = input("File to compress: ")
        start = time.time()
        compress_file(fname, fname + ".huf")
        print(f"Compressed {fname} in {time.time() - start} seconds.")
    elif mode == "d":
        fname = input("File to decompress: ")
        start = time.time()
        decompress_file(fname, fname + ".orig")
        print(f"Decompressed {fname} in {time.time() - start} seconds.")
