"""
Author: Itay Evron
See https://github.com/ievron/wltls/ for more technical and license information.
"""

class Node:
    def __init__(self, value=None, left = None, right=None):
        self.children =[left, right]
        self.value = value

#############################################################################################
# A binary tree, where every node can hold a value.
# Example usage:
# tree = BinaryTree()
# tree.store([0, 1, 1, 1, 0, 1], "Hello")
# tree.store([0, 1, 1, 1, 0, 1], "Hello!")
# tree.store([0, 1, 1, 1, 0], "World")
#
# print(tree.read([0, 1, 1, 1, 0, 1]))
# >> Hello!
# print(tree.read([0, 1, 1, 1, 0]))
# >> World!
# print(tree.read([0, 1, 0, 0, 0, 1]))
# >> KeyError
#############################################################################################
class BinaryTree:
    def __init__(self):
        self.root = Node()

    def store(self, word, value):
        current = self.root

        for w in word:
            c = (w+1) // 2
            previous = current

            current = current.children[c]

            if current is None:
                current = Node()
                previous.children[c] = current

        current.value = value

    def read(self, word):
        current = self.root

        for w in word:
            c = (w+1) // 2
            current = current.children[c]

            if current is None:
                raise KeyError

        return current.value