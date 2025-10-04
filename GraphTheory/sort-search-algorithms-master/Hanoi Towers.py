#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""


@authors: Josep, Miruna, Anna, Laura

"""


# Function to solve the Hanoi Tower of n elements and moving the disks from rod source
# to rod destination making use of the auxiliary rod
def hanoitower(n, source, destination, auxiliary):
    if n == 1:
        print(f"Move disk 1 from {source} to {destination}")
        return 1  # One move
    else:
        # Recursive case: move top n-1 disks to auxiliary
        moves = hanoitower(n - 1, source, auxiliary, destination)
        # Move the nth disk to the destination
        print(f"Move disk {n} from {source} to {destination}")
        moves += 1
        # Move the n-1 disks from auxiliary to destination
        moves += hanoitower(n - 1, auxiliary, destination, source)
        return moves

# Main

# Define the number of disks
n = 1
print("The movements required to solve the Hanoi Tower with", n, "disks are:")

# Call the recursive algorithm
hanoitower(n, 'Rod 1', 'Rod 3', 'Rod 2')