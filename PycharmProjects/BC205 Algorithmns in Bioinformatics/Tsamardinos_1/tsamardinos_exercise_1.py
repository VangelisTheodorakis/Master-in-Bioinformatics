from numpy import zeros


def manhattan(rows, columns, down_array, right_array):
    s_array = zeros((rows + 1, columns + 1), dtype=int)

    # First column computation
    for row in range(1, rows + 1):
        s_array[row][0] = s_array[row - 1][0] + down_array[row - 1][0]
    
    # First row computation
    for column in range(1, columns + 1):
        s_array[0][column] = s_array[0][column - 1] + right_array[0][column - 1]

    # Compute the interior values.
    for row in range(1, rows + 1):
        for column in range(1, columns + 1):
            # Get the max of each node if you come from up or left
            s_array[row][column] = max(s_array[row - 1][column] + down_array[row - 1][column],
                                       s_array[row][column - 1] + right_array[row][column - 1])

    # Return the last node
    return s_array[rows][columns]


def longest_common_subsequence(sequence1, sequence2):
    # Initialize the array S and iterate through all character of sequence1 and sequence2.
    S = zeros((len(sequence1), len(sequence2)), dtype=int)
    path = zeros((len(sequence1), len(sequence2)), dtype=int)

    # 1 is up 2 is left 3 is diagonal
    for i in range(1,len(sequence1)):
        for j in range(1,len(sequence2)):
            if sequence1[i] == sequence2[j]:
                S[i][j] = S[i-1][j-1]+1
            else:
                S[i][j] = max(S[i-1][j],S[i][j-1])
    # Recover a maximum substring.
            if S[i][j] == S[i-1][j]:
                path[i][j] = 1
            elif S[i][j] == S[i][j - 1]:
                path[i][j] = 2
            elif S[i][j] == S[i-1][j-1] +1:
                path[i][j] = 3
    return S, path


def printLCS(array , sequence, rows,columns, sub):
    if rows == 0 or columns == 0:
        return sub
    if array[rows][columns] == 3:
        printLCS(array, sequence, rows-1, columns-1,sub)
        sub.append(sequence[rows])
    elif array[rows][columns] == 1:
        printLCS(array, sequence, rows-1, columns,sub)
    elif array[rows][columns] == 2:
        printLCS(array, sequence, rows, columns - 1,sub)
    return sub


if __name__ == '__main__':

    # Read the input data.

    down = [[1, 1, 2, 4, 3], [4, 2, 5, 2, 1], [4, 4, 5, 2, 1], [5, 6, 9, 5, 3]]
    right = [[3, 2, 4, 0], [3, 1, 4, 2], [0, 7, 3, 3], [0, 3, 4, 2], [1, 3, 2, 2]]
    n = m = 4

    # Compute and print the answer for Manhattan Tourist Problem .
    max_distance = manhattan(n, m , down, right)
    print(max_distance)

    # Compute and print the answer for Longest Common Subsequence.
    S, path = longest_common_subsequence("AAACTTGG", "ACACTGTGA")
    sub = printLCS(path, "AAACTTGG", 7, 7, [])
    print(sub)

