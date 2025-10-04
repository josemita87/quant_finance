def lcs(X, Y):
    # Create a 2D table to store the lengths of longest common subsequence
    m = len(X)
    n = len(Y)
    
    # Create a (m+1)x(n+1) matrix initialized to 0
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # Fill the dp table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if X[i - 1] == Y[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1  # Characters match, increment the value
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])  # Take the maximum of excluding one character
    
    # The length of the LCS is stored in dp[m][n]
    lcs_length = dp[m][n]

    # Reconstruct the LCS string from the dp table
    lcs_string = []
    i, j = m, n
    while i > 0 and j > 0:
        if X[i - 1] == Y[j - 1]:
            lcs_string.append(X[i - 1])
            i -= 1
            j -= 1
        elif dp[i - 1][j] >= dp[i][j - 1]:
            i -= 1
        else:
            j -= 1
    
    # The LCS is built in reverse order, so we reverse it
    lcs_string.reverse()
    
    return lcs_length, ''.join(lcs_string)


#Modify strings X and Y to test the function
X = "ABCBDAB"
Y = "BDCAB"
length, subsequence = lcs(X, Y)
print("Length of LCS:", length)
print("LCS:", subsequence)