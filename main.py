# s1 = "ab"
# s2 = "ab"

# s1 = "ab"
# s2 = "ba"

# s1 = "abdfghjkl"
# s2 = "ghjklabdf"

# s1 = "asdfgh"
# s2 = "fghjkl"

# s1 = "asdfghjkl"
# s2 = ""

s1 = "asdfghjkl"
s2 = "asdfghjklasd"

# print(list(range(1,4)))

glob_max = 0
start = 0
stop = 0
for i in range(len(s1)):
    for j in range(len(s2)):
        loc_max = 0
        if s1[i] == s2[j]:
            for k in range(j, len(s2)):
                if i + k - j < len(s1) and s1[i + (k - j)] == s2[k]:
                    loc_max += 1
                    if k == (len(s2) - 1) and glob_max < loc_max:
                        glob_max = loc_max
                        start = j
                        stop = k
                else:
                    if glob_max < loc_max:
                        glob_max = loc_max
                        start = j
                        stop = k - 1
                    break

print(start)
print(stop)
print(s2[start:stop+1])
