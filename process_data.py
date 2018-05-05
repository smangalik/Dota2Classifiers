f = open('./dota2Test.csv', 'r')
w = open('./input.txt','w')

for line in f.readlines():
    cur = ''
    s = line.split(',')
    cur += s[0] + ' 1:' + s[1] + ' 2:' + s[2] + ' 3:' + s[3]
    
    for i in range(4, len(s)):
        cur += ' ' + str(i) + ':' + s[i]
    w.write(cur)

f.close()
w.close()

