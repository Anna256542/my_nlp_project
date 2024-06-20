import re
res=re.search("n.*?w","newnewnew")
print("非贪婪模式：",res.group())
res=re.search("n.*w","newnewnew")
print("贪婪模式：",res.group())

import re

# 使用re.search()函数
result_search = re.search(r'is', 'This is a test string')
print(result_search.group(0))

# 使用re.match()函数
result_match = re.match(r'is', 'This is a test string')
if result_match:
    print(result_match.group(0))
else:
    print("No match found")

# 使用re.findall()函数
result_findall = re.findall(r'\w+', 'This is a test string')
print(result_findall)

# 使用re.sub()函数
result_sub = re.sub(r'test', 'sample', 'This is a test string')
print(result_sub)

# 使用re.split()函数
result_split = re.split(r'\s', 'This is a test string')
print(result_split)

