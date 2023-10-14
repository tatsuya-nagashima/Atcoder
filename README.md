# Atcoder Cheatsheet for Python

## 基本操作

### 標準入力

```python
S = input()
N = int(input())
A, B = input().split()
A, B = map(int, input().split())
l = list(map(int, input().split()))
S = [input() for _ in range(H)]
G = [list(map(int, input().split())) for _ in range(H)]
```

### 標準出力
```python
print(*arr)
print(*arr, sep="\n")
print("あああ", end="")
```

### ソート

```python
A = [1, 3, 2]
A.sort()

A = [[1, 2], [3, 1], [2, 5]]
B = sorted(A, key=lambda x: x[0], reverse=False) # 0番目の要素でソート
C = sorted(A, key=lambda x: x[1], reverse=False) # 1番目の要素でソート

from operator import itemgetter
B =[(5,8), (6,10), (7,2),(4,1), (3,11),(9,0)]
print(sorted(B, key = itemgetter(0))) #第1変数で昇順ソートしてる
#[(3, 11), (4, 1), (5, 8), (6, 10), (7, 2), (9, 0)]
print(sorted(B, key = itemgetter(0),reverse=True)) #第1変数で降順ソートしてる
#[(9, 0), (7, 2), (6, 10), (5, 8), (4, 1), (3, 11)]
print(sorted(B, key = itemgetter(1))) #第2変数で昇順ソートしてる
#[(9, 0), (4, 1), (7, 2), (5, 8), (6, 10), (3, 11)]
print(sorted(B, key = itemgetter(1),reverse=True)) #第2変数で降順ソートしてる
#[(3, 11), (6, 10), (5, 8), (7, 2), (4, 1), (9, 0)]
```

### defaultdict
```python
from collections import defaultdict

cnt = defaultdict(int)
for li in l:
  cnt[li] += 1
 
ans = 0  
for k in cnt.keys():
  ans += cnt[k] * (cnt[k] - 1) // 2
```

### deepcopy
```python
from copy import deepcopy
A=[1,2,3]
B=deepcopy(A)
B[1]=100
print(A)
```

## 整数

### 最大公約数・最小公倍数

```python
import math
a = 4
b = 6
print(math.gcd(a, b))
print(math.lcm(a, b))

#a,bの最大公約数
def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

#a,bの最小公倍数
def lcm(a, b):
    return a * b // gcd (a, b)
```

### 約数列挙

```python
# nの約数を全て求める
def divisor(n):
    i = 1
    table = []
    while i * i <= n:
        if n%i == 0:
            table.append(i)
            table.append(n//i)
        i += 1
    table = list(set(table))
    return table
```

### 素因数分解

```python
# nを素因数分解したリストを返す
def prime_decomposition(n):
  i = 2
  table = []
  while i * i <= n:
    while n % i == 0:
      n /= i
      table.append(i)
    i += 1
  if n > 1:
    table.append(n)
  return table
```

### 素数判定

```python
# 引数nが素数かどうかを判定
def is_prime(n):
    for i in range(2, n + 1):
        if i * i > n:
            break
        if n % i == 0:
            return False
    return n != 1
```

### エラトステネスの篩

```python
def sieve(n):
    is_prime = [True for _ in range(n+1)]
    is_prime[0] = False

    for i in range(2, n+1):
        if is_prime[i-1]:
            j = 2 * i
            while j <= n:
                is_prime[j-1] = False
                j += i
    table = [ i for i in range(1, n+1) if is_prime[i-1]]
    return is_prime, table
```

### 冪乗の計算

```python
# xのn乗をmで割った余り
def pos(x, n, m):
    if n == 0:
        return 1
    res = pos(x*x%m, n//2, m)
    if n%2 == 1:
        res = res*x%m
    return res
```

## アルゴリズム・データ構造

### 順列全探索

```python
from itertools import permutations

l = [1, 2, 3, 4, 5, 6]
N = 3

for p in list(permutations(l, N)):
    print(p)
```

### 組み合せ全探索

```python
from itertools import combinations

l = [1, 2, 3, 4, 5, 6]
N = 3

for c in list(combinations(l, N)):
    print(c)
```

### bit 全探索

```python
from itertools import product

N = 3

for p in product((0, 1), repeat=N):
    print(p)
```

### 累積和

```python
from itertools import accumulate

l = [1, 2, 3, 4, 5, 6]

print(list(accumulate(l)))
```

### 二分探索

```python
from bisect import bisect_right, bisect_left, insort_left, insort_right

a = [10, 20, 30, 40, 50]

print(bisect_left(a, 30))
print(bisect_right(a, 30))
```

### 数え上げ

```python
from collections import Counter
a = ['a', 'b', 'c', 'd', 'a', 'a', 'b']
c = Counter(a)
print(c)
print(c['a'])
print(c['b'])

print(list(c.keys()))
print(list(c.values()))
print(list(c.items()))

m = c.most_common()
print(m)
```

### しゃくとり法

```python
from collections import deque

ans = 0
q = deque()
for c in l:
    q.append(c)  # dequeの右端に要素を一つ追加する。
    # 追加した要素に応じて何らかの処理を行う

    while not (満たすべき条件):
        rm = q.popleft() # 条件を満たさないのでdequeの左端から要素を取り除く
        # 取り除いた要素に応じて何らかの処理を行う

    ans = max(ans ,len(q))

print(ans)
```

### 幅優先探索(BFS)

```python
from collections import deque

N, M = map(int, input().split())
G = [[] for _ in range(N)]

for _ in range(M):
    a, b = map(int, input().split())
    G[a].append(b)
    G[b].append(a)

def bfs(u):
    dist = [None] * N
    dist[u] = 0
    que = deque([u])
    while que:
        v = que.popleft()
        for next_v in G[v]:
            if dist[next_v] is None:
                dist[next_v] = dist[v] + 1
                que.append(next_v)
                return dist

d = bfs(0)
print(d)
```

### 深さ優先探索(DFS)
再帰
```python
import sys
sys.setrecursionlimit(10000)

N, M = map(int, input().split())
G = [[] for _ in range(N)]

for _ in range(M):
    a, b = map(int, input().split())
    G[a].append(b)
    G[b].append(a)

def dfs(v):
    if visited[v]: return
    visited[v] = True
    for next_v in G[v]:
        dfs(next_v)
        
visited = [False] * N
dfs(i)
```

非再帰
```python
from collections import deque

N, M = map(int, input().split())
G = [[] for _ in range(N)]

for _ in range(M):
    a, b = map(int, input().split())
    G[a].append(b)
    G[b].append(a)

def dfs(u):
	stack = deque([u])
	while stack:
		v = stack.pop()
		if 条件式:
			処理
			stack.append(next_v)
			
visited = [False] * N
dfs(i)
```
### 優先度付きキュー

```python
import heapq

a = [1, 6, 8, 0, -1]
heapq.heapify(a)

heapq.heappop(a)
heapq.heappush(a, -2)
```

### UnionFind

```python
class UnionFind:
    def __init__(self, n):
        self.par = [i for i in range(n)] #親
        self.rank = [0 for _ in range(n)] #根の深さ

    #xの属する木の根を求める
    def find(self, x):
        if self.par[x] == x:
            return x
        else:
            self.par[x] = self.find(self.par[x])
            return self.par[x]

    #xとyの属する集合のマージ
    def unite(self, x, y):
        x = self.find(x)
        y = self.find(y)
        if x == y:
            return
        if self.rank[x] < self.rank[y]:
            self.par[x] = y
        else:
            self.par[y] = x
            if self.rank[x] == self.rank[y]:
                self.rank[x] += 1

    #xとyが同じ集合に属するかを判定
    def same(self, x, y):
        return self.find(x) == self.find(y)

N, M = map(int, input().split())
uf = UnionFind(N)

for _ in range(M):
    a, b = map(int, input().split())
    uf.unite(a-1, b-1)

ans = -1
for i in range(N):
    if uf.par[i] == i:
        ans += 1
print(ans)
```
