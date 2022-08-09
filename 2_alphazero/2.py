def fn(a):
  while a > 0:
    yield a
    a -= 1


for i in fn(10):
  print(i)
