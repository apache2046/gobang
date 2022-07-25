def fn(a):
  r = a
  for _ in range(5):
    r = yield r*2
    print('fn', r)

def fn2(a):
  yield from fn(a)
  return 33

def main():
  g = fn(3)
  a = next(g)
  print('m1', a)
  while True:
    try:
      a = g.send(a*3)
      print('m2', a)
    except StopIteration:
      print('fin')
      break
   
  g = fn2(3)
  a = next(g)
  print('m1', a)
  while True:
      a = g.send(a*3)
      print('m2', a)
  


if __name__ == '__main__':
  main()
