xs = [
    [2., 3., -1.],
    [3., -1., 0.5],
    [0.5, 1., 1.],
    [1., 1., -1.]
]
ys = [1., -1., -1., 1.]


from nn import MLP

n = MLP(3, [4, 4, 1])

n.fit(xs, ys)
print()
print(ys)

print(n.predict(xs))
