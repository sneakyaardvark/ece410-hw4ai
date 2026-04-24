# CF04 CMAN: Quantization

To calculate the scale factor, S:
```
w_max = np.max(np.abs(w))
s = w_max / 127
```

Max = 2.31

```
Quantized matrix, W_q =
 [[  47.  -66.   19.  115.]
 [  -4.   50. -103.    7.]
 [  85.    2.  -24. -127.]
 [ -10.   57.   42.   30.]]

Dequantized, W_deq = 
[[0.00488189 0.00047244 0.00559055 0.00826772]
 [0.00275591 0.00055118 0.00653543 0.00732283]
 [0.00393701 0.00637795 0.00346457 0.        ]
 [0.00188976 0.00677165 0.00606299 0.00433071]]
```

The element with the largest error is element (0, 3) with value 2.10 and error 0.00827. MAE = 0.00432

With S = S_bad = 0.01, the error increases. MAE = 0.17125. When S is too small, the relative difference between elements decreases, and so the dequantization becomes coarser, with the steps being determined more by the precision of the integer than the original float.
