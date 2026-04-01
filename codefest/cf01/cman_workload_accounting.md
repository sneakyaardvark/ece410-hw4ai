1. For each layer: $N_{l} * N_{l - 1}$, where N is the number of neurons in a layer _l_.

Layer 2: $N_2 * N_1 = 784 * 256 = 200704$

Layer 3: $N_3 * N_2 = 256 * 128 = 32768$

Layer 4: $N_4 * N_3 = 128 * 10 = 1280$

2. $MAC_{total} = 200704 + 32768 + 1280 = 234752$
3. The trainable parameters corresponds to the number of connections. In this case, this is equal to the number of MACs for layers 2, 3, and 4.

Parameters $= MAC_{L2} + MAC_{L3} + MAC_{L4} = 200704 + 32768 + 1280 = 234752$

4. Weight memory $= 234752 weights * 4 bytes = 939008 bytes$
5. Total activation memory $= N_{total} * 4 + mem_{input} = (784 + 256 + 128 + 10) * 4 + (784 * 4) = 4712$ bytes
6. Arithmetic intensity $= (2 * MAC_{total} / (mem_{weights} + mem_{activation})) = (2 * 234752) / (939008 + 4712) = 0.497503


