# Heilmeier Questions and Answers
[The Heilmeier catechism](https://www.darpa.mil/about/heilmeier-catechism)

1. What are you trying to do? Articulate your objectives using absolutely no jargon.
    * I am designing a chip for embedded keyword recognition. It will accelerate the processing of spikes for a spiking neural network. This will be tested with the Heidelberg dataset, which is a series of 10,000 1.4 second samples of spoken digits 0-9.
2. How is it done today, and what are the limits of current practice?
    * Currently, the processing is done using large models will single precision floating point, offering high accuracy but not speed or efficiency. Transformer-based models have hit 96% accuracy, but are complex and not energy efficient. My profiling shows the matrix multiplication dominating, as it does in most all machine learning workloads. This can be massively improved using speciality hardware, as seen with the explosive usage of GPUs and TPUs for most machine learning computation.
3. What is new in your approach, and why do you think it will be successful?
    * I am going for efficiency, not necessarily for accuracy. The existing models are large, and looking for attentions on the accuracy leaderboard. I want to make a small, efficient chip. My approach will be to quantize the weights and shrink the area needed for MAC units. The throughput of integer arithmetic will be higher than floating point and allow for better power efficiency.

