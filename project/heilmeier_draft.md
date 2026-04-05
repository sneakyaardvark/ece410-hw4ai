# Week 1: Heilmeier questions
[The Heilmeier catechism](https://www.darpa.mil/about/heilmeier-catechism)

1. What are you trying to do? Articulate your objectives using absolutely no jargon.
    * I want to design a chip for low-power, embedded use. The chip will process audio data for keyword detection.
2. How is it done today, and what are the limits of current practice?
    * The current approach is "silicon cochlea" models which mimic biological systems. These networks are difficult to train, and overall very complex. It is necessary to find new ways to improve the preprocessing and decrease the number of input channels necessary. Transformer-based models have hit 96% accuracy, but are complex and not energy efficient. There is a trade-off between the accuracy and efficiency.
3. What is new in your approach, and why do you think it will be successful?
    * I am going for efficiency, not necessarily for accuracy. The existing models are large, and looking for attentions on the accuracy leaderboard. I want to make a small, efficient chip. It will be successful because I am going where no one else is to find the improvements no one has been looking for.

