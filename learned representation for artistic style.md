https://arxiv.org/pdf/1610.07629.pdf

Ulyanov et al. (2016a), Li & Wand (2016) and Johnson et al. (2016) tackle this problem by introducing a feedforward style transfer network, which is trained to go from content to pastiche image in one pass.

The neural algorithm of artistic style proposes the following definitions:

• Two images are similar in content if their high-level features as extracted by a trained
classifier are close in Euclidian distance.

• Two images are similar in style if their low-level features as extracted by a trained classifier
share the same statistics or, more concretely, if the difference between the features’ Gram
matrices has a small Frobenius norm

-> Train a single conditional style transfer network T(c, s) for N styles.

to model a style, it is sufficient to specialize scaling and shifting parameters after normalization to each specific style.
