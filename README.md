# Donors_choose
Exploring donors choose essays using LDA and variants

## Proposal:

I am interested in exploring the DonorsChoose dataset. My motivation for choosing this project is two-fold: first, it is a large open dataset with text features, and I am interested in learning more about text as data; second, I previous taught in a Washington DC Public school and am interest in applications of data science in the public interest and education.

Specifically, I would be interested in trying to build supervised LDA models (see http://www.cs.princeton.edu/~blei/papers/BleiMcAuliffe2007.pdf) to predict the probability a project is funded given a draft essay and other features. The target in my model would be binary variable funding status, and the features would be constructed from the draft essay. I would plan on (i) learning a set of supervised LDA models (having segmented on project type and school poverty level), (ii) then deploying these models in an app that allows a user to input a draft essay and returns an estimate of the probability the project is funded. In addition, the app would reveal latent topics in funded/unfunded project essays and thus give some direction to future proposals.
