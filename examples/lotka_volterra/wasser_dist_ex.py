import jax.numpy as jnp
from Wasser_dist import WassersteinDistanceCalculator

# load saved posteriors
prefd = jnp.load('./prefd.npy', allow_pickle=True)
pd1 = jnp.load('./pd1.npy', allow_pickle=True)
pd2 = jnp.load('./pd2.npy', allow_pickle=True)
pd3 = jnp.load('./pd3.npy', allow_pickle=True)
pd4 = jnp.load('./pd4.npy', allow_pickle=True)

# load saved priors
refd = jnp.load('./refd.npy', allow_pickle=True)
d1 = jnp.load('./d1.npy', allow_pickle=True)
d2 = jnp.load('./d2.npy', allow_pickle=True)
d3 = jnp.load('./d3.npy', allow_pickle=True)
d4 = jnp.load('./d4.npy', allow_pickle=True)

# compute the results
wasser = WassersteinDistanceCalculator()

# calcualte the distance between the baseline prior and the other priors
print(f"\nFor Baseline Prior p(0) and p(1)")
wp0p1_perpara = wasser.dist_distribu(refd, d1)
print(f"\nFor Baseline Prior p(0) and p(2)")
wp0p2_perpara = wasser.dist_distribu(refd, d2)
print(f"\nFor Baseline Prior p(0) and p(3)")
wp0p3_perpara = wasser.dist_distribu(refd, d3)
print(f"\nFor Baseline Prior p(0) and p(4)")
wp0p4_perpara = wasser.dist_distribu(refd, d4)

# calculate the distance between the baseline posterior  and the other
# posteriors
print(f"\nFor Baseline Posterior P0 and P1")
WIM01_perpara = wasser.dist_distribu(prefd, pd1)
print(f"\nFor Baseline Posterior P0 and P2")
WIM02_perpara = wasser.dist_distribu(prefd, pd2)
print(f"\nFor Baseline Posterior P0 and P3")
WIM03_perpara = wasser.dist_distribu(prefd, pd3)
print(f"\nFor Baseline Posterior P0 and P4")
WIM04_perpara = wasser.dist_distribu(prefd, pd4)

# calculate the prior scaledWIM
WsIM01_perpara = WIM01_perpara / wp0p1_perpara
WsIM02_perpara = WIM02_perpara / wp0p2_perpara
WsIM03_perpara = WIM03_perpara / wp0p3_perpara
WsIM04_perpara = WIM04_perpara / wp0p4_perpara

print(f"\nThe prior scaled WIM:")
print(
    f'\nThe prior scaledWIM between baseline posterior and P1 posterior  is:\n {WsIM01_perpara}')
print(
    f'\nThe prior scaledWIM between baseline posterior and P2 posterior is:\n {WsIM02_perpara}')
print(
    f'\nThe prior scaledWIM between baseline posterior and P3 posterior is:\n {WsIM03_perpara}')
print(
    f'\nThe prior scaledWIM between baseline posterior and P4 posterior is:\n {WsIM04_perpara}')
