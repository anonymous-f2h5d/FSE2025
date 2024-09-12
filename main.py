from patterns import Patterns
from releaseQuality import ReleaseQuality
from clustering import Clustering

# RQ 1
clustering = Clustering()
clustering.cluster_and_analyze()
#
# RQ 2
releaseQuality = ReleaseQuality()
releaseQuality.pattens_quality()

# RQ 3
patterns = Patterns()
patterns.print_transition_matrix()
patterns.distributions_in_different_stages()
releaseQuality.switches_quality()
