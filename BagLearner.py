import numpy as np

class BagLearner(object):
    def __init__(self, learner=None, kwargs=None, bags=10, boost=False, verbose=False):
        if not kwargs:
            kwargs = {}
        self.learners = [learner(**kwargs) for _ in range(0, bags)]
        self.bags = bags
        self.boost = boost
        self.verbose = verbose
        self.kwargs = kwargs

    def author(self):
        return 'narora62'

    def add_evidence(self, Xdata, Ydata):
        data = np.concatenate((Xdata, Ydata[:, None]), axis=1)
        np.random.shuffle(data)

        for learner in self.learners:
            idx = np.random.choice(Xdata.shape[0], Ydata.shape[0])
            learner.add_evidence(Xdata[idx], Ydata[idx])

            if self.verbose:
                print(f"BagLearner: Learner {self.learners[idx]} built successfully")

    def query(self, points):
        output_query_list = []
        for learner in self.learners:
            output_query_list.append(learner.query(points))
            if self.verbose:
                print(f"BagLearner: Learner {learner} built successfully")

        return np.mean(output_query_list, axis=0)