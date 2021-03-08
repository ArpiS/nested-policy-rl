import matplotlib.pyplot as plt
from cfqi import CFQIagent
from fqi import FQIagent
from transitions import generate_transitions

BEHAVIOR_PATH = "./behavior.pkl"

train_tuples, test_tuples = generate_transitions()

fqi_agent = FQIagent(train_tuples=train_tuples, test_tuples=test_tuples, behavior_path=BEHAVIOR_PATH)
Q_dist = fqi_agent.runFQI(repeats=1)
plt.plot(Q_dist, label= "Vanilla FQI")
plt.xlabel("Iteration")
plt.ylabel("Q Estimate")
plt.legend()
plt.show()


cfqi_agent = CFQIagent(train_tuples=train_tuples, test_tuples=test_tuples, behavior_path=BEHAVIOR_PATH)
Q_dist = cfqi_agent.runFQI(repeats=1)
plt.plot(Q_dist, label= "Contrastive FQI")
plt.xlabel("Iteration")
plt.ylabel("Q Estimate")
plt.legend()
plt.show()


