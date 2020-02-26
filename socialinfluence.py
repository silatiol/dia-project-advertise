import numpy as np
import matplotlib.pyplot as plt

class User:
  def __init__(self, i, max = 50):
    self.state = 'susceptible'
    self.visited = False
    self.connections = []
    self.probabilities = []
    self.is_seed = False
    self.cost = 1.
    self.z = 0.
    self.id = i
    self.degree = 0
    self.max = max
    self.adjacency_features = []
    self.ucb1_estimate_param = []
    self.ts_estimate_param = []

  def connect(self, u, p):
    if(self.max > 0):
      self.connections.append(u)
      self.probabilities.append(p)
      self.adjacency_features.append([.1,.2,.1,.1])
      self.degree += 1
      self.max -= 1

  def set_seed(self):
    self.is_seed = True
    self.state = 'active'

  def update_probabilities(self, lin_comb_params):
      for i in range(self.degree):
          self.probabilities[i] = max(0, min(1, lin_comb_params[0] * self.adjacency_features[i][0] + \
                                      lin_comb_params[1] * self.adjacency_features[i][1] + \
                                      lin_comb_params[2] * self.adjacency_features[i][2] + \
                                      lin_comb_params[3] * self.adjacency_features[i][3]))

class Graph:
  def __init__(self, N, max_conn = 15, g1 = None, g2 = None):
    self.N = N
    self.init_N = 4
    self.max_connection = 5
    self.linear_comb = [.2,.55,.1,.15]

    self.nodes = []
    self.nodes.append(User(0,max_conn))
    
    for i in range(1,N):
      connections = np.random.randint(0,i, np.minimum(np.random.randint(1,self.max_connection + 1), i))
      connections = list(set(connections))
     
      node = User(i,max_conn)
      self.nodes.append(node)
      for j in connections:
        prob = np.sum(np.random.sample(4) * self.linear_comb )
        prob2 = np.sum(np.random.sample(4) * self.linear_comb )
        node.connect(self.nodes[j],prob)
        self.nodes[j].connect(node, prob2)

      if(i == 100 and g1 != None):
        g1.N = 100
        g1.nodes = self.nodes.copy()
      if(i == 1000 and g2 != None ):
        g2.N = 1000
        g2.nodes = self.nodes.copy()

  def generate_live(self, tree, check):
    res = []

    for us in tree:
      us.visited = False

    while(len(check) > 0): 
      node = tree[check.pop(0)]
      if(node.visited == False):
        node.visited = True
        res.append(node.id)
        for i in range(len(node.connections)):
          if(node.connections[i].visited == False and np.random.binomial(1,node.probabilities[i]) == 1):
            check.append(node.connections[i].id)

    return res

  def montecarlo_app(self, seeds, k):
    res = []
    tree = self.nodes.copy()

    for us in tree:
      us.z = 0

    for i in range(k):
      live_active = self.generate_live(self.nodes.copy(), seeds.copy()) #root
      for n in live_active:
        tree[n].z += 1
    
    for i in range(self.N):
      res.append(tree[i].z / k)

    return res

  def best_seeds(self, budget=0, d=0.3, e=0.1, seeds=[]):
    res = []
    tree = self.nodes.copy()

    while (budget > 0):
      increments = []
      for node in tree:
        increments.append(0)
        if(node.id not in seeds and node.cost <= budget):
          new_seeds=seeds.copy()
          new_seeds.append(node.id)
          k = int((1 / (e**2)) * np.log(len(new_seeds)) * np.log(1 / d)) + 1
          inc = sum(self.montecarlo_app(new_seeds, k))
          increments[node.id] = inc
      
      if(np.max(increments) == 0):
        break
      best_id = np.argmax(increments)
      seeds.append(best_id)
      res.append(best_id)
      budget -= tree[best_id].cost
      print(res)

    return res
  
  def init_estimates(self, estimator="ucb1", approach="pessimistic"):
      """Initializes the estimated probabilities of all edges"""
      for i in self.nodes:
          for j in range(i.degree):
              if estimator == "ucb1":
                  if approach == "pessimistic":
                      i.ucb1_estimate_param = [[0, 0, 1, False] for _ in range(i.degree)]
                  elif approach == "optimistic":
                      i.ucb1_estimate_param = [[1, 0, 1, False] for _ in range(i.degree)]
                  elif approach == "neutral":
                      i.ucb1_estimate_param = [[0.5, 0, 1, False] for _ in range(i.degree)]

              elif estimator == "ts":
                  i.ts_estimate_param = [[1, 1] for _ in range(i.degree)]

  def update_estimate(self, id_from, realizations, time=None, estimator="ucb1"):
      """Updates the parameters of each edge of the specified node"""
      if estimator == "ucb1" and time is not None:
          estimate_param = self.nodes[id_from].ucb1_estimate_param
          print(estimate_param)

          for i in range(len(realizations)):
              # if the edge was "stimulated"
              if realizations[i] != -1:
                  # if first sample ever observed, overwrite dummy sample
                  if not estimate_param[i][3]:
                      estimate_param[i][0] = realizations[i]
                      estimate_param[i][3] = True
                  else:
                      # update empirical mean
                      estimate_param[i][0] = (estimate_param[i][0] * estimate_param[i][2] + realizations[i]) / (
                              estimate_param[i][2] + 1)
                      # increase number of samples
                      estimate_param[i][2] += 1

              # update bound
              estimate_param[i][1] = np.sqrt((2 * np.log(time)) / estimate_param[i][2])

      elif estimator == "ts":
          estimate_param = self.nodes[id_from].ts_estimate_param

          for i in range(len(realizations)):
              if realizations[i] != -1:
                  estimate_param[i][0] += realizations[i]
                  estimate_param[i][1] += 1 - realizations[i]

  def update_weights(self, estimator="ucb1", use_features=False, exp_coeff=1, normalize=True):
      """Updates the estimated probabilities of each edge in the graph (weights)"""
      if estimator == "ucb1":
          all_weights = []
          for node in self.nodes:
              for i in range(node.degree):
                  # new weight = sum of empirical mean and exploration coeff. * ucb1 bound
                  new_weight = node.ucb1_estimate_param[i][0] + exp_coeff * \
                                              node.ucb1_estimate_param[i][1]
                  node.probabilities[i] = new_weight
                  all_weights.append(node.probabilities[i])

          # normalize all weights
          if normalize:
              for node in self.nodes:
                  for i in range(node.degree):
                      node.probabilities[i] = (node.probabilities[i] - min(all_weights)) / (
                              max(all_weights) - min(all_weights))
          else:
              for node in self.nodes:
                  for i in range(len(node.probabilities)):
                      if node.probabilities[i] > 1:
                          node.probabilities[i] = 1

      elif estimator == "ts":
          for node in self.nodes:
              for i in range(node.degree):
                  node.probabilities[i] = np.random.beta(a=node.ts_estimate_param[i][0],
                                                              b=node.ts_estimate_param[i][1])

      if use_features:
          self.set_lin_comb_params(self.estimate_features_parameters())
          self.update_probabilities()

  def update_probabilities(self):
    for n in self.nodes:
        n.update_probabilities(self.lin_comb_params)

  def estimate_features_parameters(self):
      dataset_x = []
      dataset_y = []
      for node in self.nodes:
          for i in range(node.degree):
              dataset_x.append(node.connections[i])
              dataset_y.append(node.probabilities[i])

      regression_model = LinearRegression(fit_intercept=False)
      regression_model.fit(X=dataset_x, y=dataset_y)

      return list(regression_model.coef_)

  def seeds_at_time_zero(self, budget):
      seeds = []
      nodes_deg = [i.degree for i in self.nodes]

      while budget > 0 and len(nodes_deg) > 0:
          seed = int(np.argmax(nodes_deg))

          if budget - self.nodes[seed].cost > 0:
              budget -= self.nodes[seed].cost
              seeds.append(seed)

          nodes_deg.pop(seed)

      return seeds, budget

  def prog_cascade(self, seeds):
      explore_next_ids = [s for s in seeds]
      realizations_per_node = []

      for s in seeds:
          self.nodes[s].state = 'active'

      for i in explore_next_ids:
          realizations = []

          print(self.nodes[i].degree)

          for j in range(self.nodes[i].degree):
              adjacent_node_id = self.nodes[i].connections[j].id

              if not self.nodes[adjacent_node_id].state == 'active':
                  realization = np.random.binomial(1, self.nodes[i].probabilities[j])

                  if realization == 1:
                      explore_next_ids.append(adjacent_node_id)
                      self.nodes[adjacent_node_id].state = 'active'

                  realizations.append(realization)

              else:
                  realizations.append(-1)

          realizations_per_node.append([i, realizations])

      nodes_activated = len(explore_next_ids)
      for id in explore_next_ids:
          self.nodes[id].state = 'susceptible'

      return realizations_per_node, nodes_activated

  def get_empirical_means(self):
      means = []
      for i in self.nodes:
          for j in range(i.degree):
              means.append(i.ucb1_estimate_param[j][0])
      return means

  def get_edges(self):
      edges = []
      for i in self.nodes:
          for j in range(len(i.probabilities)):
              edges.append(i.probabilities[j])
      return edges

    


np.random.seed(123)

graph100 = Graph(100, 20)
graph1000 = Graph(1000, 50)
graph10000 = Graph(10000, 100)

graph100_r = Graph(100, 15)
graph1000_r = Graph(1000, 15)
graph10000_r = Graph(10000, 15)

res = []
res.append(0)
j = 0
for i in range(0, graph100.N):
  j += len(graph100.nodes[i].connections)
  res.append(j)

plt.figure()
plt.title("Graph 100 user")
plt.plot(range(0,101), res , color="C1")
plt.xlabel("Users")
plt.ylabel("Connections")
plt.show()

res = []
res.append(0)
j = 0
for i in range(0, graph1000.N):
  j += len(graph1000.nodes[i].connections)
  res.append(j)

plt.figure()
plt.title("Graph 1000 user")
plt.plot(range(0,1001), res , color="C1")
plt.xlabel("Users")
plt.ylabel("Connections")
plt.show()

res = []
res.append(0)
j = 0
for i in range(0, graph10000.N):
  j += len(graph10000.nodes[i].connections)
  res.append(j)

plt.figure()
plt.title("Graph 10000 user")
plt.plot(range(0,10001), res , color="C1")
plt.xlabel("Users")
plt.ylabel("Connections")
plt.show()

res = []
res.append(0)
j = 0
for i in range(0, graph100_r.N):
  j += len(graph100_r.nodes[i].connections)
  res.append(j)

plt.figure()
plt.title("Graph 100 user restricted")
plt.plot(range(0,101), res , color="C1")
plt.xlabel("Users")
plt.ylabel("Connections")
plt.show()

res = []
res.append(0)
j = 0
for i in range(0, graph1000_r.N):
  j += len(graph1000_r.nodes[i].connections)
  res.append(j)

plt.figure()
plt.title("Graph 1000 user restricted")
plt.plot(range(0,1001), res , color="C1")
plt.xlabel("Users")
plt.ylabel("Connections")
plt.show()

res = []
res.append(0)
j = 0
for i in range(0, graph10000_r.N):
  j += len(graph10000_r.nodes[i].connections)
  res.append(j)

plt.figure()
plt.title("Graph 10000 user restricted")
plt.plot(range(0,10001), res , color="C1")
plt.xlabel("Users")
plt.ylabel("Connections")
plt.show()

deltas = [0.9, 0.5, 0.3, 0.1, 0.05, 0.0001]

social_inf = 3.
best_seeds = graph100.best_seeds(social_inf)
print("100 nodes\n")
print(best_seeds)

results = []
for d in deltas:
  k = int((1 / (0.1**2)) * np.log(len(best_seeds)) * np.log(1 / d)) + 1
  lst = graph100.montecarlo_app(best_seeds, k)
  results.append(np.sum(lst))
#plot
plt.figure()
plt.title("Objective Function")
plt.plot(deltas, results , color="C1")
plt.xlabel("Delta")
plt.ylabel("User Influenced")
plt.show()

social_inf = 5.
best_seeds = graph1000.best_seeds(social_inf)
print("1000 nodes\n")
print(best_seeds)

results = []
for d in deltas:
  k = int((1 / (0.1**2)) * np.log(len(best_seeds)) * np.log(1 / d)) + 1
  lst = graph1000.montecarlo_app(best_seeds, k)
  results.append(np.sum(lst))
#plot
plt.figure()
plt.title("Objective Function")
plt.plot(deltas, results , color="C1")
plt.xlabel("Delta")
plt.ylabel("User Influenced")
plt.show()

social_inf = 10.
best_seeds = graph10000.best_seeds(social_inf)
print("10000 nodes\n")
print(best_seeds)

results = []
for d in deltas:
  k = int((1 / (0.1**2)) * np.log(len(best_seeds)) * np.log(1 / d)) + 1
  lst = graph10000.montecarlo_app(best_seeds, k)
  results.append(np.sum(lst))
#plot
plt.figure()
plt.title("Objective Function")
plt.plot(deltas, results , color="C1")
plt.xlabel("Delta")
plt.ylabel("User Influenced")
plt.show()

### Part 3 and 4, currently not working (tried different approaches)
### Either code or logic error 

### This is the latest tested implementation, cannot make it work (inspired by colleague github)

import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed

def run_experiment(approach, repetitions, stimulations, B, delta, use_features=False, verbose=True) -> dict:

    true_graph = Graph(100)
    est_graph = Graph(100)

    est_graph.init_estimates(estimator="ucb1", approach=approach)
    time = 1

    history_cum_error = []
    history_prob_errors = []
    history_of_seeds = []

    budget = 5
    seeds, remainder = true_graph.seeds_at_time_zero(budget)
    
    for i in range(repetitions):
        # Multiple stimulations of the network
        for j in range(stimulations):
            # Witness cascade
            realizations_per_node, nodes_activated = true_graph.prog_cascade(seeds)
            time += 1
            # Update representation (est_graph) based on observations
            for record in realizations_per_node:
              est_graph.update_estimate(record[0], record[1], time=time, estimator="ucb1")

        # Update weights (edges probabilities)
        est_graph.update_weights(estimator="ucb1", use_features=use_features, exp_coeff=B, normalize=False)
        # Find the best seeds for next repetition
        seeds = est_graph.find_best_seeds(initial_seeds=[], delta=delta, budget=budget, verbose=False,
                                          randomized_search=True, randomized_search_number=100)
        # Update performance statistics (seeds selected, probabilities estimation)
        history_of_seeds.append(seeds)
        prob_errors = np.subtract(true_graph.get_edges(), est_graph.get_empirical_means())
        history_prob_errors.append(abs(prob_errors))
        cum_prob_error = np.sum(abs(prob_errors))
        history_cum_error.append(cum_prob_error)
        if verbose:
            print("\n# Repetition: {}\nbest seeds found: {}\ncumulative error: {}".format(i, seeds, cum_prob_error))

    return {"cum_error": history_cum_error, "prob_errors": history_prob_errors, "seeds": history_of_seeds}

def execute(delta_to_run):
    true_g = Graph(100)
    # PARAMETERS
    approach = "pessimistic"
    B = 0.2  
    repetitions = 10 
    stimulations = 100
    delta = delta_to_run 
    num_of_experiments = 3 
    use_features = False

    # CLAIRVOYANT
    clairvoyant_best_seeds = true_g.best_Seeds(initial_seeds=[], d=0.1, verbose=False)
    exp_clairvoyant_activations = sum(true_g.monte_carlo_sampling(1000, clairvoyant_best_seeds))


    total_seeds = []

    # RUN ALL EXPERIMENTS
    results = Parallel(n_jobs=-2, verbose=11)(  # all cpu are used with -1 (beware of lag)
        delayed(run_experiment)(approach, repetitions, stimulations, B, delta, use_features, verbose=True) for _ in
        range(num_of_experiments))  # returns a list of results (each item is a dictionary of results)

    # PLOT CUMULATIVE REGRET (with respect to clairvoyant expected activations)
    exp_rewards = []
    for exp in range(len(results)):  # for each experiment compute list of rewards
        sel_seeds = results[exp]["seeds"]
        exp_rewards.append([sum(graph.montecarlo_app(seeds, 1000)) for seeds in sel_seeds])
    avg_exp_rewards = [sum(x) / len(exp_rewards) for x in zip(*exp_rewards)]

    cum_regret = np.cumsum(np.array(exp_clairvoyant_activations) - avg_exp_rewards)
    plt.plot(cum_regret)
    plt.title("Cumulative activations regret")
    plt.show()

    # PLOT AVG N° OF ACTIVATED NODES
    plt.plot(avg_exp_rewards)
    plt.title("Avg. activated nodes")
    plt.show()


#REPEAT WITH VARIOUS DELTA AND DIFFERENT SETTINGS (UCB, TS, with/without knowledge)
deltas = [0.2]
for d in deltas:
    execute(delta_to_run=d)
