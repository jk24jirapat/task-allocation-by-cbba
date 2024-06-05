import numpy as np
import math
import time
import itertools


def heading_adv(theta_i, theta_t):
    if abs(theta_i) < math.pi / 2 and abs(theta_t) < math.pi / 2:
        return 1 - abs(theta_i - theta_t) / math.pi
    return 0.001


def vel_adv(vi, vt):
    if vi > vt:
        return 1
    elif 0.5 * vt <= vi < vt:
        return vi / vt
    elif vi < 0.5 * vt:
        return 0.1
    else:
        raise ValueError("Unhandled case in velocity advantage calculation")


def dist_adv(d):
    return 2 / (1 + math.exp(d))


def overall_adv(Dd, Dv, Dtheta):
    weights = [0.5, 0.4, 0.1]
    return weights[0] * Dd + weights[1] * Dv + weights[2] * Dtheta


def generate_probs(theta_is, theta_ts, vis, vts, ds):
    m, n = theta_is.shape
    p = np.empty([m, n])
    for i in range(m):
        for j in range(n):
            Dd = dist_adv(ds[i, j])
            Dv = vel_adv(vis[i], vts[j])
            Dtheta = heading_adv(theta_is[i, j], theta_ts[i, j])
            p[i, j] = overall_adv(Dd, Dv, Dtheta)
    return p


def calc_profit(V, W, p, assignment):
    m, n = len(W), len(V)
    total_profit = 0
    for j in range(n):
        sub_profit = V[j]
        for i in range(m):
            sub_profit *= (1 - p[i][j])**assignment[i][j]
        total_profit += sub_profit
    return total_profit


def greedy_WTA(V, W, p, timeout):
    m, n = len(W), len(V)
    assignment = np.zeros((m, n))
    start = time.perf_counter()
    for i in range(m):
        for k in range(W[i]):
            profits = [
                V[j] * p[i][j] * np.prod([(1 - p[a][j])**assignment[a][j]
                                          for a in range(m)]) for j in range(n)
            ]
            max_ind = np.argmax(profits)
            assignment[i][max_ind] += 1
            if time.perf_counter() - start >= timeout:
                print(
                    "Warning: Time exceeded timeout threshold for Greedy WTA.")
                return assignment, calc_profit(V, W, p, assignment)
    return assignment, calc_profit(V, W, p, assignment)


def getWays(num, n):

    def subset_sum(n, target, all_ways, partial=[]):
        if target == 0 and n == 0:
            all_ways.append(partial)
            return
        elif target < 0 or n == 0:
            return
        for i in range(target + 1):
            subset_sum(n - 1, target - i, all_ways, partial + [i])

    all_ways = []
    subset_sum(n, num, all_ways)
    return all_ways


def brute_force_WTA(V, W, p, timeout):
    m, n = len(W), len(V)
    start = time.perf_counter()
    assignments_by_inter = [getWays(N, n) for N in W]
    all_combinations = list(itertools.product(*assignments_by_inter))
    all_combinations = [
        np.array(assignment).reshape(-1, n) for assignment in all_combinations
    ]  # Ensure the shape is (m, n)
    best_score = float('inf')
    best_assignment = None
    for assignment in all_combinations:
        score = calc_profit(V, W, p, assignment)
        if score < best_score:
            best_score = score
            best_assignment = assignment
        if time.perf_counter() - start >= timeout:
            print(
                "Warning: Time exceeded timeout threshold for Brute Force WTA."
            )
            return best_assignment, best_score
    return best_assignment, best_score


def cbba(V, W, p, timeout):
    num_drones = len(W)
    num_tasks = len(V)
    bundles = {i: [] for i in range(num_drones)}
    task_owners = np.full(num_tasks, -1)  # -1 indicates no owner
    task_scores = np.copy(
        p)  # Use the precomputed probabilities as task scores

    start = time.perf_counter()
    max_iterations = 100  # Adjust based on required convergence

    for iteration in range(max_iterations):
        changes = False
        for drone in range(num_drones):
            best_task = None
            best_score = -1
            # Check each task to see if it can be added to the bundle
            for task in range(num_tasks):
                # A drone can consider a task if it has no owner or it can provide a better score than the current owner
                if (task_owners[task] == -1
                        or (task_scores[drone, task]
                            > task_scores[task_owners[task], task]
                            and len(bundles[task_owners[task]]) > 0)):
                    if task_scores[drone, task] > best_score:
                        best_task = task
                        best_score = task_scores[drone, task]

            # Assign the best task found, if any
            if best_task is not None and (task_owners[best_task] == -1
                                          or task_scores[drone, best_task]
                                          > task_scores[task_owners[best_task],
                                                        best_task]):
                if task_owners[best_task] != -1:
                    bundles[task_owners[best_task]].remove(best_task)
                if len(bundles[drone]) < W[drone]:  # Check drone's capacity
                    bundles[drone].append(best_task)
                    task_owners[best_task] = drone
                    changes = True

        if not changes:
            break  # Stop if no changes in an iteration, consensus reached

        # Check timeout
        if time.perf_counter() - start >= timeout:
            print("Warning: Time exceeded timeout threshold for CBBA.")
            break

    return bundles


""" Example usage for WTA Algorithm"""
if __name__ == "__main__":
    timeout = 120
    V = [1, 1, 1, 1]
    W = [1, 1, 1, 1, 1]
    p = np.array([[0.99, 0, 0.1, 0.01], [0.99, 0, 0.1, 0.01],
                  [0.99, 0, 0.1, 0.01], [0.99, 0, 0.1, 0.01],
                  [0.99, 0, 0.1, 0.01]])

    greedy1 = greedy_WTA(V, W, p, timeout)
    brute1 = brute_force_WTA(V, W, p, timeout)
    cbba1 = cbba(V, W, p, timeout)
    print("\nGreedy Assignment:", greedy1[0], "\nObjective Value:", greedy1[1])
    print("\nBrute Force Assignment:", brute1[0], "\nObjective Value:",
          brute1[1])
    print("\nCBBA Bundles:", cbba1)

    # Analysis: This algorithm will give up on intercepting targets with a low
    # corresponding p value

    V = [5, 10, 20]
    W = [5, 2, 1]
    # p: columns = num of targets, rows = num of interceptor types
    p = np.array([[0.3, 0.2, 0.5], [0.1, 0.6, 0.5], [0.4, 0.5, 0.4]])

    greedy2 = greedy_WTA(V, W, p, timeout)
    brute2 = brute_force_WTA(V, W, p, timeout)
    cbba2 = cbba(V, W, p, timeout)
    print("\nGreedy Assignment:", greedy2[0], "\nObjective Value:", greedy2[1])
    print("\nBrute Force Assignment:", brute2[0], "\nObjective Value:",
          brute2[1])
    print("\nCBBA Bundles:", cbba2)

    V = [16, 29, 43, 43]
    W = [6, 6, 7, 6]
    p = np.array([[0.36, 0.71, 0.96, 0.91], [0.96, 0.25, 0.79, 0.68],
                  [0.75, 0.14, 0.23, 1.], [1, 0.43, 0.31, 0.97]])

    greedy3 = greedy_WTA(V, W, p, timeout)
    #brute3 = brute_force_WTA(V, W, p)
    cbba3 = cbba(V, W, p, timeout)
    print("\nGreedy Assignment:", greedy3[0], "\nObjective Value:", greedy3[1])
    # print("\nBrute Force Assignment:", brute3[0], "\nObjective Value:", brute3[1])
    print("\nCBBA Bundles:", cbba3)

    V = [7, 2]
    W = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    p = np.array([[0.85, 0.22], [0.32, 0.85], [0.88, 0.14], [0.42, 0.3],
                  [0.27, 0.23], [0.48, 0.78], [1., 1.], [0.57, 0.96],
                  [0.04, 0.75], [0.83, 0.86]])

    greedy4 = greedy_WTA(V, W, p, timeout)
    brute4 = brute_force_WTA(V, W, p, timeout)
    cbba4 = cbba(V, W, p, timeout)
    print("\nGreedy Assignment:", greedy4[0], "\nObjective Value:", greedy4[1])
    print("\nBrute Force Assignment:", brute4[0], "\nObjective Value:",
          brute4[1])
    print("\nCBBA Bundles:", cbba4)
